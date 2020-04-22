#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cublas.h>
#include <time.h>

//define N
//define threads/block


//FILE IO RELATED
//max number of lines in the training dataset
#define MAX_ROWS_TRAINING 17012
// max number of columns/features in the training dataset
#define MAX_COLUMNS_TRAINING 26
// max number of rows in the testing dataset
#define MAX_ROWS_TESTING 4252
// max number of columns in the testing data
#define MAX_COLUMNS_TESTING 26
//max number of characters/line
#define MAX_CHAR 300

__constant__ int features = 26;

__global__ void logistic_func(float* log_func_v, float* betas, float* data, int n_rows) {
    int row_index = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0;
    for(int j = 0; j < features; j++) {
        temp += betas[j] + data[(row_index * features) + j];
    }
    log_func_v[row_index] = 1.0 / (1.0 + expf(-1.0 * temp))
}

__global__ void cost_func(float* total, float* betas, float* data, int* yvec,
    float* log_func_v, int n_rows) {
        float local_total = 0.0f;
        int feature_index = blockIdx.x * blockDim.x + threadIdx.x;
        for(int i = 0; i < n_rows; i++) {
            float step1 = yvec[i] * logf(log_func_v[i]);
            float step2 = (1 - yvec[i]) * logf(1 - log_func_v[i]);
            local_total += step1 - step2;
        }
        *total = local_total;
}

__device__ void extract_yvec(float** data, int* yvec, int n_rows) {
    for(int i = 0; i < n_rows; i++) {
        yvec[i] = data[i][0]; // extract predictor
        data[i][0] = 1; // pads data
    }
}

__device__ void relabel_yvec(int* yvec, int n_rows, int modelID, int n_models) {
    float high = 365.0f / n_models * (modelID + 1);
    float low = 365.0f / n_models * (modelID - 1);
    for(int i = 0; i < n_rows; i++) {
        int val = yvec[i];
        if(val >= low && val < high) {
            val = 1;
        } else {
            val = 0;
        }
        yvec[i] = val;
    }
}

bool LoadCSV(float** data, char* filename, int pRows, int pCols) {
    //assumed file is in same folder, also rename file here
    FILE *file;
    file = fopen(filename, "r+");
    if(!file) {
        printf("Can't open file \n");
        return false;
    }
    // unparsed data straight from file
    char unparsed_data[pRows+10][MAX_CHAR+10];
    char copying[pRows+10];

    //keep track which row we are on
    int ltracker=0;
    while(fgets(copying, sizeof(copying)-10, file) != 0){
        //copying the line from temp to our array
        strncpy(unparsed_data[ltracker], copying, MAX_CHAR);
        ltracker++;
    }
    //closing the file
    fclose(file);
    char* col_val;
    const char deli[2]=","; // delimiter
    // parses each value in each column per row
    for (int row = 0; row < pRows; row++){
        col_val = strtok(unparsed_data[row], deli);
        for(int col  = 0; col < pCols; col++) {
            col_val = strtok(NULL, deli);
            if(col_val != NULL) {
                data[row][col] = atof(col_val);
            }
        }
    }
    return true;
}



//on the cpu
int main(void){
//things on host:testing and training data
    // array that holds all converted training data
    float **training_data = (float **) malloc(MAX_ROWS_TRAINING * sizeof(float *));
    for(int i = 0; i < MAX_ROWS_TRAINING; i++) {
        training_data[i] = (float *) malloc(MAX_COLUMNS_TRAINING * sizeof(float));
    }
    printf("Loading training data. \n");
    char* training_data_file = "training_data.csv";
    if(LoadCSV(training_data, training_data_file, MAX_ROWS_TRAINING, MAX_COLUMNS_TRAINING)) {
        printf("Training data loaded. \n");
    } else {
        printf("Failed to load training data from %s. \n", training_data_file);
        return 0;
    }

    //array that holds all converted testing data & alloc space for host and setup input values
    float **testing_data = (float **) malloc(MAX_ROWS_TESTING * sizeof(float *));
    for(int i = 0; i < MAX_ROWS_TESTING; i++) {
        testing_data[i] = (float *) malloc(MAX_COLUMNS_TESTING * sizeof(float));
    }
    printf("Loading testing data. \n");
    char* testing_data_file = "testing_data.csv";
    if(LoadCSV(testing_data, testing_data_file, MAX_ROWS_TESTING, MAX_COLUMNS_TESTING)) {
        printf("Testing data loaded. \n");
    } else {
        printf("Failed to load testing data from %s. \n", testing_data_file);
        return 0;
    }
    printf("Actual logistic regression. \n");


    return 0;
}
