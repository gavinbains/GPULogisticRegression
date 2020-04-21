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

__global__ void grad_desc(){

}

__global__ void extract(){

}


__global__ void relabel(){

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
    if(LoadCSV(training_data, "training_data.csv", MAX_ROWS_TRAINING, MAX_COLUMNS_TRAINING)) {
        for(int i = 0; i < MAX_ROWS_TRAINING; i++) {
            printf("Row %i: ", i );
            for(int j = 0; j < MAX_COLUMNS_TRAINING; j++) {
                printf("%f, ", training_data[i][j]);
            }
            printf("\n");
            break;
        }
    }
    printf("Training data loaded. \n");

    //array that holds all converted testing data & alloc space for host and setup input values
    float **testing_data = (float **) malloc(MAX_ROWS_TESTING * sizeof(float *));
    for(int i = 0; i < MAX_ROWS_TESTING; i++) {
        testing_data[i] = (float *) malloc(MAX_COLUMNS_TESTING * sizeof(float));
    }
    printf("Loading testing data. \n");
    if(LoadCSV(testing_data, "testing_data.csv", MAX_ROWS_TESTING, MAX_COLUMNS_TESTING)) {
        for(int i = 0; i < MAX_ROWS_TESTING; i++) {
            printf("Row %i: ", i );
            for(int j = 0; j < MAX_COLUMNS_TESTING; j++) {
                printf("%f, ", testing_data[i][j]);
            }
            printf("\n");
            break;
        }
    }
    printf("Testing data loaded. \n");


    //array for the betas in all of us
    float* betas = new float[MAX_COLUMNS_TRAINING];

//things to device: copy of said testing, training data, and beta
    float **training_devi = (float **) malloc(MAX_ROWS_TRAINING * sizeof(float *));
    float **testing_devi = (float **) malloc(MAX_ROWS_TESTING * sizeof(float *));
    float* betas_devi = new float[MAX_COLUMNS_TRAINING];

    //used to set size of components
    int train_size = MAX_ROWS_TRAINING * MAX_COLUMNS_TRAINING * sizeof(float);
    int test_size = MAX_ROWS_TESTING * MAX_COLUMNS_TESTING * sizeof(float);
    int beta_size = MAX_COLUMNS_TRAINING * sizeof(float);
    //alloc space for device, copies of above
    cudaMalloc((void***)&training_devi, train_size);
    cudaMalloc((void***)&testing_devi, test_size);
    cudaMalloc((void**)& betas_devi, beta_size);

    //copy inputs to device
    cudaMemcpy(training_devi, training_data, train_size, cudaMemcpyHostToDevice);
    cudaMemcpy(testing_devi, testing_data, test_size, cudaMemcpyHostToDevice);
    cudaMemcpy(betas_devi, betas, beta_size, cudaMemcpyHostToDevice);

    //launch on gpu


    //copy result to host
    cudaMemcpy(betas, betas_devi, beta_size, cudaMemcpyDeviceToHost);


    //cleanup all the frees
    //cleaning up on host end
    for(int i = 0; i < MAX_ROWS_TESTING; i++) {
      free(testing_data[i]);
    }
    free(testing_data);

    for(int i = 0; i < MAX_ROWS_TRAINING; i++) {
      free(training_data[i]);
    }
    free(training_data);

    free(betas);

    //cleaning up on device end
    for(int i = 0; i < MAX_ROWS_TESTING; i++) {
      free(testing_devi[i]);
    }
    free(testing_devi);

    for(int i = 0; i < MAX_ROWS_TRAINING; i++) {
      free(training_devi[i]);
    }
    free(training_devi);

    free(betas_devi);



    return 0;
}
