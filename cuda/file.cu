#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cublas.h>
#include <time.h>

//FILE IO RELATED
//max number of lines in the training dataset
#define MAX_ROWS_TRAINING 16896
// max number of columns/features in the training dataset
#define MAX_COLUMNS_TRAINING 26
// max number of rows in the testing dataset
#define MAX_ROWS_TESTING 4096
// max number of columns in the testing data
#define MAX_COLUMNS_TESTING 26
//max number of characters/line
#define MAX_CHAR 300

__constant__ int features = 26;
__constant__ int num_rows = 16896;

// parallelized across the rows
__global__ void logistic_func(float* log_func_v, float* betas, float* data) {
    int row_index = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0;
    for(int j = 0; j < features; j++) {
        temp += betas[j] * data[(row_index * features) + j];
    }
    log_func_v[row_index] = 1.0 / (1.0 + expf(-1.0 * temp));
}

// parallelized across the features
__global__ void log_gradient(float* log_func_v,  float* gradient, float* betas,
    float* data, int* yvec) {
    // the logistic function itself has been pulled out
    int feature_index = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0f;
    for(int i = 0; i < num_rows; i++) {
        temp += (log_func_v[i] - yvec[i]) * data[(i * features) + feature_index];
    }
    gradient[feature_index] = temp;
}

__host__ void extract_yvec(float** data, int* yvec, int n_rows) {
    for(int i = 0; i < n_rows; i++) {
        yvec[i] = data[i][0]; // extract predictor
        data[i][0] = 1; // pads data
    }
}

__host__ void relabel_yvec(int* yvec, int n_rows, int modelID, int n_models) {
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

__host__ void initialize_betas(float* betas) {
    for(int i = 0; i < MAX_COLUMNS_TRAINING; i++) {
        betas[i] = 0.0f;
    }
}

//note: removed cost as a parameter

__host__ void grad_desc( int* yvec, float* betas, float* data, float lr, int max_iters) {
    // n_rows = MAX_ROWS_TRAINING, features defined for both GPU and CPU in constants
    // GPU memory allocation
    float* gpu_gradient;
    float* gpu_betas;
    float* gpu_data;
    int* gpu_yvec;
    float* gpu_log_func_v;
    // allocate memory onboard the GPU
    cudaMalloc((void**) &gpu_gradient, sizeof(float) * MAX_COLUMNS_TRAINING);
    cudaMalloc((void**) &gpu_betas, sizeof(float) * MAX_COLUMNS_TRAINING);
    cudaMalloc((void**) &gpu_data, sizeof(float) * MAX_COLUMNS_TRAINING * MAX_ROWS_TRAINING);
    cudaMalloc((void**) &gpu_yvec, sizeof(int) * MAX_ROWS_TRAINING);
    cudaMalloc((void**) &gpu_log_func_v, sizeof(float) * MAX_ROWS_TRAINING);
    // upload data and yvec; these properties do not change and thus do not need
    // to be reuploaded on each iteration!
    cudaMemcpy(gpu_data, data, sizeof(float) * MAX_COLUMNS_TESTING * MAX_ROWS_TESTING, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_yvec, yvec, sizeof(int) * MAX_ROWS_TRAINING, cudaMemcpyHostToDevice);

    float* gradient = (float*) malloc(sizeof(float) * MAX_COLUMNS_TRAINING);
    for(int i = 0; i < max_iters; i++) {

        // upload beta, data, and yvec values into the GPU
        cudaMemcpy(gpu_betas, betas, sizeof(float) * MAX_COLUMNS_TESTING, cudaMemcpyHostToDevice);
        // launch logistic_func kernel
        logistic_func<<</*for now!*/33, 512>>>(gpu_log_func_v, gpu_betas,
            gpu_data);
        cudaDeviceSynchronize();
        log_gradient<<</*for now!*/1, MAX_COLUMNS_TRAINING>>>(gpu_log_func_v,
            gpu_gradient, gpu_betas, gpu_data, gpu_yvec);
        // copy new gradient values
        cudaMemcpy(gradient, gpu_gradient, sizeof(float) * MAX_COLUMNS_TRAINING, cudaMemcpyDeviceToHost);
        // update betas
        for(int b = 0; b < MAX_COLUMNS_TRAINING; b++) {
            betas[b] -= lr * gradient[b];
        }
    }
    // free all your memory

    free(gradient);
    cudaFree(gpu_gradient);
    cudaFree(gpu_betas);
    cudaFree(gpu_data);
    cudaFree(gpu_yvec);
    cudaFree(gpu_log_func_v);
}

bool LoadCSV(float** data, char* filename, int actual_rows, int parsed_rows, int pCols) {
    //assumed file is in same folder, also rename file here
    FILE *file;
    file = fopen(filename, "r+");
    if(!file) {
        printf("Can't open file \n");
        return false;
    }
    // unparsed data straight from file
    char unparsed_data[actual_rows+10][MAX_CHAR+10];
    char copying[actual_rows+10];

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
    for (int row = 0; row < parsed_rows; row++){
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

__host__ void linearizeArray(float** input, float* output,  int row, int col) {
  for(int i=0; i < row; i++){
    for(int j=0; j<col; j++){
        output[i*col+j] = input[i][j];
    }
  }
}


//TODO predict
__host__ float predict(float* betas, float* data, int* yvec) {
    float* log_func_v = (float*) malloc(sizeof(float) * MAX_ROWS_TESTING);

    float* gpu_log_func_v;
    float* gpu_test_data;
    float* gpu_betas;
    cudaMalloc((void**)&gpu_log_func_v, sizeof(float) * MAX_ROWS_TESTING);
    cudaMalloc((void**)&gpu_test_data, sizeof(float) * MAX_COLUMNS_TESTING * MAX_ROWS_TESTING);
    cudaMalloc((void**)&gpu_betas, sizeof(float) * MAX_COLUMNS_TESTING);

    cudaMemcpy(gpu_test_data, data, sizeof(float) * MAX_COLUMNS_TESTING * MAX_ROWS_TESTING, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_betas, betas, sizeof(float) * MAX_COLUMNS_TESTING, cudaMemcpyHostToDevice);
    logistic_func<<<8, 512>>>(gpu_log_func_v, gpu_betas, gpu_test_data);
    // copy back down the log_func_v
    cudaMemcpy(log_func_v, gpu_log_func_v, sizeof(float) * MAX_ROWS_TESTING, cudaMemcpyDeviceToHost);

   //all the frees
   cudaFree(gpu_log_func_v);
   cudaFree(gpu_betas);
   cudaFree(gpu_test_data);


    float threshold_step = 0.1;
    float optimal_correct = 0;
    for(int t = 0; t <= 10; t++) {
        float threshold = threshold_step * t;
        float correct = 0.0;
        for(int i = 0; i< MAX_ROWS_TESTING; i++){
            if((yvec[i] == 0 && log_func_v[i] <= threshold)||(yvec[i] == 1 && log_func_v[i] > threshold)){
                correct++;
            }
        }
        float percent_correct = correct/MAX_ROWS_TESTING * 100;
        if(percent_correct > optimal_correct){
            optimal_correct = percent_correct;
        }
    }
    free(log_func_v);

    return optimal_correct;
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
    if(LoadCSV(training_data, training_data_file, 17012, MAX_ROWS_TRAINING, MAX_COLUMNS_TRAINING)) {
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
    if(LoadCSV(testing_data, testing_data_file, 4252, MAX_ROWS_TESTING, MAX_COLUMNS_TESTING)) {
        printf("Testing data loaded. \n");
    } else {
        printf("Failed to load testing data from %s. \n", testing_data_file);
        return 0;
    }
    printf("Actual logistic regression. \n");
    int modelID = 1, n_models = 2;
    float lr = 0.01;
    int max_iters = 10000;
    printf("--Extracting and re-labelling train predictor...");
    int* yvec = (int * ) malloc(sizeof(int) * MAX_ROWS_TRAINING);
    extract_yvec(training_data, yvec, MAX_ROWS_TRAINING);
    relabel_yvec(yvec, MAX_ROWS_TRAINING, modelID, n_models);
    printf(" done!-- \n");

    // linearize the data
    float * train_final = (float *) malloc(MAX_COLUMNS_TRAINING * MAX_ROWS_TRAINING * sizeof(float));
    linearizeArray(training_data, train_final, MAX_ROWS_TRAINING, MAX_COLUMNS_TRAINING );
    float * test_final = (float *) malloc(MAX_COLUMNS_TESTING * MAX_ROWS_TESTING * sizeof(float));
    linearizeArray(testing_data, test_final, MAX_ROWS_TESTING, MAX_COLUMNS_TESTING);

    printf("--Training model...");
    float* betas = (float*) malloc(sizeof(float) * MAX_COLUMNS_TESTING);
    initialize_betas(betas);

    time_t start, end;
    time(&start);
    grad_desc(yvec, betas, train_final, lr, max_iters );
    time(&end);

    printf("done! ------");
    int time_taken = int(end-start);

    printf("Training time: %i ", time_taken);
    printf("--Printing betas...\n");
    for(int i=0; i< MAX_COLUMNS_TESTING; i++){
        printf("%f, ", betas[i]);
    }
    printf("\nEnd printing betas--\n");
    free(train_final);
    free(yvec);

    printf("--- Extracting and re-labelling test predictor...");

    int* yvec_test = (int * ) malloc(sizeof(int) * MAX_ROWS_TESTING);
    extract_yvec(testing_data, yvec_test, MAX_ROWS_TESTING);
    relabel_yvec(yvec_test, MAX_ROWS_TESTING, modelID, n_models);
    printf(" done! --- ");

    printf("--- Testing model...");
    //running predict
    float percent_correct = predict(betas, test_final, yvec_test);
    printf(" done");
    printf("Percent correct: %f %", percent_correct);

    free(test_final);
    free(yvec_test);
    free(betas);
    return 0;
}
