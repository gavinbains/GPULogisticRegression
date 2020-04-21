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
    //TODO: store the lines in the file as arrays
    //unparsed_data[0] = [colum1, colum2, column3, .. ,columnN]

    //things on host
    //things to device
    //used to set size of components


    //alloc space for device, copies of above

    //alloc space for host and setup input values

    //copy inputs to device

    //launch on gpu

    //copy result to host

    //cleanup all the frees

    return 0;
}
