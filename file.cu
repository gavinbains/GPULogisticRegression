#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cublas.h>
#include <time.h>

//define N
//define threads/block


//FILE IO RELATED
//max number of lines
#define MAX_ROWS 17012
// max number of columns/features
#define MAX_COLUMNS 26
//max number of characters/line
#define MAX_CHAR 300

bool LoadCSV(float** training_data) {
    //assumed file is in same folder, also rename file here
    char *filename = "testing_data.csv";
    FILE *file;
    file = fopen(filename, "r+");
    if(!file) {
        printf("Can't open file \n");
        return false;
    }
    // unparsed data straight from file
    char unparsed_data[MAX_ROWS+10][MAX_CHAR+10];
    char copying[MAX_ROWS+10];

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
    for (int row = 0; row < MAX_ROWS; row++){
        col_val = strtok(unparsed_data[row], deli);
        for(int col  = 0; col < MAX_COLUMNS; col++) {
            col_val = strtok(NULL, deli);
            if(col_val != NULL) {
                training_data[row][col] = atof(col_val);
            }
        }
    }
    return true;
}

//on the cpu
int main(void){
    // array that holds all converted training data
    float **train_data = (float **) malloc(MAX_ROWS * sizeof(float *));
    for(int i = 0; i < MAX_ROWS; i++) {
        train_data[i] = (float *) malloc(MAX_COLUMNS * sizeof(float));
    }

    if(LoadCSV(train_data)) {
        for(int i = 0; i < MAX_ROWS; i++) {
            printf("Row %i: ", i );
            for(int j = 0; j < MAX_COLUMNS; j++) {
                printf("%f, ", train_data[i][j]);
            }
            printf("\n");
            break;
        }
    }
    //data parsing completed

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
