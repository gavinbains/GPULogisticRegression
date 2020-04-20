#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cublas.h>
#include <time.h>

//define N
//define threads/block

//function to be run on the device


//FILE IO RELATED
//max number of lines
#define MAX_ROWS 17012
// max number of columns/features
#define MAX_COLUMNS 26
//max number of characters/line
#define MAX_CHAR 300


//on the cpu
int main(void){
    // array that holds all converted training data
    float **train_data = (float **) malloc(MAX_ROWS * sizeof(float *));
    for(int i = 0; i < MAX_ROWS; i++) {
        train_data[i] = (float *) malloc(MAX_COLUMNS * sizeof(float));
    }
    for(int i = 0; i < MAX_ROWS; i++) {
	for(int j = 0; j < MAX_COLUMNS; j++) {
	    train_data[i][j] = 0.0f;
	}
    }
    printf("done initializing. \n");
    //assumed file is in same folder, also rename file here
    char *filename = "testing_data.csv";
    FILE *file;
    file = fopen(filename, "r+");
    if(!file) {
        printf("Can't open file \n");
        return 0;
    }
    //thinking about how we want to access and store the unparsed_data
    //array of lines, does each line break up into array of columns?
    //feel free to try other ways we could arrange the unparsed_data

    //var to store all of the unparsed_data
    char unparsed_data[MAX_ROWS+10][MAX_CHAR+10];
    char copying[MAX_ROWS+10];

    //keep track which line we are on
    int ltracker=0;

    while(fgets(copying, sizeof(copying)-10, file) != 0){
        //copying the line from temp to our array
        strncpy(unparsed_data[ltracker], copying, MAX_CHAR);
        ltracker++;
    }

    fclose(file);

    //storing the column indvidual value
    char* col_val;

    //the delimiter, in this file its commas
    const char deli[2]=",";

    for (int row = 0; row < MAX_ROWS; row++){
        //telling it to separate out a line in unparsed_data by commas
        col_val = strtok(unparsed_data[row], deli);
        for(int col  = 0; col < MAX_COLUMNS; col++) {
            col_val = strtok(NULL, deli);
	    if(col_val != NULL) {
 		train_data[row][col] = atof(col_val);
	    }
        }
    }

    for(int row = 0; row < MAX_ROWS; row++) {
        printf("Row: %f: ", row);
        for(int col = 0; col < MAX_COLUMNS; col++) {
            printf("%f, ", train_data[row][col]);
        }
        printf("\n");
        break;
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
