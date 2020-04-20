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
    #define MAX_LINES 1000
//max number of characters/line
    #define MAX_CHAR 300


//on the cpu
int main(void){

    //assumed file is in same folder, also rename file here
    char *filename = "sample_data.csv";
    FILE *file;

    file = fopen(filename, "r+");

    if(!file){
        printf("Can't open file\n");
        return 0;
    }

    //thinking about how we want to access and store the data
    //array of lines, does each line break up into array of columns?
    //feel free to try other ways we could arrange the data

    //var to store all of the data
    char data[MAX_LINES+10][MAX_CHAR+10];
    char temp[MAX_LINES+10];

    //keep track which line we are on
    int ltracker=0;

    while(fgets(temp, sizeof(temp)-10, file) != 0){
        //copying the line from temp to our array
        strncpy(data[ltracker], temp, MAX_CHAR);  
        ltracker++;
    }

    //print output to check
    /*for(int i=0; i<25; i++ ){
        printf("%s",data[i]);
    }*/

    
    fclose(file);

    char* temp2;
    //the delimiter, in this file its commas
    const char deli[2]=",";

    for (int i=0; i<25; i++){
        //telling it to separate out a line in data by commas
    
        temp2 = strtok(data[i], deli);
        while(temp2 != NULL){
            printf("%s \n", temp2);
            temp2 = strtok(NULL, deli);	
        }
        
    }




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
