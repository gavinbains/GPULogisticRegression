#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

/* Test cuda file that writes to an array all 0's. */

#define RESULT_SIZE 12

__global__ void SetZero(int* result) {
    int index = blockIdx.x;
    result[index] = 0;
}

int main() {
    // arr exists on the CPU/host
    int* arr = (int*) malloc(sizeof(int) * RESULT_SIZE);
    for(int i = 0; i < RESULT_SIZE; i++) {
        arr[i] = -1;
    }
    int* result = (int*) malloc(sizeof(int)*RESULT_SIZE);
    // gpu_arr exists on the GPU/device
    int* gpu_arr;
    cudaMalloc((void**)&gpu_arr, sizeof(int)*RESULT_SIZE);
    // copy arr into gpu_arr
    cudaMemcpy(gpu_arr, arr, sizeof(int) * RESULT_SIZE, cudaMemcpyHostToDevice);

    dim3 dimGrid(12); // gives 12 "cores"? Can I do this?
    SetZero<<<dimGrid, 1>>>(gpu_arr);
    // copy back to device
    cudaMemcpy(arr, gpu_arr, sizeof(int) * RESULT_SIZE, cudaMemcpyDeviceToHost);
    // check all of our result
    for(int i = 0; i < RESULT_SIZE; i++) {
        printf("%d:", i);
        printf("%d \n", arr[i]);
    }
    free(arr);
    free(result);
    cudaFree(gpu_arr);
}
