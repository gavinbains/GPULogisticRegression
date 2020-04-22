#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

/* Test cuda file that writes to an array all 0's. */

#define RESULT_SIZE 12

__constant__ int features = 5;

__global__ void SetZero(int* result) {
    int index = blockIdx.x;
    result[index] = 0;
}

__global__ void mult(int* results, int* data, int* vec) {
    int index = blockIdx.x;
    int result_val = 0;
    for(int i = 0; i < features; i++) {
        result_val += vec[i] * data[(index * features) + i];
    }
    results[index] = result_val;
}

int main() {
    // arr exists on the CPU/host
    int rows = 6;
    int* vec = (int*) malloc(sizeof(int) * features * 1);
    for(int i = 0; i < features; i++) {
        vec[i] = i;
    }
    int* data = (int*) malloc(sizeof(int) * features * rows);
    for(int i = 0; i < features * rows; i++) {
        data[i] = 1;
    }

    int* result = (int*) malloc(sizeof(int)* features);
    // copy vector and data to gpu
    int* gpu_vec;
    int* gpu_data;
    int* gpu_result;
    cudaMalloc((void**)&gpu_vec, sizeof(int) * features);
	cudaMalloc((void**)&gpu_data, sizeof(int) * features * rows);
	cudaMalloc((void**)&gpu_result, sizeof(int) * features);

    cudaMemcpy(gpu_vec, vec, sizeof(int) * features, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_data, data, sizeof(int) * features * rows cudaMemcpyHostToDevice);

    dim3 grid(rows);
    mult<<<rows, 1>>>(gpu_result, gpu_data, gpu_vec);
    // copy back to device
    cudaMemcpy(result, gpu_result, sizeof(int) * features, cudaMemcpyDeviceToHost);
    // check all of our result
    for(int i = 0; i < features; i++) {
        printf("%d \n", result[i]);
    }
}
