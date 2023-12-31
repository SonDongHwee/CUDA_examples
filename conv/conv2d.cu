/*
* Description : Implement of 2d convolution
* Compile : nvcc -o conv2d conv2d.cu
* Run : ./conv2d
* Argument : 
*           --width=<N> : specify the width of input image
*           --height=<N> : specify the height of input image
*           --channel=<N> : specify #of channels
*           --filter=<N> : specify #of filter width
*           --kernel=<N> : specify which kernel to run
*                   [0] : basic 1d conv with constant memory
*                   [1] : tiled 2d conv with constant memory
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "../include/common.h"
#include "../include/common_string.h"

#define O_TILE_WIDTH 16
#define MAX_KERNEL_WIDTH 10

__constant__ float M[MAX_KERNEL_WIDTH * MAX_KERNEL_WIDTH];
void convolution2D_CPU(float* in, float* out, float* kernel, int width, int height, int channels, int kernel_width);
__global__ void convolution2D(float* in, float* out, int width, int height, int channels, int kernel_width);
__global__ void convolution2D_tiled(float* P, float* N, int width, int height, int channels, int kernel_width);
bool run(int width, int height, int channels, int kernel_width, int whichKernel);

int main(int argc, char* argv[]){
    printf("[2D convolution..!] \n\n\n");

    int height = 1080;
    int width = 1920;
    int channels = 1;
    int kernel_width = 5;
    int whichKernel = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "width")) {
        width = getCmdLineArgumentInt(argc, (const char **)argv, "width");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "height")) {
        height = getCmdLineArgumentInt(argc, (const char **)argv, "height");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "channels")) {
        channels = getCmdLineArgumentInt(argc, (const char **)argv, "channels");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "filter")) {
        kernel_width = getCmdLineArgumentInt(argc, (const char **)argv, "filter");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "kernel")) {
        whichKernel = getCmdLineArgumentInt(argc, (const char **)argv, "kernel");
    }

    printf("Kernel Size: %d x %d\n", kernel_width, kernel_width);
    printf("Input Size: %d x %d x %d\n", width, height, channels);

    int dev = 0;
    cudaSetDevice(dev);

    bool result = run(width, height, channels, kernel_width, whichKernel);
    
    printf(result ? "Test PASSED\n" : "Test FAILED!\n");
    cudaDeviceReset();

    return 0;
}

bool run(int width, int height, int channels, int kernel_width, int whichKernel){
    unsigned int bytes = width * height * channels * sizeof(float);
    float* h_in, *h_out, *h_kernel;
    float* d_in, *d_out;

    // allocate host memory
    h_in = (float*)malloc(bytes);
    h_out = (float*)malloc(bytes);
    h_kernel = (float*)malloc(kernel_width*kernel_width*sizeof(float));

    // init value
    for (int c = 0; c < channels; c++)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                h_in[c*height*width + i*width + j] = rand() / (float)RAND_MAX;
    for (int i = 0; i < kernel_width; i++)
        for (int j = 0; j < kernel_width; j++)
            h_kernel[i*kernel_width + j] = rand() / (float)RAND_MAX;
    
    // allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(M, h_kernel, kernel_width*kernel_width*sizeof(float)));

    // launch Kernel
    printf("\nLaunch Kernel...\n");
    double start, finish;
    if (whichKernel > 1)
        whichKernel = 0;
    if (whichKernel == 0) {
        // basic 2d conv
        dim3 dimBlock(O_TILE_WIDTH, O_TILE_WIDTH, 1);
        dim3 dimGrid((width + O_TILE_WIDTH - 1) / O_TILE_WIDTH, (height + O_TILE_WIDTH - 1) / O_TILE_WIDTH, channels);
        printf("block dim: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
        printf("grid dim: %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);

        //warm up
        convolution2D<<<dimGrid, dimBlock>>>(d_in, d_out, width, height, channels, kernel_width);
        CUDA_CHECK(cudaGetLastError());
        cudaDeviceSynchronize();

        GET_TIME(start);
        convolution2D<<<dimGrid, dimBlock>>>(d_in, d_out, width, height, channels, kernel_width);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        GET_TIME(finish);
    }
    else {
        // tiled 2d conv
        const int I_TILE_WIDTH = O_TILE_WIDTH + kernel_width - 1;
        dim3 dimBlock(I_TILE_WIDTH, I_TILE_WIDTH, 1);
        dim3 dimGrid((width + O_TILE_WIDTH - 1) / O_TILE_WIDTH, (height + O_TILE_WIDTH - 1) / O_TILE_WIDTH, channels);
        printf("TILE size: %d\n", I_TILE_WIDTH);
        printf("block dim: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
        printf("grid dim: %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
        
        //warm up
        convolution2D_tiled<<<dimGrid, dimBlock>>>(d_in, d_out, width, height, channels, kernel_width);
        CUDA_CHECK(cudaGetLastError());
        cudaDeviceSynchronize();

        GET_TIME(start);
        convolution2D_tiled<<<dimGrid, dimBlock>>>(d_in, d_out, width, height, channels, kernel_width);
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        GET_TIME(finish);
    }

    // result in CPU
    float* cpu_out = (float*)malloc(bytes);
    printf("\nCalculating in CPU...\n");
    convolution2D_CPU(h_in, cpu_out, h_kernel, width, height, channels, kernel_width);
    
    int precision = 8;
    double threshold = 1e-8 * channels*width*height;
    double diff = 0.0;
    for (int i = 0; i < channels*width*height; i++) {
        diff += fabs((double)cpu_out[i] - (double)h_out[i]);
    }
    diff /= (double)channels*width*height;

    printf("[Kernel %d] Throughput = %.4f GB/s, Time = %.5f ms\n",
        whichKernel, ((double)bytes / (finish-start))*1.0e-9, (finish-start)*1000);
    printf("Error : %.*f (threshold: %f)\n", precision, (double)diff, threshold);

    // free memory
    free(h_in);
    free(h_out);
    free(h_kernel);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(cpu_out);
    
    return (diff < threshold);
}

void convolution2D_CPU(float* in, float* out, float* kernel, int width, int height, int channels, int kernel_width){
    //TODO
}
__global__ void convolution2D(float* in, float* out, int width, int height, int channels, int kernel_width){
    //TODO
}
__global__ void convolution2D_tiled(float* P, float* N, int width, int height, int channels, int kernel_width){
    //TODO
}