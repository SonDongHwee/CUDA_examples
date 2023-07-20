/*
* Description : Implement of 1d convolution
* Compile : nvcc -o conv1d conv1d.cu
* Run : ./conv1d
* Argument : 
*           --n=<N> : specify #of elements to reduce (default : 1<<20)
*           --threads=<N> : specify #of threads per block
*           --iteration=<N> : specify #of iteration
*           --filter=<N> : specify #of filter width
*           --kernel=<N> : specify which kernel to run
*                   [0] : basic 1d conv without constant memory
*                   [1] : basic 1d conv with constant memory
*                   [2] : tiled 1d conv
*                   [3] : tiled 1d cond with L2 cache
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "../include/common.h"
#include "../include/common_string.h"

#define MAX_KERNEL_WIDTH 10

__constant__ float M[MAX_KERNEL_WIDTH];

bool run(int size, int kernel_width, int threads, int blocks, int nIter, int whichKernel);
void convolution1D_CPU(float* N_h, float* M_h, float* P_h, int Kernel_Width, int Width);
__global__ void convolution1D_naive_woConMem(float* N_d,float* M_d, float*P_d, int Kernel_Width, int Width);
__global__ void convolution1D_naive_wConMem(float* N_d, float* P_d, int Kernel_Width, int Width);
template<unsigned int TILE_SIZE>
__global__ void convolution1D_tiled(float* N_d, float* P_d, int Kernel_Width, int Width);
template<unsigned int TILE_SIZE>
__global__ void convolution1D_tiled_L2(float* N_d, float* P_d, int Kernel_Width, int Width);

int main(int argc, char* argv[]){
    printf("[1D convoltion..!]\n\n\n");

    int size = 1<<20;
    int threads = 256;
    int nIter = 100;
    int whichKernel = 1;
    int kernel_width = 5;

    if(checkCmdLineFlag(argc, (const char**)argv, "n")) {
        size = getCmdLineArgumentInt(argc, (const char**)argv, "n");
    }
    if(checkCmdLineFlag(argc, (const char**)argv, "threads")){
        threads = getCmdLineArgumentInt(argc, (const char**)argv, "threads");
    }
    if(checkCmdLineFlag(argc, (const char**)argv, "kernel")){
        whichKernel = getCmdLineArgumentInt(argc, (const char**)argv, "kernel");
    }
    if(checkCmdLineFlag(argc, (const char**)argv, "iteration")){
        nIter = getCmdLineArgumentInt(argc, (const char**)argv, "iteration");
    }
    if(checkCmdLineFlag(argc, (const char**)argv, "filter")){
        kernel_width = getCmdLineArgumentInt(argc, (const char**)argv, "filter");
    }
    printf("Kernel Width: %d\n", kernel_width);
    printf("%d elements\n", size);
    printf("%d threads\n", threads);
    int blocks = (size + threads - 1) / threads;
    printf("%d blocks\n", blocks);

    int dev = 0;
    cudaSetDevice(dev);

    bool result = run(size, kernel_width, threads, blocks, nIter, whichKernel);

    printf(result ? "Test Passed\n" : "Test Failed\n");

    return 0;
}

bool run(int size, int kernel_width, int threads, int blocks, int nIter, int whichKernel){
    unsigned int bytes = size * sizeof(float);
    float *N_h, *M_h, *P_h;
    float *N_d, *M_d, *P_d;

    //host mem alloc
    N_h = (float*)malloc(bytes);
    M_h = (float*)malloc(kernel_width*sizeof(float));
    P_h = (float*)malloc(bytes);

    for(int i=0;i<size;i++)
        N_h[i] = rand()/(float)RAND_MAX;
    for(int i=0;i<kernel_width;i++)
        M_h[i] = rand()/(float)RAND_MAX;

    CUDA_CHECK(cudaMalloc((void**)&N_d, bytes));
    CUDA_CHECK(cudaMalloc((void**)&M_d, kernel_width*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&P_d, bytes));

    CUDA_CHECK(cudaMemcpy(N_d, N_h, bytes, H2D));
    if(whichKernel == 0) {
        CUDA_CHECK(cudaMemcpy(M_d, M_h, kernel_width*sizeof(float), H2D));
    }
    else{
        CUDA_CHECK(cudaMemcpyToSymbol(M, M_h, kernel_width*sizeof(float))); // If u are utilizing __const__ or __device__ , u should use this API.
    }

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks,1,1);

    convolution1D_naive_woConMem<<<dimGrid, dimBlock>>>(N_d, M_d, P_d, kernel_width, size);

    double start, finish, total_time = 0.f;
    for(int i=0;i<nIter;i++){
        cudaDeviceSynchronize();
        GET_TIME(start);
        switch(whichKernel){
            case 0:
                convolution1D_naive_woConMem<<<dimGrid, dimBlock>>>(N_d, M_d, P_d, kernel_width, size);
                break;
            default:
            case 1:
                convolution1D_naive_wConMem<<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                break;
            case 2:
                switch (threads) {
                    case 1024:
                        convolution1D_tiled<1024><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                    case 512:
                        convolution1D_tiled<512><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                    case 256:
                        convolution1D_tiled<256><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                    case 128:
                        convolution1D_tiled<128><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                    case 64:
                        convolution1D_tiled<64><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                    case 32:
                        convolution1D_tiled<32><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                    case 16:
                        convolution1D_tiled<16><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                    case 8:
                        convolution1D_tiled<8><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                    case 4:
                        convolution1D_tiled<4><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                    case 2:
                        convolution1D_tiled<2><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                    case 1:
                        convolution1D_tiled<1><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                }
            case 3:
                switch (threads) {
                    case 1024:
                        convolution1D_tiled_L2<1024><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                    case 512:
                        convolution1D_tiled_L2<512><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                    case 256:
                        convolution1D_tiled_L2<256><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                    case 128:
                        convolution1D_tiled_L2<128><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                    case 64:
                        convolution1D_tiled_L2<64><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                    case 32:
                        convolution1D_tiled_L2<32><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                    case 16:
                        convolution1D_tiled_L2<16><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                    case 8:
                        convolution1D_tiled_L2<8><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                    case 4:
                        convolution1D_tiled_L2<4><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                    case 2:
                        convolution1D_tiled_L2<2><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                    case 1:
                        convolution1D_tiled_L2<1><<<dimGrid, dimBlock>>>(N_d, P_d, kernel_width, size);
                        break;
                }
                break;
        }
    }

    CUDA_CHECK(cudaMemcpy(P_h, P_d, bytes, D2H));
    cudaDeviceSynchronize();
    GET_TIME(finish);

    total_time += (finish - start);


     // result in CPU
    float* cpu_P = (float*)malloc(bytes);
    convolution1D_CPU(N_h, M_h, cpu_P, kernel_width, size);

    int precision = 8;
    double threshold = 1e-8 * size;
    double diff = 0.0;
    for (int i = 0; i < size; i++) {
        diff += fabs((double)cpu_P[i] - (double)P_h[i]);
    }
    diff /= (double)size;

    double elapsedTime = (total_time / (double)nIter);
    printf("[Kernel %d] Throughput = %.4f GB/s, Time = %.5f ms\n",
        whichKernel, ((double)bytes / elapsedTime)*1.0e-9, elapsedTime * 1000);
    printf("Error : %.*f\n", precision, (double)diff);

    free(N_h);
    free(M_h);
    free(P_h);
    CUDA_CHECK(cudaFree(P_d));
    CUDA_CHECK(cudaFree(M_d));
    CUDA_CHECK(cudaFree(N_d));
    free(cpu_P);

    return (diff < threshold);
}

void convolution1D_CPU(float* N_h, float* M_h, float* P_h, int Kernel_Width, int Width){
    for(int i=0;i<Width; i++){
        float Pvalue = 0.f;
        int N_start_point = i-(Kernel_Width/2);
        for(int j=0; j<Kernel_Width; j++){
            if(N_start_point + j >= 0 && N_start_point + j <Width)
                Pvalue += N_h[N_start_point + j] * M_h[j];
        }
        P_h[i] = Pvalue;
    }
}

__global__ void convolution1D_naive_woConMem(float* N_d, float* M_d ,float* P_d, int Kernel_Width, int Width){
    // TODO
}
__global__ void convolution1D_naive_wConMem(float* N_d, float* P_d, int Kernel_Width, int Width){
    //TODO
}
template<unsigned int TILE_SIZE>
__global__ void convolution1D_tiled(float* N_d, float* P_d, int Kernel_Width, int Width){
    //TODO
}
template<unsigned int TILE_SIZE>
__global__ void convolution1D_tiled_L2(float* N_d, float* P_d, int Kernel_Width, int Width){
    //TODO
}