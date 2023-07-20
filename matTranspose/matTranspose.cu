#include <stdio.h>
#include "../include/common.h"
#include "../include/common_string.h"

__global__ void transposeKernel_0(float* A, float* B, long long int n); //1d : coalesced read, non-coalesced write 
__global__ void transposeKernel_1(float* A, float* B, long long int n); //1d : coalesced write, non-coalesced read
__global__ void transposeKernel_2(float* A, float* B, long long int n); //2d
__global__ void transposeKernel_3(float* A, float* B, long long int n); //2d 
__global__ void transposeKernel_4(float* A, float* B, long long int n); //utilize shared memory
void checkElements(float *A, float* B, long long int n){
    for(int i=0;i<n*n;i++){
        if(A[i] != B[i])
            {
            printf("FAIL: Wrongly Transposed\n");
            exit(1);
            }
    }
    printf("SUCCESS: Correctly Transposed\n\n");
}

int main(int argc, char** argv){
    
    long long int n = 2<<15;

    if(checkCmdLineFlag(argc, (const char**)argv, "n")) {
        n = getCmdLineArgumentInt(argc, (const char**)argv, "n");
    }
    long long int bytes = n*n*sizeof(float);

    // host mem alloc
    float* A_h = (float *)malloc(bytes);
    float* B_h = (float *)malloc(bytes);
    if(A_h == NULL || B_h == NULL){
        fprintf(stderr, "Failed to allocate matrices at host\n");
        exit(0);
    }

    // Init matrices
    double start, finish, elapsed = 0.f;
    printf("Initializing matrices... \n\n");
    GET_TIME(start);
    common_random_init_matrix<float>(A_h,n,n);
    common_random_init_matrix<float>(B_h,n,n);
    GET_TIME(finish);
    elapsed = finish - start;
    printf("\n\n..%fms elapsed for initializing matrices\n\n\n", elapsed);

    //device mem alloc
    float *A_d, *B_d;
    CUDA_CHECK(cudaMalloc(&A_d, bytes));
    CUDA_CHECK(cudaMalloc(&B_d, bytes));

    //H2D Copy
    CUDA_CHECK(cudaMemcpy(A_d, A_h, bytes, H2D));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, bytes, H2D));

    int threadPerBlock = 1024;
    dim3 dimBlock(threadPerBlock,1,1);
    dim3 dimGrid(SDIV(n*n,threadPerBlock),1,1);

    // Lauch Kernel 0 
    GET_TIME(start);
    transposeKernel_0<<<dimGrid, dimBlock>>>(A_d,B_d,n);
    GET_TIME(finish);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    elapsed = finish - start;

    printf("Launching Kernel 0\n");
    printf("%f ms elapsed for computing %lldx%lld matrix transpose\n", elapsed*1000, n, n);
    printf("Bandwidth : %fGBps\n\n", n*n*4*2/(elapsed*1000*1000*1000));
    //

    // Launch Kernel 1
    GET_TIME(start);    
    transposeKernel_1<<<dimGrid, dimBlock>>>(B_d,A_d,n);
    GET_TIME(finish);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    elapsed = finish - start;

    printf("Lanching Kernel 1\n");
    printf("%f ms elapsed for computing %lldx%lld matrix transpose\n", elapsed*1000, n, n);
    printf("Bandwidth : %fGBps\n\n", n*n*4*2/(elapsed*1000*1000*1000));
    //Correctness Check
    float* A_c = (float *)malloc(bytes);
    float* B_c = (float *)malloc(bytes);
    if(A_c == NULL || B_c == NULL){
        fprintf(stderr, "Failed to allocate matrices at host\n");
        exit(0);
    }
    CUDA_CHECK(cudaMemcpy(A_c,A_d,bytes,D2H));
    CUDA_CHECK(cudaMemcpy(B_c,B_d,bytes,D2H));
    checkElements(A_c,A_h,n);
    //

    // Launch Kernel 2
    dim3 dimBlock_2(16,16,1);
    dim3 dimGrid_2(SDIV(n*n,16*16));

    GET_TIME(start);    
    transposeKernel_2<<<dimGrid_2, dimBlock_2>>>(A_d,B_d,n);
    GET_TIME(finish);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    elapsed = finish - start;

    printf("Lanching Kernel 2\n");
    printf("%f ms elapsed for computing %lldx%lld matrix transpose\n", elapsed*1000, n, n);
    printf("Bandwidth : %fGBps\n\n", n*n*4*2/(elapsed*1000*1000*1000));
    //

    // Launch Kernel 3
    GET_TIME(start);    
    transposeKernel_3<<<dimGrid_2, dimBlock_2>>>(B_d,A_d,n);
    GET_TIME(finish);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    elapsed = finish - start;

    printf("Lanching Kernel 3\n");
    printf("%f ms elapsed for computing %lldx%lld matrix transpose\n", elapsed*1000, n, n);
    printf("Bandwidth : %fGBps\n\n", n*n*4*2/(elapsed*1000*1000*1000));
    CUDA_CHECK(cudaMemcpy(A_c,A_d,bytes,D2H));
    CUDA_CHECK(cudaMemcpy(B_c,B_d,bytes,D2H));
    checkElements(A_c,A_h,n);
    //

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    free(A_h);
    free(B_h);
    
}

__global__ void transposeKernel_0(float* A, float* B, long long int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int col = tid % n;
    int row = tid / n;

    if(row < n && col <n){
        B[col*n + row] = A[row*n + col];
    }
}
__global__ void transposeKernel_1(float* A, float* B, long long int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int col = tid % n;
    int row = tid / n;

    if(row < n && col <n){
        B[row*n + col] = A[col*n + row];
    }
}
__global__ void transposeKernel_2(float* A, float* B, long long int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < n && col < n){
        B[col*n + row] = A[row*n +col];
    }
}
__global__ void transposeKernel_3(float* A, float* B, long long int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < n && col < n){
        B[col*n + row] = A[row*n +col];
    }
}
