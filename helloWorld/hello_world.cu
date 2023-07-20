#include <stdio.h>

__global__ void hello_kernel(){
    const int tid = blockDim.x*blockIdx.x + threadIdx.x;

    printf("Hello from tid : %d\n", tid);
}

int main(int argc, char* argv[]){
    cudaSetDevice(0);
    
    int thread_num = atoi(argv[1]);
    hello_kernel<<<1,thread_num>>>();

    cudaDeviceSynchronize();
}