// Compute vector sum C = A+B
// Each thread performs one pair-wise addition
#include <stdlib.h>
#include <iostream>
#include <math.h>

__global__
void vecAddKernel(float* A_d, float* B_d, float* C_d, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i<n) C_d[i] = A_d[i] + B_d[i];
}


int main(int argc, char* argv[])
{
  if(argc < 2) {
    std::cout<<"Usage: ./add Num_items"<<std::endl;
    return 0;
  }
	int N = atoi(argv[1]);
	int size = N * sizeof(float); 
	float *A_h, *B_h, *C_h;
	float *A_d, *B_d, *C_d;

  A_h = (float*)malloc(sizeof(float)*N);
  B_h = (float*)malloc(sizeof(float)*N);
  C_h = (float*)malloc(sizeof(float)*N);

  for(int i=0;i<N;i++) {
    A_h[i] = i%100;
    B_h[i] = i*100;
    C_h[i] = 0.0;
  }

	cudaMalloc((void **) &A_d, size);
  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  cudaMalloc((void **) &B_d, size);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
 

	// Allocate device memory for
	cudaMalloc((void **) &C_d, size);

	// Kernel invocation code – to be shown later
  vecAddKernel<<<ceil(N/256.0), 256>>>(A_d, B_d, C_d, N);

  // Transfer C from device to host
	cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

	// Free device memory for A, B, C
	cudaFree(A_d); cudaFree(B_d); cudaFree (C_d);
  free(A_h); free(B_h); free(C_h);

  for(int i=0;i<N;i+=N/10) {
    std::cout<<"C["<<i<<"]="<<C_h[i]<<std::endl;
  }

	return 0;
}
