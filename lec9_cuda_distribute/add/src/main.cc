
#include <chrono>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "add.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

int main(int argc, char* argv[])
{
  if(argc < 2) {
    std::cout<<"Usage: ./add Num_items"<<std::endl;
    return 0;
  }
	long long int N = atoll(argv[1]);
	size_t size = N * sizeof(float); 
	float *A_h, *B_h, *C_h;
	float *A_d, *B_d, *C_d;

  A_h = (float*)malloc(sizeof(float)*N);
  B_h = (float*)malloc(sizeof(float)*N);
  C_h = (float*)malloc(sizeof(float)*N);

  for(long long int i=0;i<N;i++) {
    A_h[i] = i%100;
    B_h[i] = i*100;
    C_h[i] = 0.0;
  }


	cudaMalloc((void **) &A_d, size);
	cudaMalloc((void **) &B_d, size);
	cudaMalloc((void **) &C_d, size);
  std::chrono::duration<double> h2d_diff;
  auto h2d_start = std::chrono::steady_clock::now();
	cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

	// Allocate device memory for
  auto h2d_end = std::chrono::steady_clock::now();
  h2d_diff = h2d_end - h2d_start;
  std::cout<<"h2d copy took "<<h2d_diff.count()<<" sec"<<std::endl;

	// Kernel invocation code – to be shown later
  std::chrono::duration<double> cuda_diff;
  auto cuda_start = std::chrono::steady_clock::now();
  vecAdd(A_d, B_d, C_d, N);
  auto cuda_end = std::chrono::steady_clock::now();
  cuda_diff = cuda_end - cuda_start;
  std::cout<<"cuda kernel took "<<cuda_diff.count()<<" sec"<<std::endl;
  // Transfer C from device to host
  std::chrono::duration<double> d2h_diff;
  auto d2h_start = std::chrono::steady_clock::now();
	cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
  auto d2h_end = std::chrono::steady_clock::now();
  d2h_diff = d2h_end - d2h_start;
  std::cout<<"d2h copy took "<<d2h_diff.count()<<" sec"<<std::endl;

  
  std::chrono::duration<double> cuda_all_diff = d2h_end - h2d_start;
  std::cout<<"end to end cuda took "<<cuda_all_diff.count()<<" sec"<<std::endl;
	// Free device memory for A, B, C
	cudaFree(A_d); cudaFree(B_d); cudaFree (C_d);
  vecAdd(A_h, B_h, C_h, N);

  for(long long int i=0;i<N;i+=N/10) {
    std::cout<<"C_d["<<i<<"]="<<C_h[i]<<std::endl;
  }
  for(long long int i=0;i<N;i++) {
    A_h[i] = i%100;
    B_h[i] = i*100;
    C_h[i] = 0.0;
  }

  std::chrono::duration<double> mp_diff;
  auto mp_start = std::chrono::steady_clock::now();
  vecAdd_omp(A_h, B_h, C_h, N);
  auto mp_end = std::chrono::steady_clock::now();
  mp_diff = mp_end - mp_start;
  std::cout<<"omp add took "<<mp_diff.count()<<" sec"<<std::endl;
  for(int i=0;i<N;i+=N/10) {
    std::cout<<"C_h["<<i<<"]="<<C_h[i]<<std::endl;
  }
  free(A_h); free(B_h); free(C_h);
	return 0;
}
