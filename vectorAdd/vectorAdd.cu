#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "../include/common.h"
#include "../include/common_string.h"

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}


__global__ void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  // Note that implemented grid-Stride loop
  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main(int argc, char** argv)
{
  int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  if(checkCmdLineFlag(argc , (const char**) argv, "n")){
    N = getCmdLineArgumentInt(argc, (const char**)argv, "n");
  }
  CUDA_CHECK(cudaMallocManaged(&a, size));
  CUDA_CHECK(cudaMallocManaged(&b, size));
  CUDA_CHECK(cudaMallocManaged(&c, size));

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);
    
  int threadsPerBlock = 256;
  dim3 dimBlock(threadsPerBlock,1,1);
  dim3 dimGrid(SDIV(N, threadsPerBlock)); //(N + threadsPerBlock - 1) / threadsPerBlock;  
  
  double start, finish, elapsed = 0.f;
  GET_TIME(start);
  addVectorsInto<<<dimGrid, dimBlock>>>(c, a, b, N);
  GET_TIME(finish);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  elapsed = finish - start;

  checkElementsAre(7, c, N);
  printf("\n\n\n%f ms elapsed for computing %d element vector addition\n", elapsed*1000, N);

  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(c));
}