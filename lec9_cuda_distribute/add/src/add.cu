// Compute vector sum C = A+B
// Each thread performs one pair-wise addition
#include "add.h"


__global__
void vecAddKernel(float* A_d, float* B_d, float* C_d, long long int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i<n) C_d[i] = A_d[i] + B_d[i];
}

void vecAdd(float* A_d, float* B_d, float* C_d, long long int n)
{
  vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);
}
