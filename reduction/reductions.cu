#include <stdio.h>
#include "../include/common.h"
#include "../include/common_string.h"

__global__ void reduce0(int *g_idata, int* g_odata){
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + tid;
    sdata[tid] = g_idata[i];
    __syncthreads();

    // Do reduction in shared mem
    for(unsigned int s=1;s<blockDim.x;s*=2){
        if(tid % (2*s) == 0){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // Should this work well? 
    // What is problem? 
    // in code (tid % (2*s) == 0) <- highly branch divergent warp are very inefficient and  % operator is slow. c 

    if(tid==0) g_odata[blockIdx.x] = sdata[0];
}
__global__ void reduce1(int *g_idata, int* g_odata){
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + tid;
    sdata[tid] = g_idata[i];
    __syncthreads();

    // Do reduction in shared mem
    for(unsigned int s=1;s<blockDim.x;s*=2){
        int index = 2*s*tid;

        if(index < blockDim.x){
            sdata[index] += sdata[index + s];
        }
        __syncthreads(); 
    }
    // New problem : Shared memory bank conflict


    if(tid==0) g_odata[blockIdx.x] = sdata[0];
}
__global__ void reduce2(int *g_idata, int* g_odata){
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + tid;
    sdata[tid] = g_idata[i];
    __syncthreads();

    // Do reduction in shared mem
    // Sequential Addressing
    for(unsigned int s=blockDim.x/2;s>0;s>>=1){
        id(tid < s){
            sdata[tid] += sdata[tid + s]; 
        }
        __syncthreads(); 
    } 
    // Problem : Idle threads - Half of threads are idle on first loop iteration


    if(tid==0) g_odata[blockIdx.x] = sdata[0];
} 
__global__ void reduce3(int *g_idata, int* g_odata){
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + tid;
    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();

    // Do reduction in shared mem
    // First Add During Load
    for(unsigned int s=blockDim.x/2;s>0;s>>=1){
        id(tid < s){
            sdata[tid] += sdata[tid + s]; 
        }
        __syncthreads(); 
    } 
    // 
    if(tid==0) g_odata[blockIdx.x] = sdata[0];
} 
__global__ void reduce4(int *g_idata, int* g_odata){
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + tid;
    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();

    // Do reduction in shared mem
    // Unroll the last warp
    // Why? As reduction proceeds, # of active threads decreases
    // Instruction are SIMD within a warp
    // That means, when s<=32 we don't need to __syncthreads()
    for(unsigned int s=blockDim.x/2;s>32;s>>=1){
        id(tid < s){
            sdata[tid] += sdata[tid + s]; 
            __syncthreads(); 
        }
        if(tid < 32){
            sdata[tid] += sdata[tid + 32];
            sdata[tid] += sdata[tid + 16];
            sdata[tid] += sdata[tid + 8];
            sdata[tid] += sdata[tid + 4];
            sdata[tid] += sdata[tid + 2];
            sdata[tid] += sdata[tid + 1];
        } 
    } 
    // Note that this saves useless work in all warps, not just the last one!
    if(tid==0) g_odata[blockIdx.x] = sdata[0];
} 
//Final optimized kernel
template <unsigned int blockSize> // note that template parameter determined at compile time
__global__ void reduce5(int *g_idata, int* g_odata){
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    // grid-stride loop!
    while(i < n){
        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        i+=gridSize
    }
    
    __syncthreads();

    if(blockSize >= 512){ if(tid < 256){sdata[tid] += sdata[tid + 256];}__syncthreads();}
    if(blockSize >= 256){ if(tid < 128){sdata[tid] += sdata[tid + 128];}__syncthreads();}
    if(blockSize >= 128){ if(tid <  64){sdata[tid] += sdata[tid +  64];}__syncthreads();}

    if(tid < 32){
        if(blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if(blockSize >= 32) sdata[tid] += sdata[tid + 16]; 
        if(blockSize >= 16) sdata[tid] += sdata[tid +  8];
        if(blockSize >=  8) sdata[tid] += sdata[tid +  4];
        if(blockSize >=  4) sdata[tid] += sdata[tid +  2];
        if(blockSize >=  2) sdata[tid] += sdata[tid +  1];
    }
    if(tid==0) g_odata[blockIdx.x] = sdata[0];
} 