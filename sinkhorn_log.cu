#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>


using namespace std;

#define N 500
#define SIZE 32

void init(float *h_k, float *h_u, float *h_v){

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);


    for(int i = 0; i < N*N; i ++){
        float number = distribution(generator);
        h_k[i] = number;
    }
    
    for(int i = 0; i < N; i ++){
        h_u[i] = 0;
        h_v[i] = 0;
    }

    return ;
}

__device__ __forceinline__
void atomicMaxFloat(float *addr, float val){
    atomicMax((int*)sh_max, __float_as_int(val));
} 

// each row[i] add d_u[i]
// then calculate the max for each row
__global__ void step1(float *d_k, float *d_v, float *d_row_max){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // change in the future, we need to set the size to be the block size
    __shared__ float sh_uv[SIZE];
    __shared__ float sh_max[SIZE];

    if(threadIdx.x == 0 && row < N){
        sh_uv[threadIdx.y] = d_v[row];
        sh_max[threadIdx.y] = -1.0;
        
    }
    __syncthreads();

    if(row < N && col < N){
        sh_max[threadIdx.y] = atomicMaxFloat(&sh_max[threadIdx.y], d_k[row * N + col] + sh_uv[threadIdx.y]);
    }



}

// calculate the maximum value per row
__global__ void step2(){

}

int main(){

    size_t bytes = sizeof(float) * N * N;

    float *h_k, *h_u, *h_v, *h_row_max;
    float *d_k, *d_u, *d_v, *d_row_max;

    // for h_cost
    h_k = (float*)malloc(bytes);
    h_u = (float*)malloc(sizeof(float) * N);
    h_v = (float*)malloc(sizeof(float) * N);
    h_row_max = (float*)malloc(sizeof(float) * N);
    init(h_cost, h_u, h_v);

    // cuda memeory allocation for GPU
    cudaMalloc(&d_k, bytes);
    cudaMalloc(&d_u, sizeof(float) * N);
    cudaMalloc(&d_v, sizeof(float) * N);
    cudaMalloc(&d_row_max, sizeof(float) * N);
     

    //copy memeory from host to device
    cudaMemcpy(d_k, h_k, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, h_u, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, sizeof(float) * N, cudaMemcpyHostToDevice);

    int BLOCK_SIZE = min(SIZE, 1024);
    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(GRID_SIZE, GRID_SIZE)

    // each row[i] add u[i]
    step1<<<grid, threads>>>(d_k, d_v, d_row_max);
    
    // calculate the maximum value for each row


}