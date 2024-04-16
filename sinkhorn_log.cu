#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>
#include <stdio.h>
#include <float.h>

using namespace std;

#define N 37
#define SIZE 32

void init(float *h_k, float *h_u, float *h_v){

    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);


    for(int i = 0; i < N*N; i ++){
        // float number = distribution(generator);
        h_k[i] = i;
    }
    
    for(int i = 0; i < N; i ++){
        h_u[i] = 1.0f/N;
        h_v[i] = 1.0f/N;
    }

    return ;
}

__device__ __forceinline__
void atomicMaxFloat(float *addr, float val){
    atomicMax((int*)addr, __float_as_int(val));
} 

__global__ void matrix_transpose_cuda(float *d_t_k, float *d_k){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(row < N && col < N){
        d_t_k[col*N + row] = d_k[row*N + col];
    }
}

__global__ void sinkhorn_log_cuda(float *d_k, float *add, float *res){
    
    int row = blockIdx.x;
    int col = threadIdx.x;
    int tid = threadIdx.x;
    
    float t_max = -FLT_MAX;
    float sum = 0;
    // shared_max store the max value for each row
    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];

    // use local memory for eac threads
    for(int i = col; i < N; i += blockDim.x){
        int idx = row * N + i;
        t_max = fmaxf(t_max, d_k[idx] + logf(add[i]));
    }

    shared_max[tid] = t_max;
    __syncthreads();

   for(int i = blockDim.x; i > 0; i /= 2){
        if(tid + i < N){
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid+i]);
        }
        __syncthreads();
   }
//    shared_max[0] store the maximum for each row;
   for(int i = col; i < N; i += blockDim.x){
        int idx = row * N + i;
        sum += expf(d_k[idx] + logf(add[i]) - shared_max[0]); 
   }

   shared_sum[tid] = sum;
   __syncthreads();

   for(int i = blockDim.x; i > 0; i /= 2){
        if(tid + i < N){
            shared_sum[tid] += shared_sum[tid+i];
        }
        __syncthreads(); 
    }

    if(tid == 0){
        res[row] = logf(1) - (logf(shared_sum[0]) + shared_max[0]);
    }

}

int main(){

    size_t bytes = sizeof(float) * N * N;

    float *h_k, *h_u, *h_v, *h_row_max, *h_k_t;
    float *d_k, *d_u, *d_v, *d_row_max, *d_t_k;

    // for h_cost
    h_k = (float*)malloc(bytes);
    h_k_t = (float*)malloc(bytes); 
    h_u = (float*)malloc(sizeof(float) * N);
    h_v = (float*)malloc(sizeof(float) * N);
    h_row_max = (float*)malloc(sizeof(float) * N);
    init(h_k, h_u, h_v);

    // cuda memeory allocation for GPU
    cudaMalloc(&d_k, bytes);
    cudaMalloc(&d_t_k, bytes);
    cudaMalloc(&d_u, sizeof(float) * N);
    cudaMalloc(&d_v, sizeof(float) * N);
    cudaMalloc(&d_row_max, sizeof(float) * N);
     

    //copy memeory from host to device
    cudaMemcpy(d_k, h_k, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, h_u, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, sizeof(float) * N, cudaMemcpyHostToDevice);


    matrix_transpose_cuda<<<dim3((N + SIZE - 1) / SIZE, (N + SIZE - 1) / SIZE), dim3(SIZE, SIZE)>>>(d_t_k, d_k);

    int BLOCK_SIZE = min(SIZE, 1024);
    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 threads(SIZE);
    dim3 grid(N);

    // calculate the sinkhorn log cuda
    sinkhorn_log_cuda<<<grid, threads>>>(d_t_k, d_u, d_v);
    sinkhorn_log_cuda<<<grid, threads>>>(d_k, d_v, d_u);

    // cudaMemcpy(h_k_t, d_t_k, bytes, cudaMemcpyDeviceToHost);
    // calculate the maximum value for each row
    // cudaMemcpy(h_row_max, d_row_max, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // for(int i = 0; i < N; i ++){
    //     std::cout << "id = " << i << " max value = " << h_row_max[ i ] << std::endl;
    // }
    
    // for(int i = 0; i < N; i ++){
    //     for(int j = 0; j < N; j ++){
    //         std::cout<< h_k_t[i*N+j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    

    return 0;
}