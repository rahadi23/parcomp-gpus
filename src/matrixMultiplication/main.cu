#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../utils/helper.cuh"

extern "C"
{
#include "../utils/helper.h"
}

/*
*********************************************************************
function name: gpu_mult

description: dot product of two square matrix

parameters:
            &a GPU device pointer to a n X n matrix (A)
            &b GPU device pointer to a n X n matrix (B)
            &c GPU device output purpose pointer to a n X n matrix (C)
            to store the result

Note:
    grid and block should be configured as:
        dim3 dimGrid((k + blockSize - 1) / blockSize, (m + blockSize - 1) / blockSize);
        dim3 dimBlock(blockSize, blockSize);

return: none
*********************************************************************
*/
__global__ void gpu_mult(int *a, int *b, int *c, int N)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int sum = 0;
  if (col < N && row < N)
  {
    for (int i = 0; i < N; i++)
    {
      sum += a[row * N + i] * b[i * N + col];
    }
    c[row * N + col] = sum;
  }
}

/*
*********************************************************************
function name: gpu_mult_shared

description: dot product of two square matrix in GPU
             by using shared memory

parameters:
            &a GPU device pointer to a n X n matrix (A)
            &b GPU device pointer to a n X n matrix (B)
            &c GPU device output purpose pointer to a n X n matrix (C)
            to store the result
Note:
    grid and block should be configured as:

        dim3 dim_grid((n - 1) / blockSize + 1, (n - 1) / blockSize + 1, 1);
        dim3 dim_block(blockSize, blockSize, 1);

return: none
*********************************************************************
*/
__global__ void gpu_mult_shared(int *d_a, int *d_b, int *d_result, int N, int blockSize)
{
  extern __shared__ int **tile_a;
  extern __shared__ int **tile_b;

  int row = blockIdx.y * blockSize + threadIdx.y;
  int col = blockIdx.x * blockSize + threadIdx.x;
  int tmp = 0;
  int idx;

  for (int sub = 0; sub < gridDim.x; ++sub)
  {
    idx = row * N + sub * blockSize + threadIdx.x;
    if (idx >= N * N)
    {
      // n may not divisible by blockSize
      tile_a[threadIdx.y][threadIdx.x] = 0;
    }
    else
    {
      tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
    }

    idx = (sub * blockSize + threadIdx.y) * N + col;
    if (idx >= N * N)
    {
      tile_b[threadIdx.y][threadIdx.x] = 0;
    }
    else
    {
      tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
    }

    __syncthreads();

    for (int k = 0; k < blockSize; ++k)
    {
      tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < N && col < N)
  {
    d_result[row * N + col] = tmp;
  }
}

/*
*********************************************************************
function name: cpu_mult

description: dot product of two square matrix in CPU,
             for validating GPU results

parameters:
            &a CPU host pointer to a N X N matrix (A)
            &b CPU host pointer to a N X N matrix (B)
            &c CPU host output purpose pointer to a N X N matrix (C)
            to store the result
return: none
*********************************************************************
*/
void cpu_mult(int *h_a, int *h_b, int *h_result, int N)
{
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      int tmp = 0.0;
      for (int h = 0; h < N; ++h)
      {
        tmp += h_a[i * N + h] * h_b[h * N + j];
      }
      h_result[i * N + j] = tmp;
    }
  }
}

void parseArgs(int argc, char *argv[], int *N, int *blockSize)
{
  // Check for the right number of arguments
  if (argc != 3)
  {
    fprintf(stderr, "[ERROR] Must be run with exactly 2 argument, found %d!\nUsage: %s <N> <blockSize>\n", argc - 1, argv[0]);
    exit(1);
  }

  parseArgsInt(argv[1], N);
  parseArgsInt(argv[2], blockSize);
}

int main(int argc, char *argv[])
{
  int N, blockSize;

  parseArgs(argc, argv, &N, &blockSize);

  // allocate memory in host RAM
  int *h_a, *h_b, *h_c, *hs_c, *h_cc;
  cudaMallocHost((void **)&h_a, sizeof(int) * N * N);
  cudaMallocHost((void **)&h_b, sizeof(int) * N * N);
  cudaMallocHost((void **)&h_c, sizeof(int) * N * N);
  cudaMallocHost((void **)&hs_c, sizeof(int) * N * N);
  cudaMallocHost((void **)&h_cc, sizeof(int) * N * N);

  // random initialize matrix A
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      h_a[i * N + j] = rand() % 1024;
    }
  }

  // random initialize matrix B
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      h_b[i * N + j] = rand() % 1024;
    }
  }

  unsigned int grid_rows = (N + blockSize - 1) / blockSize;
  unsigned int grid_cols = (N + blockSize - 1) / blockSize;
  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(blockSize, blockSize);

  float gpu_elapsed_time_ms, gpu_shared_elapsed_time_ms, cpu_elapsed_time_ms;

  // some events to count the execution time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // start to count execution time of GPU version
  cudaEventRecord(start, 0);

  // initialize GPU
  int *d_a, *d_b, *d_c;

  // Allocate memory space on the device
  cudaMalloc((void **)&d_a, sizeof(int) * N * N);
  cudaMalloc((void **)&d_b, sizeof(int) * N * N);
  cudaMalloc((void **)&d_c, sizeof(int) * N * N);

  // copy matrix A and B from host to device memory
  cudaMemcpy(d_a, h_a, sizeof(int) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, sizeof(int) * N * N, cudaMemcpyHostToDevice);

  gpu_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);

  // Transfer results from device to host
  cudaMemcpy(h_c, d_c, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // time counting terminate
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // compute time elapse on GPU computing
  cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
  printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", N, N, N, N, gpu_elapsed_time_ms);

  // start to count execution time of GPU with shared memory version
  cudaEventRecord(start, 0);

  // initialize GPU
  int *ds_a, *ds_b, *ds_c;

  // Allocate memory space on the device
  cudaMalloc((void **)&ds_a, sizeof(int) * N * N);
  cudaMalloc((void **)&ds_b, sizeof(int) * N * N);
  cudaMalloc((void **)&ds_c, sizeof(int) * N * N);

  // copy matrix A and B from host to device memory
  cudaMemcpy(ds_a, h_a, sizeof(int) * N * N, cudaMemcpyHostToDevice);
  cudaMemcpy(ds_b, h_b, sizeof(int) * N * N, cudaMemcpyHostToDevice);

  gpu_mult_shared<<<dimGrid, dimBlock, blockSize * blockSize * sizeof(int)>>>(ds_a, ds_b, ds_c, N, blockSize);

  // Transfer results from device to host
  cudaMemcpy(hs_c, ds_c, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // time counting terminate
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // compute time elapse on GPU with shared memory computing
  cudaEventElapsedTime(&gpu_shared_elapsed_time_ms, start, stop);
  printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU with shared memory: %f ms.\n\n", N, N, N, N, gpu_shared_elapsed_time_ms);

  // start the CPU version
  cudaEventRecord(start, 0);

  cpu_mult(h_a, h_b, h_cc, N);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
  printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n\n", N, N, N, N, cpu_elapsed_time_ms);

  // validate results computed by GPU
  int all_ok = 1;
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      // printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_cc[i*k + j], i, j, h_c[i*k + j]);
      if (h_cc[i * N + j] != h_c[i * N + j] || h_cc[i * N + j] != hs_c[i * N + j])
      {
        all_ok = 0;
      }
    }
    // printf("\n");
  }

  // roughly compute speedup
  if (all_ok)
  {
    printf("all results are correct!!!, speedup = %f, speedup shared = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms, cpu_elapsed_time_ms / gpu_shared_elapsed_time_ms);
  }
  else
  {
    printf("incorrect results\n");
  }

  // free memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(ds_a);
  cudaFree(ds_b);
  cudaFree(ds_c);
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);
  cudaFreeHost(hs_c);
  cudaFreeHost(h_cc);

  return 0;
}
