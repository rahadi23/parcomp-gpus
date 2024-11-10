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
  extern __shared__ int tiles[];

  int *tile_a = tiles;
  int *tile_b = (int *)&tiles[blockSize * blockSize];

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
      tile_a[threadIdx.y * blockSize + threadIdx.x] = 0;
    }
    else
    {
      tile_a[threadIdx.y * blockSize + threadIdx.x] = d_a[idx];
    }

    idx = (sub * blockSize + threadIdx.y) * N + col;
    if (idx >= N * N)
    {
      tile_b[threadIdx.y * blockSize + threadIdx.x] = 0;
    }
    else
    {
      tile_b[threadIdx.y * blockSize + threadIdx.x] = d_b[idx];
    }

    __syncthreads();

    for (int k = 0; k < blockSize; ++k)
    {
      tmp += tile_a[threadIdx.y * blockSize + k] * tile_b[k * blockSize + threadIdx.x];
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

void parseArgs(int argc, char *argv[], unsigned long *NMin, unsigned long *NMax, int *NMult, int *blockMin, int *blockMax, int *blockMult)
{
  // Check for the right number of arguments
  if (argc != 7)
  {
    fprintf(stderr, "[ERROR] Must be run with exactly 6 argument, found %d!\nUsage: %s <NMin> <NMax> <NMult> <blockMin> <blockMax> <blockMult>\n", argc - 1, argv[0]);
    exit(1);
  }

  parseArgsULong(argv[1], NMin);
  parseArgsULong(argv[2], NMax);
  parseArgsInt(argv[3], NMult);
  parseArgsInt(argv[4], blockMin);
  parseArgsInt(argv[5], blockMax);
  parseArgsInt(argv[6], blockMult);
}

int main(int argc, char *argv[])
{
  unsigned long NMin, NMax;
  int k, l, NMult, NIter, blockMin, blockMax, blockMult, blockIter;

  parseArgs(argc, argv, &NMin, &NMax, &NMult, &blockMin, &blockMax, &blockMult);

  NIter = log10(NMax / NMin) / log10(NMult) + 1;
  blockIter = log10(blockMax / blockMin) / log10(blockMult) + 1;

  printf("\n----------------------------------------------------------------------------------------------------------------------------\n");
  printf("|         N |  gridSize | blockSize |      isOk |      gpuTime |    gpuShTime |      cpuTime |   gpuSpeedUp | gpuShSpeedUp |\n");
  printf("|           |  (nBlock) | (nThread) |           |         (ms) |         (ms) |         (ms) |              |              |\n");
  printf("----------------------------------------------------------------------------------------------------------------------------\n");

  for (k = 0; k < NIter; k++)
  {
    for (l = 0; l < blockIter; l++)
    {
      unsigned long N = NMin * pow(NMult, k);
      int blockSize = blockMin * pow(blockMult, l);

      size_t NNSize = sizeof(unsigned long) * N * N;

      // allocate memory in host RAM
      int *h_a, *h_b, *h_c, *hs_c, *h_cc;
      cudaMallocHost((void **)&h_a, NNSize);
      cudaMallocHost((void **)&h_b, NNSize);
      cudaMallocHost((void **)&h_c, NNSize);
      cudaMallocHost((void **)&hs_c, NNSize);
      cudaMallocHost((void **)&h_cc, NNSize);

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

      unsigned int gridSize = (N + blockSize - 1) / blockSize;
      dim3 dimGrid(gridSize, gridSize);
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
      cudaMalloc((void **)&d_a, NNSize);
      cudaMalloc((void **)&d_b, NNSize);
      cudaMalloc((void **)&d_c, NNSize);

      // copy matrix A and B from host to device memory
      cudaMemcpy(d_a, h_a, NNSize, cudaMemcpyHostToDevice);
      cudaMemcpy(d_b, h_b, NNSize, cudaMemcpyHostToDevice);

      gpu_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);

      // Transfer results from device to host
      cudaMemcpy(h_c, d_c, NNSize, cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();

      // time counting terminate
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);

      // compute time elapse on GPU computing
      cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);

      // start to count execution time of GPU with shared memory version
      cudaEventRecord(start, 0);

      // initialize GPU
      int *ds_a, *ds_b, *ds_c;

      // Allocate memory space on the device
      cudaMalloc((void **)&ds_a, NNSize);
      cudaMalloc((void **)&ds_b, NNSize);
      cudaMalloc((void **)&ds_c, NNSize);

      // copy matrix A and B from host to device memory
      cudaMemcpy(ds_a, h_a, NNSize, cudaMemcpyHostToDevice);
      cudaMemcpy(ds_b, h_b, NNSize, cudaMemcpyHostToDevice);

      gpu_mult_shared<<<dimGrid, dimBlock, blockSize * blockSize * sizeof(unsigned long) * 2>>>(ds_a, ds_b, ds_c, N, blockSize);

      // Transfer results from device to host
      cudaMemcpy(hs_c, ds_c, NNSize, cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();

      // time counting terminate
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);

      // compute time elapse on GPU with shared memory computing
      cudaEventElapsedTime(&gpu_shared_elapsed_time_ms, start, stop);

      // start the CPU version
      cudaEventRecord(start, 0);

      cpu_mult(h_a, h_b, h_cc, N);

      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);

      // validate results computed by GPU
      int resultIsOk = 1;

      for (int i = 0; i < N; ++i)
      {
        for (int j = 0; j < N; ++j)
        {
          // printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_cc[i*k + j], i, j, h_c[i*k + j]);
          if (h_cc[i * N + j] != h_c[i * N + j] || h_cc[i * N + j] != hs_c[i * N + j])
          {
            resultIsOk = 0;
          }
        }
        // printf("\n");
      }

      // roughly compute speedup
      printf("| %9ld | %9d | %9d | %9d | %12.8f | %12.8f | %12.8f | %12.8f | %12.8f |\n", N, gridSize, blockSize, resultIsOk, gpu_elapsed_time_ms, gpu_shared_elapsed_time_ms, cpu_elapsed_time_ms, cpu_elapsed_time_ms / gpu_elapsed_time_ms, cpu_elapsed_time_ms / gpu_shared_elapsed_time_ms);

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
    }
  }

  printf("----------------------------------------------------------------------------------------------------------------------------\n\n");
  return 0;
}
