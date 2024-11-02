#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N 15
#define BLOCK_SIZE 5

#define CUDACHECK(err)                     \
  do                                       \
  {                                        \
    cuda_check((err), __FILE__, __LINE__); \
  } while (false)

inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
  if (error_code != cudaSuccess)
  {
    fprintf(stderr, "CUDA Error %d: '%s'. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
    fflush(stderr);
    exit(error_code);
  }
}

__global__ void indicesOnDevice(int *a, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n)
  {
    a[idx] = blockIdx.x;
  }
}

int main(int argc, char const *argv[])
{
  int *h_a, *d_a;

  int nBlocks = N / BLOCK_SIZE + (N % BLOCK_SIZE == 0 ? 0 : 1);

  // dim3 dimGrid(GRID_SIZE);
  // dim3 dimBlock(BLOCK_SIZE);

  size_t memSize = sizeof(int) * N;

  cudaMallocHost((void **)&h_a, memSize);
  cudaMalloc((void **)&d_a, memSize);

  // cudaMemcpy(d_a, h_a, memSize, cudaMemcpyHostToDevice);

  indicesOnDevice<<<nBlocks, BLOCK_SIZE>>>(d_a, N);

  CUDACHECK(cudaPeekAtLastError());

  cudaMemcpy(h_a, d_a, memSize, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++)
  {
    printf("%d ", h_a[i]);
  }

  printf("\n");

  cudaFreeHost(h_a);
  cudaFree(d_a);

  return 0;
}
