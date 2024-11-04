#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define CUDACHECK(err)                    \
  do                                      \
  {                                       \
    cudaCheck((err), __FILE__, __LINE__); \
  } while (false)

inline void cudaCheck(cudaError_t error_code, const char *file, int line)
{
  if (error_code != cudaSuccess)
  {
    fprintf(stderr, "CUDA Error %d: '%s'. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
    fflush(stderr);
    exit(error_code);
  }
}

void parseArgsInt(char *arg, int *val)
{
  char *cp;
  long lVal;

  cp = arg;

  if (*cp == 0)
  {
    fprintf(stderr, "[ERROR] Argument '%s' is an empty string\n", arg);
    exit(1);
  }

  lVal = strtol(cp, &cp, 10);

  if (*cp != 0)
  {
    fprintf(stderr, "[ERROR] Argument '%s' is not an integer -- '%s'\n", arg, cp);
    exit(1);
  }

  *val = (int)lVal;
}

void parseArgs(int argc, char *argv[], int *gridSize, int *blockSize, char *logFileName[])
{
  // Check for the right number of arguments
  if (argc != 3)
  {
    fprintf(stderr, "[ERROR] Must be run with exactly 2 argument, found %d!\nUsage: %s <gridSize> <blockSize>\n", argc - 1, argv[0]);
    exit(1);
  }

  parseArgsInt(argv[1], gridSize);
  parseArgsInt(argv[2], blockSize);
}

__global__ void getIndicesOnDevice(int *block, int *thread, int *index)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  block[idx] = blockIdx.x;
  thread[idx] = threadIdx.x;
  index[idx] = idx;
}

int main(int argc, char *argv[])
{
  int gridSize, blockSize, *h_block, *d_block, *h_thread, *d_thread, *h_index, *d_index;
  char *logFileName;

  parseArgs(argc, argv, &gridSize, &blockSize, &logFileName);

  int N = gridSize * blockSize;

  size_t memSize = sizeof(int) * N;

  cudaMallocHost((void **)&h_block, memSize);
  cudaMallocHost((void **)&h_thread, memSize);
  cudaMallocHost((void **)&h_index, memSize);

  cudaMalloc((void **)&d_block, memSize);
  cudaMalloc((void **)&d_thread, memSize);
  cudaMalloc((void **)&d_index, memSize);

  getIndicesOnDevice<<<gridSize, blockSize>>>(d_block, d_thread, d_index);
  CUDACHECK(cudaPeekAtLastError());

  cudaMemcpy(h_block, d_block, memSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_thread, d_thread, memSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_index, d_index, memSize, cudaMemcpyDeviceToHost);

  char *logFileName = (char *)malloc(25 * sizeof(char));

  sprintf(logFileName, "logs/get-indices_%02d-%02d.csv", gridSize, blockSize);

  FILE *logFile = fopen(logFileName, "a");
  fprintf(logFile, "i,block,thread,index\n");

  printf("\nGrid Size: %6d  Block Size: %6d\n", gridSize, blockSize);

  printf("-------------------------------------\n");
  printf("|      i |  block | thread |  index |\n");
  printf("-------------------------------------\n");

  for (int i = 0; i < N; i++)
  {
    printf("| %6d | %6d | %6d | %6d |\n", i, h_block[i], h_thread[i], h_index[i]);
    fprintf(logFile, "%d,%d,%d,%d\n", i, h_block[i], h_thread[i], h_index[i]);
  }

  printf("-------------------------------------\n");
  printf("                            N: %6d\n\n", N);

  cudaFreeHost(h_block);
  cudaFree(d_block);

  return 0;
}
