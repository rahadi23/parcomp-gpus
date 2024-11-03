#include <stdio.h>
#include <assert.h>
#include <cuda.h>

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

void parseArgs(int argc, char *argv[], int *NMin, int *NMax, int *NInc, int *blockMin, int *blockMax, int *blockInc)
{
  // Check for the right number of arguments
  if (argc != 7)
  {
    fprintf(stderr, "[ERROR] Must be run with exactly 6 argument, found %d!\nUsage: %s <NMin> <NMax> <NInc> <blockMin> <blockMax> <blockInc>\n", argc - 1, argv[0]);
    exit(1);
  }

  parseArgsInt(argv[1], NMin);
  parseArgsInt(argv[2], NMax);
  parseArgsInt(argv[3], NInc);
  parseArgsInt(argv[4], blockMin);
  parseArgsInt(argv[5], blockMax);
  parseArgsInt(argv[6], blockInc);
}

void incrementArrayOnHost(float *a, int N)
{
  int i;
  for (i = 0; i < N; i++)
  {
    a[i] = a[i] + 1.f;
  }
}

__global__ void incrementArrayOnDevice(float *a, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
  {
    a[idx] = a[idx] + 1.f;
  }
}

int main(int argc, char *argv[])
{
  int j, k, NMin, NMax, NInc, NIter, blockMin, blockMax, blockInc, blockIter;

  parseArgs(argc, argv, &NMin, &NMax, &NInc, &blockMin, &blockMax, &blockInc);

  NIter = (NMax - NMin) / NInc + 1;
  blockIter = (blockMax - blockMin) / blockInc + 1;

  printf("\n-------------------------------------------------------------------------------\n");
  printf("|         N |   dimSize | blockSize |      isOk |      gpuTime |      cpuTime |\n");
  printf("-------------------------------------------------------------------------------\n");

  for (k = 0; k < NIter; k++)
  {
    for (j = 0; j < blockIter; j++)
    {
      float *a_h, *b_h; // pointers to host memory
      int i, N = NMin + k * NInc, blockSize = blockMin + j * blockInc;
      size_t size = N * sizeof(float);

      // allocate arrays on host
      a_h = (float *)malloc(size);
      b_h = (float *)malloc(size);

      // initialization of host data
      for (i = 0; i < N; i++)
      {
        a_h[i] = (float)i;
      }

      float gpu_elapsed_time_ms, cpu_elapsed_time_ms;

      // some events to count the execution time
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      // start to count execution
      cudaEventRecord(start, 0);

      // allocate array on device
      float *a_d; // pointer to device memory
      cudaMalloc((void **)&a_d, size);

      // copy data from host to device
      cudaMemcpy(a_d, a_h, sizeof(float) * N, cudaMemcpyHostToDevice);

      // do calculation on device:
      // Part 1 of 2. Compute execution configuration
      int dimSize = N / blockSize + (N % blockSize == 0 ? 0 : 1);

      // Part 2 of 2. Call incrementArrayOnDevice kernel
      incrementArrayOnDevice<<<dimSize, blockSize>>>(a_d, N);
      CUDACHECK(cudaPeekAtLastError());

      // Retrieve result from device and store in b_h
      cudaMemcpy(b_h, a_d, sizeof(float) * N, cudaMemcpyDeviceToHost);

      cudaThreadSynchronize();
      // time counting terminate
      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);

      // compute time elapse on GPU computing
      cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);

      // start the CPU version
      cudaEventRecord(start, 0);

      // do calculation on host
      incrementArrayOnHost(a_h, N);

      cudaEventRecord(stop, 0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);

      // check results
      bool resultIsOk = true;

      for (i = 0; i < N; i++)
      {
        resultIsOk = a_h[i] == b_h[i];
      }

      // assert(resultIsOk);
      printf("| %9d | %9d | %9d | %9d | %12.8f | %12.8f |\n", N, dimSize, blockSize, resultIsOk, gpu_elapsed_time_ms, cpu_elapsed_time_ms);

      // cleanup
      free(a_h);
      free(b_h);
      cudaFree(a_d);
    }
  }

  printf("-------------------------------------------------------------------------------\n\n");
}
