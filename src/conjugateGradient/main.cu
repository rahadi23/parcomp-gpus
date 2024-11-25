/*
 * Implementation of conjugate-gradient for symmetric PSD systems using CUDA.
 *
 * Author: Tim Lebailly
 */
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <sys/stat.h>

#include "../utils/helper.cuh"

extern "C"
{
#include "utils/helper.h"
#include "utils/sequential.h"
#include "../utils/helper.h"
}

// vecVec
#define BLOCK_DIM_VEC 1024

// matVec
#define NB_ELEM_MAT 32
#define BLOCK_SIZE_MAT 32

#define LOG_FILE_FORMAT "logs/conjugateGradient-%d.csv"

/*
 * --Naive implementation--
 * Computes a (square) matrix vector product
 * Input: pointer to 1D-array-stored matrix, 1D-array-stored vector
 * Stores the product in memory at the location of the pointer out
 */
__global__ void matVec(float *A, float *b, float *out, int N)
{
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x < N)
	{
		float tmp = 0;
		for (int i = 0; i < N; i++)
		{
			tmp += b[i] * A[N * index_x + i];
		}
		out[index_x] = tmp;
	}
}

/*
 * --More efficient implementation--
 * Computes a (square) symmetric matrix vector product
 * Input: pointer to 1D-array-stored matrix, 1D-array-stored vector
 * Stores the product in memory at the location of the pointer out
 */
__global__ void matVec2(float *A, float *b, float *out, int N)
{
	__shared__ float b_shared[NB_ELEM_MAT];

	int effective_block_width;
	if ((blockIdx.x + 1) * NB_ELEM_MAT <= N)
	{
		effective_block_width = NB_ELEM_MAT;
	}
	else
	{
		// needed to avoid overflow in next row
		effective_block_width = N % NB_ELEM_MAT;
	}

	if (threadIdx.x < effective_block_width)
		b_shared[threadIdx.x] = b[blockIdx.x * NB_ELEM_MAT + threadIdx.x];

	__syncthreads();

	int idy = blockIdx.y * BLOCK_SIZE_MAT + threadIdx.x;
	float tmp_scal = 0.0;
	// threads outside matrix dimension are not needed (vertical)
	if (idy < N)
	{
		for (int i = 0; i < effective_block_width; i++)
		{
			// take advantage of symmetric matrix for coalesced memory access
			tmp_scal += b_shared[i] * A(blockIdx.x * NB_ELEM_MAT + i, idy, N);
		}
		atomicAdd(out + idy, tmp_scal);
	}
}

/*
 * Computes the sum of 2 vectors
 * Input: pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 * Stores the sum in memory at the location of the pointer out
 */
__global__ void vecPlusVec(float *a, float *b, float *out, int N)
{
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x < N)
	{
		out[index_x] = b[index_x] + a[index_x];
	}
}

/*
 * Computes the sum of 2 vectors
 * Input: pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 * Stores the sum in memory at the location of the pointer out
 * Also 0's the vector b
 */
__global__ void vecPlusVec2(float *a, float *b, float *out, int N)
{
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x < N)
	{
		out[index_x] = b[index_x] + a[index_x];
		b[index_x] = 0.0;
	}
}

/*
 * Computes the difference of 2 vectors
 * Input: pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 * Stores the sum in memory at the location of the pointer out
 */
__global__ void vecMinVec(float *a, float *b, float *out, int N)
{
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x < N)
	{
		out[index_x] = a[index_x] - b[index_x];
	}
}

/*
 * --Naive implementation--
 * Computes the inner product of 2 vectors
 * Input: pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 * Stores the product in memory at the location of the pointer out
 */
__global__ void vecVec(float *a, float *b, float *out, int N)
{
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	float tmp = 0.0;
	if (index_x == 0)
	{
		for (int i = 0; i < N; i++)
		{
			tmp += b[i] * a[i];
		}
		*out = tmp;
	}
}

/*
 * --More efficient implementation--
 * Computes the inner product of 2 vectors
 * Input: pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 * Stores the product in memory at the location of the pointer out
 */
__global__ void vecVec2(float *a, float *b, float *out, int N)
{
	// each block has it's own shared_tmp of size BLOCK_DIM_VEC
	__shared__ float shared_tmp[BLOCK_DIM_VEC];

	// needed for atomicAdd
	if (threadIdx.x + blockDim.x * blockIdx.x == 0)
	{
		*out = 0.0;
	}

	if (blockIdx.x * blockDim.x + threadIdx.x < N)
	{
		shared_tmp[threadIdx.x] = a[blockIdx.x * blockDim.x + threadIdx.x] * b[blockIdx.x * blockDim.x + threadIdx.x];
	}
	else
	{
		// needed for the reduction
		shared_tmp[threadIdx.x] = 0.0;
	}

	// reduction within block
	for (int i = blockDim.x / 2; i >= 1; i = i / 2)
	{
		// threads access memory position written by other threads so sync is needed
		__syncthreads();
		if (threadIdx.x < i)
		{
			shared_tmp[threadIdx.x] += shared_tmp[threadIdx.x + i];
		}
	}

	// atomic add the partial reduction in out
	if (threadIdx.x == 0)
	{
		atomicAdd(out, shared_tmp[0]);
	}
}

/*
 * Computes the product of a scalar with a vector
 * Input: pointer to scalar, pointer to 1D-array-stored vector
 * Stores the sum in memory at the location of the pointer out
 */
__global__ void scalarVec(float *scalar, float *a, float *out, int N)
{
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x < N)
	{
		out[index_x] = a[index_x] * *scalar;
	}
}

/*
 * Copies the content of vector in to vector out
 * Input: pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 */
__global__ void memCopy(float *in, float *out, int N)
{
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x < N)
	{
		out[index_x] = in[index_x];
	}
}

/*
 * Computes the quotient of 2 scalars
 * Input: pointer to scalar, pointer to scalar
 * Stores the quotient in memory at the location of the pointer out
 */
__global__ void divide(float *num, float *den, float *out)
{
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x == 0)
	{
		*out = *num / *den;
	}
}

/*
 * Main CG solver
 * All the given pointers are device pointers, with correct initial values
 */
void solveCG_cuda(float *A, float *b, float *x, float *p, float *r, float *temp,
									float *alpha, float *beta, float *r_norm, float *r_norm_old,
									float *temp_scal, float *h_x, float *h_r_norm, int *cnt,
									int N, int maxIter, float eps)
{

	dim3 vec_block_dim(BLOCK_DIM_VEC);
	dim3 vec_grid_dim((N + BLOCK_DIM_VEC - 1) / BLOCK_DIM_VEC);

	dim3 mat_grid_dim((N + NB_ELEM_MAT - 1) / NB_ELEM_MAT, (N + BLOCK_SIZE_MAT - 1) / BLOCK_SIZE_MAT);
	dim3 mat_block_dim(BLOCK_SIZE_MAT);

	vecVec2<<<vec_grid_dim, vec_block_dim>>>(r, r, r_norm_old, N);
	int k = 0;
	while ((k < maxIter) && (*h_r_norm > eps))
	{
		// temp = A * p (only compute matrix vector product once)
		matVec2<<<mat_grid_dim, mat_block_dim>>>(A, p, temp, N);

		// alpha_k = ...
		vecVec2<<<vec_grid_dim, vec_block_dim>>>(p, temp, temp_scal, N);
		divide<<<1, 1>>>(r_norm_old, temp_scal, alpha);

		// r_{k+1} = ...
		scalarVec<<<vec_grid_dim, vec_block_dim>>>(alpha, temp, temp, N);
		vecMinVec<<<vec_grid_dim, vec_block_dim>>>(r, temp, r, N);

		// x_{k+1} = ...
		scalarVec<<<vec_grid_dim, vec_block_dim>>>(alpha, p, temp, N);
		vecPlusVec<<<vec_grid_dim, vec_block_dim>>>(x, temp, x, N);

		// beta_k = ...
		vecVec2<<<vec_grid_dim, vec_block_dim>>>(r, r, r_norm, N);
		divide<<<1, 1>>>(r_norm, r_norm_old, beta);

		// p_{k+1} = ...
		scalarVec<<<vec_grid_dim, vec_block_dim>>>(beta, p, temp, N);
		vecPlusVec2<<<vec_grid_dim, vec_block_dim>>>(r, temp, p, N);

		// set r_norm_old to r_norm
		memCopy<<<1, 1>>>(r_norm, r_norm_old, N);

		// copy to r_norm to CPU (to evaluate stop condition)
		cudaMemcpy(h_r_norm, r_norm, sizeof(float), cudaMemcpyDeviceToHost);
		k++;
	}

	*cnt = k;
	// printf("Time spent gpu per iter [s]: %e\n", (float)((micro_end_gpu - micro_begin_gpu) / k) / 1e6);
}

void parseArgs(int argc, char *argv[], int *NMin, int *NMax, int *NMult, int *MAX_ITER, float *EPS, float *TOL)
{
	// Check for the right number of arguments
	if (argc != 7)
	{
		fprintf(stderr, "[ERROR] Must be run with exactly 6 argument, found %d!\nUsage: %s <NMin> <NMax> <NMult> <MAX_ITER> <EPS> <TOL>\n", argc - 1, argv[0]);
		exit(1);
	}

	parseArgsInt(argv[1], NMin);
	parseArgsInt(argv[2], NMax);
	parseArgsInt(argv[3], NMult);
	parseArgsInt(argv[4], MAX_ITER);
	parseArgsFloat(argv[5], EPS);
	parseArgsFloat(argv[6], TOL);
}

////////////////////////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[])
{
	int j, NMin, NMax, NMult, NIter, MAX_ITER;
	float EPS, TOL;

	parseArgs(argc, argv, &NMin, &NMax, &NMult, &MAX_ITER, &EPS, &TOL);

	struct stat buffer;

	int logId = 1;
	char *logFileNameWithId = (char *)malloc(sizeof(char) * strlen(LOG_FILE_FORMAT));

	do
	{
		sprintf(logFileNameWithId, LOG_FILE_FORMAT, logId);
		logId++;
	} while (stat(logFileNameWithId, &buffer) == 0);

	FILE *log_file = fopen(logFileNameWithId, "w");
	fprintf(log_file, "j,N,grid_size,block_size,is_ok,gpu_time,cpu_time,gpu_r_norm,cpu_r_norm,gpu_iter,cpu_iter,speedup\n");
	fclose(log_file);

	printf("\n----------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
	printf("|          N |   gridSize |  blockSize |      isOk |         gpuTime |         cpuTime |        gpuRNorm |        cpuRNorm |   gpuIter |   cpuIter |         speedUp |\n");
	printf("|            |   (nBlock) |  (nThread) |           |            (ms) |            (ms) |                 |                 |           |           |                 |\n");
	printf("----------------------------------------------------------------------------------------------------------------------------------------------------------------------\n");

	NIter = log10(NMax / NMin) / log10(NMult) + 1;

	for (j = 0; j < NIter; j++)
	{
		int N = N = NMin * pow(NMult, j);

		// allocate host memory
		float *h_A = generateA(N);
		float *h_b = generateb(N);
		float *h_x = (float *)calloc(N, sizeof(float));
		float *h_r_norm = (float *)malloc(sizeof(float));
		*h_r_norm = 1.0;

		// times
		int gpu_cnt, cpu_cnt;
		float cpu_r_norm, gpu_elapsed_time_ms, cpu_elapsed_time_ms;

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		// allocate device memory
		float *d_A;
		float *d_b;
		float *d_x;
		float *d_p;
		float *d_r;
		float *d_temp;
		cudaMalloc((void **)&d_A, N * N * sizeof(float));
		cudaMalloc((void **)&d_b, N * sizeof(float));
		cudaMalloc((void **)&d_x, N * sizeof(float));
		cudaMalloc((void **)&d_p, N * sizeof(float));
		cudaMalloc((void **)&d_r, N * sizeof(float));
		cudaMalloc((void **)&d_temp, N * sizeof(float));

		// copy host memory to device
		cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
		// assume x0 = 0
		cudaMemcpy(d_p, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_r, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

		// 4 floats needed
		float *d_beta;
		float *d_alpha;
		float *d_r_norm;
		float *d_r_norm_old;
		float *d_temp_scal;
		cudaMalloc((void **)&d_beta, sizeof(float));
		cudaMalloc((void **)&d_alpha, sizeof(float));
		cudaMalloc((void **)&d_r_norm, sizeof(float));
		cudaMalloc((void **)&d_r_norm_old, sizeof(float));
		cudaMalloc((void **)&d_temp_scal, sizeof(float));

		cudaEventRecord(start, 0);

		// run the main function
		solveCG_cuda(d_A, d_b, d_x, d_p, d_r, d_temp, d_alpha, d_beta, d_r_norm,
								 d_r_norm_old, d_temp_scal, h_x, h_r_norm, &gpu_cnt, N, MAX_ITER, EPS);

		CUDACHECK(cudaPeekAtLastError());

		// allocate memory for the result on host side
		cudaDeviceSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		// compute time elapse on GPU computing
		cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);

		// copy result from device to host
		cudaMemcpy(h_x, d_x, sizeof(float) * N, cudaMemcpyDeviceToHost);

		// compare output with sequential version
		float *h_x_seq = (float *)calloc(N, sizeof(float));

		cudaEventRecord(start, 0);

		solveCG_seq(h_A, h_b, h_x_seq, &cpu_r_norm, &cpu_cnt, N, MAX_ITER, EPS);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		// compute time elapse on CPU computing
		cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);

		// assert(moreOrLessEqual(h_x, h_x_seq, N, TOL) == 1);
		int resultIsOk = moreOrLessEqual(h_x, h_x_seq, N, TOL) == 1;
		int gridSize = (N + BLOCK_DIM_VEC - 1) / BLOCK_DIM_VEC;
		int blockSize = BLOCK_DIM_VEC;
		float speedup = cpu_elapsed_time_ms / gpu_elapsed_time_ms;

		FILE *log_file = fopen(logFileNameWithId, "a");
		fprintf(log_file, "%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%d,%d,%.6f\n",
						j, N, gridSize, blockSize, resultIsOk, gpu_elapsed_time_ms,
						cpu_elapsed_time_ms, *h_r_norm, cpu_r_norm, gpu_cnt, cpu_cnt, speedup);
		fclose(log_file);

		printf("| %10d | %10d | %10d | %9d | %15.6f | %15.6f | %15.9e | %15.9e | %9d | %9d | %15.6f |\n",
					 N, gridSize, blockSize, resultIsOk, gpu_elapsed_time_ms,
					 cpu_elapsed_time_ms, *h_r_norm, cpu_r_norm, gpu_cnt, cpu_cnt, speedup);

		// printf("\nAssertion passed!\n");

		// cleanup memory host
		free(h_A);
		free(h_b);
		free(h_x);
		free(h_r_norm);

		// cleanup memory device
		cudaFree(d_A);
		cudaFree(d_b);
		cudaFree(d_x);
		cudaFree(d_p);
		cudaFree(d_r);
		cudaFree(d_temp);
		cudaFree(d_alpha);
		cudaFree(d_beta);
		cudaFree(d_r_norm);
		cudaFree(d_r_norm_old);
		cudaFree(d_temp_scal);
	}

	printf("----------------------------------------------------------------------------------------------------------------------------------------------------------------------\n\n");

	free(logFileNameWithId);

	return 0;
}
