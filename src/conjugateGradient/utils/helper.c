#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <sys/time.h>

#include "helper.h"

/*
 * Generates a dense PSD symmetric SIZE x SIZE matrix.
 */
float *generateA(int N)
{
	int i, j;
	float *A = malloc(sizeof(float) * N * N);

	float temp;
	for (i = 0; i < N; i++)
	{
		for (j = 0; j <= i; j++)
		{
			// temp = (float)rand() / RAND_MAX;
			temp = (float)rand();
			if (i == j)
			{
				A(i, j, N) = temp + N;
			}
			else
			{
				A(i, j, N) = temp;
				A(j, i, N) = temp;
			}
		}
	}
	return A;
}

/*
 * Returns the time in microseconds
 * Taken from https://gist.github.com/sevko/d23646ba07c77c15fde9
 */
long getMicrotime()
{
	struct timeval currentTime;
	gettimeofday(&currentTime, NULL);
	return currentTime.tv_sec * (int)1e6 + currentTime.tv_usec;
}

/*
 * Generates a random vector of size SIZE
 */
float *generateb(int N)
{
	int i;
	float *b = malloc(sizeof(float) * N);
	for (i = 0; i < N; i++)
	{
		// b[i] = (float)rand() / RAND_MAX;
		b[i] = (float)rand();
	}
	return b;
}

/*
 * Prints a formated (square) matrix
 * Input: pointer to 1D-array-stored matrix (row major)
 */
void printMat(float *A, int N)
{
	int i;
	for (i = 0; i < N * N; i++)
	{
		printf("%.3e ", A[i]);
		if ((i + 1) % N == 0)
		{
			printf("\n");
		}
	}
	printf("\n");
}

/*
 * Prints a formated vector
 * Input: pointer to 1D-array-stored vector
 */
void printVec(float *b, int N)
{
	printf("__begin_vector__\n");
	int i;
	for (i = 0; i < N; i++)
	{
		if (b[i] > 0)
		{
			printf("+%.6e\n", b[i]);
		}
		else
		{
			printf("%.6e\n", b[i]);
		}
	}
	printf("__end_vector__\n");
}

/*
 * Computes the max squared difference of 2 vectors
 */
float getMaxDiffSquared(float *a, float *b, int N)
{
	float max = 0.0;
	float tmp;
	int i;
	for (i = 0; i < N; i++)
	{
		tmp = (a[i] - b[i]) * (a[i] - b[i]);
		if (tmp > max)
		{
			max = tmp;
		}
	}
	return max;
}
