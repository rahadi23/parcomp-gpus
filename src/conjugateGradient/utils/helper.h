#ifndef HELPER_H_
#define HELPER_H_

#define A(row, col, N) (A[(row) * N + (col)])
#define b(x) (b[(x)])

float *generateA(int N);
float *generateb(int N);
void printMat(float *A, int N);
void printVec(float *b, int N);
float getMaxDiffSquared(float *a, float *b, int N);
long getMicrotime();

#endif /* HELPER_H_ */
