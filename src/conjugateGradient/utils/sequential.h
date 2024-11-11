#ifndef SEQUENTIAL_H_
#define SEQUENTIAL_H_

void solveCG_seq(float *A, float *b, float *x, float *r_norm, int *cnt, int N, int MAX_ITER, float EPS);
int moreOrLessEqual(float *a, float *b, int N, float TOL);

#endif /* SEQUENTIAL_H_ */
