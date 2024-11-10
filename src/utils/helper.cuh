#ifndef GLOBAL_HELPER_CUH_
#define GLOBAL_HELPER_CUH_

#define CUDACHECK(err)

inline void cudaCheck(cudaError_t error_code, const char *file, int line);

#endif /* GLOBAL_HELPER_CUH_ */
