# Conjugate Gradient with CUDA

Adapted from [tileb1/CG-CUDA](https://github.com/tileb1/CG-CUDA)

To compile, include all of the code using the following command (assuming we are on the root directory of the project):

```bash
nvcc ./src/conjugateGradient/helper.c ./src/conjugateGradient/sequential.c ./src/conjugateGradient/main.cu -o ./out/conjugateGradient.o

```
