#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef USE_MQX
#include "mqx.h"
#endif

#define TVAL(t)         ((t).tv_sec * 1000.0 + (t).tv_usec / 1000.0)
#define TDIFF(t1, t2)   (TVAL(t2) - TVAL(t1))

#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(call) \
    do { \
        cudaError_t err = call; \
        if(cudaSuccess != err) { \
            fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
#endif

#define BLOCK_SIZE  16
#define WIDTH       (BLOCK_SIZE * 128)
#define HEIGHT      WIDTH

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

//float multiplication kernel called by MatMul()
__global__ void MatMulKernel(float *A, float *B, float *C)
{
	// Each thread computes one element of C by accumulating results into Cvalue
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	for (int e = 0; e < WIDTH; ++e)
		Cvalue += A[row * WIDTH + e] * B[e * WIDTH + col];
	C[row * WIDTH + col] = Cvalue;
}

// float multiplication - Host code
// float dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const float *A, const float *B, float *C)
{
	size_t size = WIDTH * HEIGHT * sizeof(float);
	float *d_A, *d_B, *d_C;
	struct timeval t1, t2;

	gettimeofday(&t1, NULL);

	// Load A and B to device memory
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_A, size));
	CUDA_SAFE_CALL(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMalloc((void**)&d_B, size));
	CUDA_SAFE_CALL(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

	// Allocate C in device memory
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_C, size));

	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(WIDTH / dimBlock.x, HEIGHT / dimBlock.y);

#ifdef USE_MQX
	CUDA_SAFE_CALL(cudaAdvise(0, CADV_INPUT));
	CUDA_SAFE_CALL(cudaAdvise(1, CADV_INPUT));
	CUDA_SAFE_CALL(cudaAdvise(2, CADV_OUTPUT));
#endif
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	// Read C from device memory
	cudaMemcpy(C, d_C, size,cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	gettimeofday(&t2, NULL);
	printf("Matrix multiplication took %f ms\n", TDIFF(t1, t2));
}

int main(int argc, char* argv[])
{
	unsigned int size = WIDTH*HEIGHT;
	unsigned int mem_size = sizeof(float) * size;
	float *h_A, *h_B, *h_C;

	// Allocate host memory for matrices A and B
	h_A = (float*)malloc(mem_size);
	h_B = (float*)malloc(mem_size);
	h_C = (float*)malloc(mem_size);

	// set seed for rand()
	srand(2014);

	// initialize host memory
	randomInit(h_A, size);
	randomInit(h_B, size);

	//invoke MatMul
	MatMul(h_A,h_B,h_C);

	free(h_C);
	free(h_B);
	free(h_A);
	return 0;
}
