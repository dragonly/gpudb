// kernel launch
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mqx.h"
#include "test.h"

__global__ void kernel_inc(int *data, int count)
{
    int tot_threads = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (; i < count; i += tot_threads)
        data[i]++;
}

void init_rand_data(void *data, size_t size)
{
    size_t i;

    for (i = 0; i < size; i += sizeof(int)) {
        if (i + sizeof(int) <= size)
            *((int *)((char *)data + i)) = rand();
        else
            break;
    }
    if (i < size) {
        while (i < size) {
            *((char *)((char *)data + i)) = rand() % 256;
            ++i;
        }
    }
}

int test_launch()
{
    int *dptr = NULL, *ptr = NULL, *ptr2 = NULL;
    int count = 1024 * 1024 * 8;
    size_t size = sizeof(int) * count;
    int ret = 0;

    ptr = (int *)malloc(size);
    if (!ptr) {
        MQX_TPRINT("malloc failed for ptr");
        return -1;
    }
    ptr2 = (int *)malloc(size);
    if (!ptr2) {
        MQX_TPRINT("malloc failed for ptr2");
        free(ptr);
        return -1;
    }
    if (cudaMalloc(&dptr, size) != cudaSuccess) {
        MQX_TPRINT("cudaMalloc failed");
        free(ptr2);
        free(ptr);
        return -1;
    }

    MQX_TPRINT("Initializing input data");
    srand(time(NULL));
    init_rand_data(ptr, size);
    MQX_TPRINT("Input data initialized");

    if (cudaMemcpy(dptr, ptr, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        MQX_TPRINT("cudaMemcpyHostToDevice failed");
        ret = -1;
        goto finish;
    }
    MQX_TPRINT("cudaMemcpyHostToDevice succeeded");

    if (cudaAdvise(0, CADV_INPUT | CADV_OUTPUT) != cudaSuccess) {
        MQX_TPRINT("cudaAdvise failed");
        ret = -1;
        goto finish;
    }
    kernel_inc<<<256, 128>>>(dptr, count);

    if (cudaDeviceSynchronize() != cudaSuccess) {
        MQX_TPRINT("cudaThreadSynchronize returned error");
        ret = -1;
        goto finish;
    }
    else
        MQX_TPRINT("Kernel finished successfully");

    if (cudaMemcpy(ptr2, dptr, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        MQX_TPRINT("cudaMemcpy DtoH failed");
        ret = -1;
        goto finish;
    }
    MQX_TPRINT("cudaMemcpyDeviceToHost succeeded");

    for(int i = 0; i < count; i++)
        if (ptr2[i] != ptr[i] + 1) {
            MQX_TPRINT("Verification failed at i = %d, ptr = %d, ptr2 = %d",
                    i, ptr[i], ptr2[i]);
            ret = -1;
            goto finish;
        }
    MQX_TPRINT("Verification passed");

finish:
    if (cudaFree(dptr) != cudaSuccess) {
        MQX_TPRINT("cudaFree failed");
    }
    free(ptr2);
    free(ptr);

    return ret;
}
