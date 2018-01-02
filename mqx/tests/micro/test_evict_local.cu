// local evictions
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

int test_evict_local()
{
    int *dptr, *dptr2, *ptr, *ptr2;
    size_t size, sfree, stotal;
    int count, ret = 0;

    if (cudaMemGetInfo(&sfree, &stotal) != cudaSuccess) {
        MQX_TPRINT("Failed to get mem info");
        return -1;
    }
    size = stotal * 3 / 4;
    count = size / sizeof(int);

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

    if (cudaMalloc(&dptr2, size) != cudaSuccess) {
        MQX_TPRINT("cudaMalloc failed");
        cudaFree(dptr);
        free(ptr2);
        free(ptr);
        return -1;
    }

    MQX_TPRINT("Initializing input data");
    srand(time(NULL));
    init_rand_data(ptr, size);
    MQX_TPRINT("Input data initialized");

    if (cudaMemcpy(dptr, ptr, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        MQX_TPRINT("cudaMemcpyHostToDevice to dptr failed");
        ret = -1;
        goto finish;
    }
    MQX_TPRINT("cudaMemcpyHostToDevice succeeded for dptr");
    if (cudaMemcpy(dptr2, ptr, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        MQX_TPRINT("cudaMemcpyHostToDevice to deptr2 failed");
        ret = -1;
        goto finish;
    }
    MQX_TPRINT("cudaMemcpyHostToDevice succeeded for dptr2");

    if (cudaAdvise(0, CADV_DEFAULT) != cudaSuccess) {
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
        MQX_TPRINT("First kernel finished");

    // Accessing dptr2, dptr will be evicted.
    if (cudaAdvise(0, CADV_DEFAULT) != cudaSuccess) {
        MQX_TPRINT("cudaAdvise failed");
        ret = -1;
        goto finish;
    }
    kernel_inc<<<256, 128>>>(dptr2, count);

    if (cudaDeviceSynchronize() != cudaSuccess) {
        MQX_TPRINT("cudaThreadSynchronize returned error");
        ret = -1;
        goto finish;
    }
    else
        MQX_TPRINT("Second kernel finished");

    // Re-accessing dptr, dptr2 will be evicted.
    if (cudaAdvise(0, CADV_DEFAULT) != cudaSuccess) {
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
        MQX_TPRINT("Third kernel finished");

    if (cudaMemcpy(ptr2, dptr, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        MQX_TPRINT("cudaMemcpy DtoH failed");
        ret = -1;
        goto finish;
    }
    MQX_TPRINT("cudaMemcpyDeviceToHost succeeded for dptr");

    for(int i = 0; i < count; i++)
        if (ptr2[i] != ptr[i] + 2) {
            MQX_TPRINT("Verification failed: i = %d, ptr = %d, ptr2 = %d",
                    i, ptr[i], ptr2[i]);
            ret = -1;
            goto finish;
        }
    MQX_TPRINT("First verification passed");

    memset(ptr2, 0, size);
    if (cudaMemcpy(ptr2, dptr2, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        MQX_TPRINT("cudaMemcpy DtoH failed");
        ret = -1;
        goto finish;
    }
    MQX_TPRINT("cudaMemcpyDeviceToHost succeeded for dptr2");

    for(int i = 0; i < count; i++)
        if (ptr2[i] != ptr[i] + 1) {
            MQX_TPRINT("Verification failed: i = %d, ptr = %d, ptr2 = %d",
                    i, ptr[i], ptr2[i]);
            ret = -1;
            goto finish;
        }
    MQX_TPRINT("Second verification passed");

finish:
    if (cudaFree(dptr2) != cudaSuccess) {
        MQX_TPRINT("cudaFree failed");
    }
    if (cudaFree(dptr) != cudaSuccess) {
        MQX_TPRINT("cudaFree failed");
    }
    free(ptr2);
    free(ptr);

    return ret;
}
