// device pointer arrays
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mqx.h"
#include "test.h"

__global__ void kernel_inc(int **data, int dim, int count)
{
    int tot_threads = gridDim.x * blockDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (; i < count; i += tot_threads) {
        for (int j = 0; j < dim; j++)
            data[j][i]++;
    }
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

int test_ptarray()
{
    int **dptr, *dptr1, *dptr2, *ptr, *ptr2;
    int count = 1024 * 1024 * 10;
    size_t size = sizeof(int) * count;
    int nk, ret = 0;

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
    if (cudaMalloc(&dptr1, size) != cudaSuccess) {
        MQX_TPRINT("cudaMalloc failed for dptr1");
        free(ptr2);
        free(ptr);
        return -1;
    }
    if (cudaMalloc(&dptr2, size) != cudaSuccess) {
        MQX_TPRINT("cudaMalloc failed for dptr2");
        cudaFree(dptr1);
        free(ptr2);
        free(ptr);
        return -1;
    }
    if (cudaMallocEx((void **)&dptr, sizeof(int *) * 2, FLAG_PTARRAY)
            != cudaSuccess) {
        MQX_TPRINT("cudaMalloc failed for dptr");
        cudaFree(dptr2);
        cudaFree(dptr1);
        free(ptr2);
        free(ptr);
        return -1;
    }

    srand(time(NULL));
    init_rand_data(ptr, size);

    if (cudaMemcpy(dptr1, ptr, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        MQX_TPRINT("cudaMemcpyHostToDevice failed");
        ret = -1;
        goto finish;
    }
    if (cudaMemcpy(dptr2, ptr, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        MQX_TPRINT("cudaMemcpyHostToDevice failed");
        ret = -1;
        goto finish;
    }
    if (cudaMemcpy(dptr, &dptr1, sizeof(int *), cudaMemcpyHostToDevice)
            != cudaSuccess) {
        MQX_TPRINT("cudaMemcpyHostToDevice failed");
        ret = -1;
        goto finish;
    }
    if (cudaMemcpy(dptr + 1, &dptr2, sizeof(int *), cudaMemcpyHostToDevice)
            != cudaSuccess) {
        MQX_TPRINT("cudaMemcpyHostToDevice failed");
        ret = -1;
        goto finish;
    }
    MQX_TPRINT("cudaMemcpyHostToDevice succeeded");

    // Launch nk kernels
    nk = 2;
    do {
        if (cudaAdvise(0, CADV_INPUT | CADV_PTADEFAULT) != cudaSuccess) {
            MQX_TPRINT("cudaAdvise failed");
            ret = -1;
            goto finish;
        }
        kernel_inc<<<256, 128>>>(dptr, 2, count);
    } while (--nk > 0);

    if (cudaDeviceSynchronize() != cudaSuccess) {
        MQX_TPRINT("cudaThreadSynchronize returned error");
        ret = -1;
        goto finish;
    }
    else
        MQX_TPRINT("Kernel finished successfully");

    if (cudaMemcpy(ptr2, dptr1, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        MQX_TPRINT("cudaMemcpy DtoH failed");
        ret = -1;
        goto finish;
    }
    MQX_TPRINT("First cudaMemcpyDeviceToHost succeeded");

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
    MQX_TPRINT("Second cudaMemcpyDeviceToHost succeeded");

    for(int i = 0; i < count; i++)
        if (ptr2[i] != ptr[i] + 2) {
            MQX_TPRINT("Verification failed: i = %d, ptr = %d, ptr2 = %d",
                    i, ptr[i], ptr2[i]);
            ret = -1;
            goto finish;
        }
    MQX_TPRINT("Second verification passed");

finish:
    if (cudaFree(dptr) != cudaSuccess) {
        MQX_TPRINT("cudaFree failed");
    }
    if (cudaFree(dptr2) != cudaSuccess) {
        MQX_TPRINT("cudaFree failed");
    }
    if (cudaFree(dptr1) != cudaSuccess) {
        MQX_TPRINT("cudaFree failed");
    }
    free(ptr2);
    free(ptr);

    return ret;
}
