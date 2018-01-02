// memory copy
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mqx.h"
#include "test.h"

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

int cmp_data(void *data1, void *data2, size_t size)
{
    unsigned char *c1 = (unsigned char *)data1;
    unsigned char *c2 = (unsigned char *)data2;
    size_t i;

    for (i = 0; i < size; i++) {
        if (*(c1 + i) < *(c2 + i)) {
            MQX_TPRINT("Diff: i = %lu, c1 = %d, c2 = %d",
                    i, *(c1 + i), *(c2 + i));
            return -1;
        }
        else if (*(c1 + i) > *(c2 + i)) {
            MQX_TPRINT("Diff: i = %lu, c1 = %d, c2 = %d",
                    i, *(c1 + i), *(c2 + i));
            return 1;
        }
    }
    return 0;
}

int do_test_memcpy(size_t size)
{
    void *dptr, *dptr2, *ptr, *ptr2, *ptrcmp;
    int ret = 0;

    // Mallocs
    ptr = malloc(size);
    if (!ptr) {
        MQX_TPRINT("malloc failed for ptr");
        return -1;
    }

    ptr2 = malloc(size);
    if (!ptr2) {
        MQX_TPRINT("malloc failed for ptr2");
        free(ptr);
        return -1;
    }

    ptrcmp = malloc(size);
    if (!ptrcmp) {
        MQX_TPRINT("malloc failed for ptrcmp");
        free(ptr2);
        free(ptr);
        return -1;
    }

    if (cudaMalloc(&dptr, size) != cudaSuccess) {
        MQX_TPRINT("cudaMalloc failed for dptr");
        free(ptrcmp);
        free(ptr2);
        free(ptr);
        return -1;
    }

    if (cudaMalloc(&dptr2, size) != cudaSuccess) {
        MQX_TPRINT("cudaMalloc failed for dptr2");
        cudaFree(dptr);
        free(ptrcmp);
        free(ptr2);
        free(ptr);
        return -1;
    }

    // Initialize source buffer
    init_rand_data(ptrcmp, size);
    memcpy(ptr, ptrcmp, size);

    if (cudaMemcpy(dptr, ptr, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        MQX_TPRINT("cudaMemcpy HtoD failed");
        ret = -1;
        goto finish;
    }
    //MQX_TPRINT("cudaMemcpyHostToDevice succeeded");

    if (cudaMemcpy(dptr2, dptr, size, cudaMemcpyDeviceToDevice) != cudaSuccess) {
        MQX_TPRINT("cudaMemcpy DtoD failed");
        ret = -1;
        goto finish;
    }
    //MQX_TPRINT("cudaMemcpyDeviceToDevice succeeded");

    if (cudaMemcpy(ptr2, dptr2, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        MQX_TPRINT("cudaMemcpy DtoH failed");
        ret = -1;
        goto finish;
    }
    //MQX_TPRINT("cudaMemcpyDeviceToHost succeeded");

    if (cmp_data(ptr2, ptrcmp, size) != 0) {
        MQX_TPRINT("Memcpy test of size %lu: verification failed", size);
        ret = -1;
    }
    else
        MQX_TPRINT("Memcpy test of size %lu: verification passed", size);

finish:
    if (cudaFree(dptr) != cudaSuccess) {
        MQX_TPRINT("cudaFree for dptr failed");
    }
    if (cudaFree(dptr2) != cudaSuccess) {
        MQX_TPRINT("cudaFree for dptr2 failed");
    }
    free(ptrcmp);
    free(ptr);
    free(ptr2);

    return ret;
}

int test_memcpy()
{
    bool test_failed = false;
    size_t size = 4096;

    srand(time(NULL));

    while (size < 1024 * 1024 * 10) {
        test_failed |= do_test_memcpy(size) < 0;
        size *= 2;
    }

    return test_failed ? -1 : 0;
}
