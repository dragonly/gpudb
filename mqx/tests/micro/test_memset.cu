// memset
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

int do_test_memset(size_t size, size_t memset_off, size_t memset_len)
{
    void *dptr, *ptr, *ptr2;
    int memset_value = 11;
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

    if (cudaMalloc(&dptr, size) != cudaSuccess) {
        MQX_TPRINT("cudaMalloc failed for dptr");
        free(ptr2);
        free(ptr);
        return -1;
    }

    // Initialize source buffer and the device memory region
    init_rand_data(ptr, size);
    if (cudaMemcpy(dptr, ptr, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        MQX_TPRINT("cudaMemcpy HtoD failed");
        ret = -1;
        goto finish;
    }
    //MQX_TPRINT("Device memory region initialized");

    if (cudaMemset((char *)dptr + memset_off, memset_value, memset_len)
            != cudaSuccess) {
        MQX_TPRINT("cudaMemset failed");
        ret = -1;
        goto finish;
    }
    //MQX_TPRINT("cudaMemset succeeded");

    if (cudaMemcpy(ptr2, dptr, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        MQX_TPRINT("cudaMemcpy DtoH failed");
        ret = -1;
        goto finish;
    }
    //MQX_TPRINT("cudaMemcpyDeviceToHost succeeded");

    memset((char *)ptr + memset_off, memset_value, memset_len);
    if (cmp_data(ptr2, ptr, size) != 0) {
        MQX_TPRINT("Memset test of size(%lu) off(%lu) len(%lu): "
        		"verification failed", size, memset_off, memset_len);
        ret = -1;
    }
    else
        MQX_TPRINT("Memset test of size(%lu) off(%lu) len(%lu): "
        		"verification passed", size, memset_off, memset_len);

finish:
    if (cudaFree(dptr) != cudaSuccess) {
        MQX_TPRINT("cudaFree for dptr failed");
    }
    free(ptr);
    free(ptr2);

    return ret;
}

int test_memset()
{
    bool test_failed = false;
    size_t size = 4096;

    srand(time(NULL));

    while (size < 1024 * 1024 * 10) {
        // Partial memset
        test_failed |= do_test_memset(size, size / 4, size / 2) < 0;
        // Complete memset
        test_failed |= do_test_memset(size, 0, size) < 0;
        size *= 2;
    }

    return test_failed ? -1 : 0;
}
