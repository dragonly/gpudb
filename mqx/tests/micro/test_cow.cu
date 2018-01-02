// copy-on-write
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

int test_cow_modify(size_t size)
{
    void *dptr, *ptr, *ptr2, *ptrcmp;
    int ret = 0;

    // Mallocs
    ptr = malloc(size);
    if (!ptr) {
        MQX_TPRINT("malloc failed for ptr");
        return -1;
    }

    ptr2 = malloc(size);
    if (!ptr) {
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
        MQX_TPRINT("cudaMalloc for dptr failed");
        free(ptrcmp);
        free(ptr2);
        free(ptr);
        return -1;
    }

    // Initialize source buffer
    init_rand_data(ptrcmp, size);
    memcpy(ptr, ptrcmp, size);

    // Transferring data from ptr to dptr triggers COW within MQX library.
    if (cudaMemcpy(dptr, ptr, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        MQX_TPRINT("cudaMemcpy HtoD failed");
        ret = -1;
        goto finish;
    }
    //MQX_TPRINT("cudaMemcpyHostToDevice succeeded");

    // By modifying a COW page within ptr, the data in ptr will now
    // be copied to dptr.
    *((char *)ptr + size / 2) = *((char *)ptrcmp + size / 2);
    //MQX_TPRINT("User source buffer modified");

    // Now dptr should contain the data from ptr.
    // Copy the data out for verification.
    if (cudaMemcpy(ptr2, dptr, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        MQX_TPRINT("cudaMemcpy DtoH failed");
        ret = -1;
        goto finish;
    }
    //MQX_TPRINT("cudaMemcpyDeviceToHost succeeded");

    // Verify that data in ptr2 is the same with that in ptrcmp.
    if (cmp_data(ptr2, ptrcmp, size) != 0) {
        MQX_TPRINT("COW_modify test of size %lu: verification failed", size);
        ret = -1;
    }
    else
        MQX_TPRINT("COW_modify test of size %lu: verification passed", size);

finish:
    if (cudaFree(dptr) != cudaSuccess) {
        MQX_TPRINT("cudaFree for dptr failed");
    }
    free(ptrcmp);
    free(ptr2);
    free(ptr);
    return ret;
}

int test_cow_free(size_t size)
{
    void *dptr, *ptr = NULL, *ptr2, *ptrcmp;
    int ret = 0;

    // Mallocs
    ptr = malloc(size);
    if (!ptr) {
        MQX_TPRINT("malloc failed for ptr");
        return -1;
    }

    ptr2 = malloc(size);
    if (!ptr) {
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
        MQX_TPRINT("cudaMalloc for dptr failed");
        free(ptrcmp);
        free(ptr2);
        free(ptr);
        return -1;
    }

    // Initialize source buffer
    init_rand_data(ptrcmp, size);
    memcpy(ptr, ptrcmp, size);

    // Transferring data from ptr to dptr triggers COW within MQX library.
    if (cudaMemcpy(dptr, ptr, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        MQX_TPRINT("cudaMemcpy HtoD failed");
        ret = -1;
        goto finish;
    }
    //MQX_TPRINT("cudaMemcpyHostToDevice succeeded");

    // By freeing the COW source buffer, the data in ptr will
    // be copied to dptr.
    //MQX_TPRINT("Freeing COW source buffer");
    free(ptr);
    ptr = NULL;
    //MQX_TPRINT("COW source buffer freed");

    // Now dptr should contain the data from ptr.
    // Copy the data out for verification.
    if (cudaMemcpy(ptr2, dptr, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        MQX_TPRINT("cudaMemcpy DtoH failed");
        ret = -1;
        goto finish;
    }
    //MQX_TPRINT("cudaMemcpyDeviceToHost succeeded");

    // Verify that data in ptr2 is the same with that in ptrcmp
    if (cmp_data(ptr2, ptrcmp, size) != 0) {
        MQX_TPRINT("COW_free test of size %lu: verification failed", size);
        ret = -1;
    }
    else
        MQX_TPRINT("COW_free test of size %lu: verification passed", size);

finish:
    if (cudaFree(dptr) != cudaSuccess) {
        MQX_TPRINT("cudaFree for dptr failed");
    }
    free(ptrcmp);
    free(ptr2);
    if (ptr)
        free(ptr);
    return ret;
}

int test_cow()
{
    bool test_failed = false;
    size_t size = 4096;

    srand(time(NULL));

    while (size < 1024 * 1024 * 10) {
        test_failed |= test_cow_modify(size) < 0;
        test_failed |= test_cow_free(size) < 0;
        size *= 2;
    }

    return test_failed ? -1 : 0;
}
