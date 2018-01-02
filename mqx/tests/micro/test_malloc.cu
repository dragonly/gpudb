// memory region allocation
#include <stdio.h>
#include <cuda.h>

#include "test.h"
#include "mqx.h"

int test_malloc()
{
    size_t size_free, size_total;
    size_t size = 1024;
    void *dptr = NULL;

    if (cudaMemGetInfo(&size_free, &size_total) != cudaSuccess) {
        MQX_TPRINT("Cannot get device memory info");
        return -1;
    }

    while (size < size_free - 1024L * 1024L * 64L) {
        MQX_TPRINT("Allocating %lu bytes", size);
        if (cudaMalloc(&dptr, size) != cudaSuccess) {
            MQX_TPRINT("cudaMalloc failed");
            return -1;
        }
        //mqx_print_dptr(dptr);
        if (cudaFree(dptr) != cudaSuccess) {
            MQX_TPRINT("cudaFree failed");
            return -1;
        }
        size *= 2;
    }

    return 0;
}
