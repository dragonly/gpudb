/*
    Copyright (c) 2012-2013 The Ohio State University.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#ifndef __GPU_CUDALIB_H_
#define __GPU_CUDALIB_H_

#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>

#define CUDA_SAFE_CALL_NO_SYNC( call) do {                               \
    cudaError_t err = call;                                              \
    if( cudaSuccess != err) {                                            \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %d.\n", __FILE__, __LINE__, err);  \
        exit(EXIT_FAILURE);                                              \
    } } while (0)

#define CUDA_SAFE_CALL( call) do {                                       \
    CUDA_SAFE_CALL_NO_SYNC(call);                                        \
    cudaError_t err = cudaDeviceSynchronize();                           \
    if( cudaSuccess != err) {                                            \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %d.\n", __FILE__, __LINE__, err);  \
        exit(EXIT_FAILURE);                                              \
    } } while (0)


#define CUDA_SAFE_CALL_NO_SYNC_RETRY( call) do {                         \
    cudaError_t err = call;                                              \
    while( cudaSuccess != err) {                                         \
    err = call;                             \
    } } while (0)

static size_t getGpuGlobalMem(int deviceID){
        size_t free = 0, total = 0;

        cudaMemGetInfo(&free,&total);
        return total;
}

static long getAvailMem(void){
    long mem = sysconf(_SC_AVPHYS_PAGES) * sysconf(_SC_PAGE_SIZE);
    return mem;
}

static double getCurrentTime(void){
    struct timeval tv;
        gettimeofday(&tv, NULL);
        double curr  = (tv.tv_sec) * 1000 + (tv.tv_usec) / 1000 ;
    return curr;
}

#endif
