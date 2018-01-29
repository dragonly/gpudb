/*
 * Copyright (c) 2017 Yilong Li <liyilongko@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef _LIBMPSSERVER_H_
#define _LIBMPSSERVER_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "list.h"
#include "protocol.h"
#include "libmps.h"

cudaError_t mpsserver_cudaMalloc(void **devPtr, size_t size, uint32_t flags);
cudaError_t mpsserver_cudaFree(void *devPtr);
cudaError_t mpsserver_cudaMemcpy(struct mps_client *client, void *dst, void *src, size_t size, enum cudaMemcpyKind kind);
cudaError_t mpsserver_cudaMemset(struct mps_client *client, void *dst, int32_t value, size_t size);
cudaError_t mpsserver_cudaAdvise(struct mps_client *client, int iarg, int advice);
cudaError_t mpsserver_cudaSetFunction(struct mps_client *client, int index);
cudaError_t mpsserver_cudaConfigureCall(struct mps_client *client, dim3 gridDim, dim3 blockDim, size_t sharedMem, CUstream stream);
cudaError_t mpsserver_cudaSetupArgument(struct mps_client *client, void *arg, size_t size, size_t offset);
cudaError_t mpsserver_cudaLaunchKernel(struct mps_client *client);
cudaError_t mpsserver_cudaMemGetInfo(size_t *free, size_t *total);
cudaError_t mpsserver_cudaDeviceSynchronize(struct mps_client *client);
cudaError_t mpsserver_cudaEventCreate(cudaEvent_t *event);
cudaError_t mpsserver_cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t mpsserver_cudaEventRecord(struct mps_client *client, cudaEvent_t event, cudaStream_t stream);
cudaError_t mpsserver_cudaEventSynchronize(cudaEvent_t event);
cudaError_t mpsserver_cudaGetDevice(int *device);
int dma_channel_init(struct mps_dma_channel *channel, int isHtoD);
void dma_channel_destroy(struct mps_dma_channel *channel);
struct mps_region *find_allocated_region(struct global_context*, const void*);
cudaError_t mpsserver_getColumnBlockAddress(void **devPtr, const char *colname, uint32_t iblock);
cudaError_t mpsserver_getColumnBlockHeader(struct columnHeader **ph, const char *colname, uint32_t iblock);

#endif

