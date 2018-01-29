#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include "common.h"
#include "interfaces.h"
#include "libmpsclient.h"

cudaError_t (*nv_cudaMemcpy)(void *, const void *, size_t, enum cudaMemcpyKind) = NULL;

volatile uint8_t mps_initialized = 0;

__attribute__((constructor)) void mps_init() {
  DEFAULT_API_POINTER("cudaMemcpy", nv_cudaMemcpy);
  if (nv_cudaMemcpy == NULL) {
    mqx_print(FATAL, "fail to find CUDA runtime shared library");
    exit(-1);
  }
  if (mpsclient_init()) {
    mqx_print(FATAL, "fail to connect to mps server");
    exit(-1);
  }
  mps_initialized = 1;
}
__attribute__((destructor)) void mps_destroy() {
  mpsclient_destroy();
  mps_initialized = 0;
}
cudaError_t cudaMalloc(void **devPtr, size_t size) {
  return mpsclient_cudaMalloc(devPtr, size, 0);
}
cudaError_t cudaMallocEx(void **devPtr, size_t size, uint32_t flags) {
  return mpsclient_cudaMalloc(devPtr, size, flags);
}
cudaError_t cudaFree(void *devPtr) {
  return mpsclient_cudaFree(devPtr);
}
cudaError_t cudaMemcpy(void *dst, const void *src, size_t size, enum cudaMemcpyKind kind) {
  switch (kind) {
    case cudaMemcpyHostToDevice:
    case cudaMemcpyDeviceToHost:
    case cudaMemcpyDeviceToDevice:
    case cudaMemcpyDefault:
      return mpsclient_cudaMemcpy(dst, src, size, kind);
    case cudaMemcpyHostToHost:
    default:
      return nv_cudaMemcpy(dst, src, size, kind);
  }
}
cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
  return mpsclient_cudaMemset(devPtr, value, count);
}
cudaError_t cudaAdvise(int iarg, int advice) {
  return mpsclient_cudaAdvise(iarg, advice);
}
cudaError_t cudaSetFunction(int index) {
  return mpsclient_cudaSetFunction(index);
}
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
  return mpsclient_cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
}
cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
  return mpsclient_cudaSetupArgument(arg, size, offset);
}
cudaError_t cudaLaunch(const void *func) {
  return mpsclient_cudaLaunch(func);
}
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
  return mpsclient_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}
cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
  return mpsclient_cudaMemGetInfo(free, total);
}
cudaError_t cudaDeviceSynchronize() {
  return mpsclient_cudaDeviceSynchronize();
}
cudaError_t cudaEventCreate(cudaEvent_t *event) {
  return mpsclient_cudaEventCreate(event);
}
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
  return mpsclient_cudaEventElapsedTime(ms, start, end);
}
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  return mpsclient_cudaEventRecord(event, stream);
}
cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  return mpsclient_cudaEventSynchronize(event);
}
cudaError_t cudaGetDevice(int *device) {
  return mpsclient_cudaGetDevice(device);
}
cudaError_t cudaGetColumnBlockAddress(void **devPtr, const char *colname, uint32_t iblock) {
  return mpsclient_cudaGetColumnBlockAddress(devPtr, colname, iblock);
}
cudaError_t cudaGetColumnBlockHeader(struct columnHeader *pheader, const char *colname, uint32_t iblock) {
  return mpsclient_cudaGetColumnBlockHeader(pheader, colname, iblock);
}

