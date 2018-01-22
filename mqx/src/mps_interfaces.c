#include <stdint.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "interfaces.h"
#include "libmpsclient.h"
#include "common.h"

cudaError_t (*nv_cudaMalloc)(void **, size_t) = NULL;
cudaError_t (*nv_cudaFree)(void *) = NULL;
cudaError_t (*nv_cudaMemcpy)(void *, const void *, size_t, enum cudaMemcpyKind) = NULL;
cudaError_t (*nv_cudaSetupArgument)(const void *, size_t, size_t) = NULL;
cudaError_t (*nv_cudaConfigureCall)(dim3, dim3, size_t, cudaStream_t) = NULL;
cudaError_t (*nv_cudaLaunch)(const void *) = NULL;
cudaError_t (*nv_cudaLaunchKernel)(const void*, dim3, dim3, void**, size_t, cudaStream_t);

volatile uint8_t mps_initialized = 0;

__attribute__((constructor)) void mps_init() {
  DEFAULT_API_POINTER("cudaMalloc", nv_cudaMalloc);
  DEFAULT_API_POINTER("cudaFree", nv_cudaFree);
  DEFAULT_API_POINTER("cudaMemcpy", nv_cudaMemcpy);
  DEFAULT_API_POINTER("cudaSetupArgument", nv_cudaSetupArgument);
  DEFAULT_API_POINTER("cudaConfigureCall", nv_cudaConfigureCall);
  DEFAULT_API_POINTER("cudaLaunch", nv_cudaLaunch);
  DEFAULT_API_POINTER("cudaLaunchKernel", nv_cudaLaunchKernel);

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
cudaError_t cudaMemset(void *devPtr, int32_t value, size_t count) {
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

