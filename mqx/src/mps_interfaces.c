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
cudaError_t (*nv_cudaLaunchKernel)(const void*, dim3, dim3, void**, size_t, cudaStream_t);

volatile uint8_t mps_initialized = 0;

__attribute__((constructor)) void mps_init() {
  DEFAULT_API_POINTER("cudaMalloc", nv_cudaMalloc);
  DEFAULT_API_POINTER("cudaFree", nv_cudaFree);
  DEFAULT_API_POINTER("cudaMemcpy", nv_cudaMemcpy);
  DEFAULT_API_POINTER("cudaLaunchKernel", nv_cudaLaunchKernel);

  if (mpsclient_init()) {
    mqx_print(FATAL, "fail to connect to mps server");
    return;
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
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
  switch (kind) {
    case cudaMemcpyHostToDevice:
      return mpsclient_cudaMemcpyHtoD(dst, src, count);
    case cudaMemcpyDeviceToHost:
      return mpsclient_cudaMemcpyDtoH(dst, src, count);
    case cudaMemcpyDeviceToDevice:
      return mpsclient_cudaMemcpyDtoD(dst, src, count);
    case cudaMemcpyDefault:
      return mpsclient_cudaMemcpyDefault(dst, src, count);
    case cudaMemcpyHostToHost:
    default:
      return nv_cudaMemcpy(dst, src, count, kind);
  }
}
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
  return mpsclient_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}
