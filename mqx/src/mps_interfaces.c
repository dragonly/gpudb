#include <stdint.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "interfaces.h"
#include "libmpsclient.h"
#include "common.h"

CUresult (*nv_cuMemAlloc)(CUdeviceptr*, size_t) = NULL;
CUresult (*nv_cuMemFree)(CUdeviceptr) = NULL;
CUresult (*nv_cuMemcpy)(CUdeviceptr, CUdeviceptr, size_t) = NULL;
CUresult (*nv_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**);

volatile uint8_t mps_initialized = 0;

__attribute__((constructor)) void mps_init() {
  DEFAULT_API_POINTER("cuMemAlloc", nv_cuMemAlloc);
  DEFAULT_API_POINTER("cuMemFree", nv_cuMemFree);
  DEFAULT_API_POINTER("cuMemcpy", nv_cuMemcpy);
  DEFAULT_API_POINTER("cuLaunchKernel", nv_cuLaunchKernel);

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
      return nv_cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, count);
  }
}

