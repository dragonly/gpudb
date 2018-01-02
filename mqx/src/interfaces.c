/*
 * Copyright (c) 2014 Kaibo Wang (wkbjerry@gmail.com)
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
// This file contains the interfaces intercepted and enhanced by MQX
// for managing GPU resources. Other functions added by MQX also reside here.

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdint.h>

#include "advice.h"
#include "client.h"
#include "common.h"
#include "core.h"
#include "cow.h"
#include "interfaces.h"
#include "protocol.h"

// Handlers to default CUDA runtime APIs.
#ifdef MQX_SET_MAPHOST
cudaError_t (*nv_cudaSetDeviceFlags)(unsigned int) = NULL;
#endif
cudaError_t (*nv_cudaMalloc)(void **, size_t) = NULL;
cudaError_t (*nv_cudaFree)(void *) = NULL;
cudaError_t (*nv_cudaMemcpy)(void *, const void *, size_t, enum cudaMemcpyKind) = NULL;
cudaError_t (*nv_cudaMemcpyAsync)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t) = NULL;
cudaError_t (*nv_cudaStreamCreate)(cudaStream_t *) = NULL;
cudaError_t (*nv_cudaStreamDestroy)(cudaStream_t) = NULL;
cudaError_t (*nv_cudaStreamSynchronize)(cudaStream_t) = NULL;
cudaError_t (*nv_cudaMemGetInfo)(size_t *, size_t *) = NULL;
cudaError_t (*nv_cudaSetupArgument)(const void *, size_t, size_t) = NULL;
cudaError_t (*nv_cudaConfigureCall)(dim3, dim3, size_t, cudaStream_t) = NULL;
cudaError_t (*nv_cudaMemset)(void *, int, size_t) = NULL;
cudaError_t (*nv_cudaMemsetAsync)(void *, int, size_t, cudaStream_t) = NULL;
#ifdef MQX_CONFIG_PROFILE
cudaError_t (*nv_cudaDeviceSynchronize)(void) = NULL;
#endif
cudaError_t (*nv_cudaLaunch)(const void *) = NULL;
cudaError_t (*nv_cudaStreamAddCallback)(cudaStream_t, cudaStreamCallback_t, void *, unsigned int) = NULL;

// Indicates whether the MQX environment has been initialized successfully.
volatile int initialized = 0;

// The library constructor.
// The order of initialization matters. First, link to the default
// CUDA API implementations, because CUDA interfaces have been
// intercepted by our library and we should be able to redirect
// CUDA calls to their default implementations if MQX environment
// fails to initialize successfully. Then, initialize MQX local
// context. Finally, after the local context has been initialized
// successfully, we connect our local context to the global context
// and join the party.
__attribute__((constructor)) void mqx_init(void) {
#ifdef MQX_SET_MAPHOST
  DEFAULT_API_POINTER("cudaSetDeviceFlags", nv_cudaSetDeviceFlags);
#endif
  DEFAULT_API_POINTER("cudaMalloc", nv_cudaMalloc);
  DEFAULT_API_POINTER("cudaFree", nv_cudaFree);
  DEFAULT_API_POINTER("cudaMemcpy", nv_cudaMemcpy);
  DEFAULT_API_POINTER("cudaMemcpyAsync", nv_cudaMemcpyAsync);
  DEFAULT_API_POINTER("cudaStreamCreate", nv_cudaStreamCreate);
  DEFAULT_API_POINTER("cudaStreamDestroy", nv_cudaStreamDestroy);
  DEFAULT_API_POINTER("cudaStreamSynchronize", nv_cudaStreamSynchronize);
  DEFAULT_API_POINTER("cudaMemGetInfo", nv_cudaMemGetInfo);
  DEFAULT_API_POINTER("cudaSetupArgument", nv_cudaSetupArgument);
  DEFAULT_API_POINTER("cudaConfigureCall", nv_cudaConfigureCall);
  DEFAULT_API_POINTER("cudaMemset", nv_cudaMemset);
  DEFAULT_API_POINTER("cudaMemsetAsync", nv_cudaMemsetAsync);
#ifdef MQX_CONFIG_PROFILE
  DEFAULT_API_POINTER("cudaDeviceSynchronize", nv_cudaDeviceSynchronize);
#endif
  DEFAULT_API_POINTER("cudaLaunch", nv_cudaLaunch);
  DEFAULT_API_POINTER("cudaStreamAddCallback", nv_cudaStreamAddCallback);

  mqx_print_init();
  if (context_init()) {
    mqx_print(FATAL, "Failed to initialize MQX local context.");
    return;
  }
  if (client_init()) {
    mqx_print(FATAL, "Failed to attach to MQX global context.");
    context_fini();
    return;
  }
  if (cow_init()) {
    mqx_print(FATAL, "Failed to initialize copy-on-write handler.");
    // NOTE: context_fini has to happen before client_fini.
    // See mqx_fini for reasons.
    context_fini();
    client_fini();
    return;
  }

#ifdef MQX_SET_MAPHOST
  // If the user program wants to map pinned host memory to device
  // address space, cudaDeviceMapHost device flag must be set before
  // any other CUDA functions are called. This has to be done here.
  nv_cudaSetDeviceFlags(cudaDeviceMapHost);
#endif

  // Warm up the underlying CUDA runtime environment.
  // The user program may call cudaMalloc and cudaFree to warm up
  // the device. But since these APIs have been intercepted by our
  // library, and device memory space is not immediately allocated,
  // we have to invoke the default cudaMalloc and cudaFree functions
  // to warm up the device here.
  {
    void *dummy;
    if (nv_cudaMalloc(&dummy, 1 << 10) != cudaSuccess) {
      mqx_print(FATAL, "Dummy CUDA malloc failed.");
      cow_fini();
      // NOTE: context_fini has to happen before client_fini.
      // See mqx_fini for reasons.
      context_fini();
      client_fini();
      return;
    }
    nv_cudaFree(dummy);
  }
  initialized = 1;
  mqx_print(DEBUG, "MQX initialized.");
}

// The library destructor.
__attribute__((destructor)) void mqx_fini(void) {
  if (initialized) {
    cow_fini();
    // NOTE: context_fini has to happen before client_fini
    // because garbage collections need to update global info.
    // XXX: Possible bugs if client thread is busy when context
    // is being freed?
    context_fini();
    client_fini();
    initialized = 0;
  }
  mqx_print(DEBUG, "MQX finished.");
  // NOTE: mqx_print_fini has to be the last called function before
  // the library is unloaded so that all mqx_print messages can get
  // recorded and printed correctly if MQX_PRINT_BUFFER is enabled.
  mqx_print_fini();
}

MQX_EXPORT
cudaError_t cudaMalloc(void **devPtr, size_t size) {
  cudaError_t ret;

  if (initialized)
    ret = mqx_cudaMalloc(devPtr, size, 0);
  else {
    mqx_print(WARN, "cudaMalloc called when MQX is uninitialized.");
    mqx_profile("cudaMalloc begin %lu", size);
    ret = nv_cudaMalloc(devPtr, size);
    mqx_profile("cudaMalloc end %p", *devPtr);
  }

  return ret;
}

// MQX-specific: device memory allocations with flags
// such as FLAG_PTARRAY (`device memory pointer array').
MQX_EXPORT
cudaError_t cudaMallocEx(void **devPtr, size_t size, int flags) {
  cudaError_t ret;

  if (initialized)
    ret = mqx_cudaMalloc(devPtr, size, flags);
  else {
    mqx_print(WARN, "cudaMallocEx called when MQX is uninitialized.");
    mqx_profile("cudaMalloc begin %lu", size);
    ret = nv_cudaMalloc(devPtr, size);
    mqx_profile("cudaMalloc end %p", *devPtr);
  }

  return ret;
}

// TODO: Add a cudaPin/cudaUnpin function to allow the user to pin/unpin a
// specific region to/from device memory. cudaPin should accept a flag that
// specifies weather pinning should happen immediately or not until the
// region is to be accessed by a kernel.
//
// cudaError_t cudaPin(void *devPtr, int pin_now);
// cudaError_t cudaUnpin(void *devPtr);

MQX_EXPORT
cudaError_t cudaFree(void *devPtr) {
  cudaError_t ret;

  if (initialized)
    ret = mqx_cudaFree(devPtr);
  else {
    mqx_print(WARN, "cudaFree called when MQX is uninitialized.");
    mqx_profile("cudaFree begin %p", devPtr);
    ret = nv_cudaFree(devPtr);
    mqx_profile("cudaFree end %p", devPtr);
  }

  return ret;
}

MQX_EXPORT
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
  cudaError_t ret;

  if (initialized) {
    if (kind == cudaMemcpyHostToDevice)
      ret = mqx_cudaMemcpyHtoD(dst, src, count);
    else if (kind == cudaMemcpyDeviceToHost)
      ret = mqx_cudaMemcpyDtoH(dst, src, count);
    else if (kind == cudaMemcpyDeviceToDevice)
      ret = mqx_cudaMemcpyDtoD(dst, src, count);
    else if (kind == cudaMemcpyDefault)
      ret = mqx_cudaMemcpyDefault(dst, src, count);
    else {
      // Host-to-host memory copy does not need to go
      // through MQX's management.
      ret = nv_cudaMemcpy(dst, src, count, kind);
    }
  } else {
    mqx_print(WARN, "cudaMemcpy called when MQX is uninitialized.");
    mqx_profile("cudaMemcpy begin %lu %d", count, kind);
    ret = nv_cudaMemcpy(dst, src, count, kind);
    mqx_profile("cudaMemcpy end %lu %d", count, kind);
  }

  return ret;
}

// TODO: cudaMemcpyAsync

MQX_EXPORT
cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
  cudaError_t ret;

  if (initialized)
    ret = mqx_cudaMemset(devPtr, value, count);
  else {
    mqx_print(WARN, "cudaMemset called when MQX is uninitialized.");
    mqx_profile("cudaMemset begin %lu", count);
    ret = nv_cudaMemset(devPtr, value, count);
    mqx_profile("cudaMemset end %lu", count);
  }

  return ret;
}

// TODO: cudaMemsetAsync

MQX_EXPORT
cudaError_t cudaMemGetInfo(size_t *size_free, size_t *size_total) {
  cudaError_t ret;

  if (initialized)
    ret = mqx_cudaMemGetInfo(size_free, size_total);
  else {
    mqx_print(WARN, "cudaMemGetInfo called when MQX is uninitialized.");
    ret = nv_cudaMemGetInfo(size_free, size_total);
  }

  return ret;
}

MQX_EXPORT
cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
  cudaError_t ret;

  if (initialized)
    ret = mqx_cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
  else {
    mqx_print(WARN, "cudaConfigureCall called when MQX is uninitialized.");
    mqx_profile("cudaConfigureCall");
    ret = nv_cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
  }

  return ret;
}

MQX_EXPORT
cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
  cudaError_t ret;

  if (initialized)
    ret = mqx_cudaSetupArgument(arg, size, offset);
  else {
    mqx_print(WARN, "cudaSetupArgument called when MQX is uninitialized.");
    if (size == sizeof(void *))
      mqx_profile("cudaSetupArgument %lu %p", size, *(void **)arg);
    else
      mqx_profile("cudaSetupArgument %lu", size);
    ret = nv_cudaSetupArgument(arg, size, offset);
  }

  return ret;
}

MQX_EXPORT
cudaError_t cudaLaunch(const void *entry) {
  cudaError_t ret;

  if (initialized)
    ret = mqx_cudaLaunch(entry);
  else {
    mqx_print(WARN, "cudaLaunch called when MQX is uninitialized.");
    mqx_profile("cudaLaunch begin");
    ret = nv_cudaLaunch(entry);
#ifdef MQX_CONFIG_PROFILE
    if (nv_cudaDeviceSynchronize() != cudaSuccess)
      mqx_profile("cudaLaunch end unsuccessfully");
    else
      mqx_profile("cudaLaunch end successfully");
#endif
  }

  return ret;
}

// For passing data reference and access advices before each kernel launch.
// %nadvices indicates the total number of advices given by the user.
// For each advice (advice_refs[i], advice_accs[i]), advice_refs[i] tells
// which argument is a device memory pointer or pointer array that is to
// be referenced in the next kernel launch, and advice_accs[i] tells how
// it is going to be accessed, which can be INPUT, OUTPUT, or BOTH.
// TODO: Should prepare the following data structures for each stream.
volatile int advice_refs[MAX_ARGS];
volatile int advice_accs[MAX_ARGS];
volatile int nadvices = 0;

MQX_EXPORT
cudaError_t NOcudaAdvise(int which_arg, int advice) { return 0; }
// MQX-specific: pass data reference and access advices.
// %which_arg tells which argument (starting with 0) in the following
// cudaSetupArgument calls is a device memory pointer or pointer array.
// %advice is the data access flag. The MQX runtime expects to see call
// sequence like:
// cudaAdvice, ..., cudaAdvice, cudaConfigureCall,
// cudaSetupArgument, ..., cudaSetupArgument, cudaLaunch.
MQX_EXPORT
cudaError_t cudaAdvise(int which_arg, int advice) {
  int i;

  mqx_print(DEBUG, "cudaAdvise: %d %d", which_arg, advice);
  if (!initialized) {
#ifdef MQX_CONFIG_PROFILE
    mqx_profile("cudaAdvise %d", which_arg);
    // Allow cudaAdvice to return success and, therefore, the
    // user program to continue execution, in profiling mode.
    return cudaSuccess;
#else
    return cudaErrorInitializationError;
#endif
  }

  if (which_arg >= 0 && which_arg < MAX_ARGS) {
    for (i = 0; i < nadvices; i++) {
      if (advice_refs[i] == which_arg)
        break;
    }
    if (i == nadvices) {
      advice_refs[nadvices] = which_arg;
      advice_accs[nadvices] = advice;
      nadvices++;
    } else {
      advice_accs[i] |= advice;
    }
  } else {
    mqx_print(ERROR, "Bad cudaAdvise argument %d (max %d).", which_arg, MAX_ARGS - 1);
    return cudaErrorInvalidValue;
  }

  return cudaSuccess;
}
