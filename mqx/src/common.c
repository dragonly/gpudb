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
#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#ifdef MQX_PRINT_BUFFER
#include "libc_hooks.h" // For __libc_malloc and __libc_free
#endif

inline void __checkCudaErrors( CUresult err, const char *file, const int line )
{
  if (CUDA_SUCCESS != err) {
    mqx_print(FATAL, "CUDA Driver API error = %04d  \"%s\" from file <%s>, line %i.",
            err, getCudaDrvErrorString(err), file, line );
    exit(-1);
  }
}

// Error Code string definitions here
typedef struct
{
	char const *error_string;
	int error_id;
} s_CudaErrorStr;

/**
 * Error codes
 */
s_CudaErrorStr sCudaDrvErrorString[] =
{
	/**
	 * The API call returned with no errors. In the case of query calls, this
	 * can also mean that the operation being queried is complete (see
	 * ::cuEventQuery() and ::cuStreamQuery()).
	 */
	{ "CUDA_SUCCESS", 0 },

	/**
	 * This indicates that one or more of the parameters passed to the API call
	 * is not within an acceptable range of values.
	 */
	{ "CUDA_ERROR_INVALID_VALUE", 1 },

	/**
	 * The API call failed because it was unable to allocate enough memory to
	 * perform the requested operation.
	 */
	{ "CUDA_ERROR_OUT_OF_MEMORY", 2 },

	/**
	 * This indicates that the CUDA driver has not been initialized with
	 * ::cuInit() or that initialization has failed.
	 */
	{ "CUDA_ERROR_NOT_INITIALIZED", 3 },

	/**
	 * This indicates that the CUDA driver is in the process of shutting down.
	 */
	{ "CUDA_ERROR_DEINITIALIZED", 4 },

	/**
	 * This indicates profiling APIs are called while application is running
	 * in visual profiler mode.
	 */
	{ "CUDA_ERROR_PROFILER_DISABLED", 5 },
	/**
	 * This indicates profiling has not been initialized for this context.
	 * Call cuProfilerInitialize() to resolve this.
	 */
	{ "CUDA_ERROR_PROFILER_NOT_INITIALIZED", 6 },
	/**
	 * This indicates profiler has already been started and probably
	 * cuProfilerStart() is incorrectly called.
	 */
	{ "CUDA_ERROR_PROFILER_ALREADY_STARTED", 7 },
	/**
	 * This indicates profiler has already been stopped and probably
	 * cuProfilerStop() is incorrectly called.
	 */
	{ "CUDA_ERROR_PROFILER_ALREADY_STOPPED", 8 },
	/**
	 * This indicates that no CUDA-capable devices were detected by the installed
	 * CUDA driver.
	 */
	{ "CUDA_ERROR_NO_DEVICE (no CUDA-capable devices were detected)", 100 },

	/**
	 * This indicates that the device ordinal supplied by the user does not
	 * correspond to a valid CUDA device.
	 */
	{ "CUDA_ERROR_INVALID_DEVICE (device specified is not a valid CUDA device)", 101 },


	/**
	 * This indicates that the device kernel image is invalid. This can also
	 * indicate an invalid CUDA module.
	 */
	{ "CUDA_ERROR_INVALID_IMAGE", 200 },

	/**
	 * This most frequently indicates that there is no context bound to the
	 * current thread. This can also be returned if the context passed to an
	 * API call is not a valid handle (such as a context that has had
	 * ::cuCtxDestroy() invoked on it). This can also be returned if a user
	 * mixes different API versions (i.e. 3010 context with 3020 API calls).
	 * See ::cuCtxGetApiVersion() for more details.
	 */
	{ "CUDA_ERROR_INVALID_CONTEXT", 201 },

	/**
	 * This indicated that the context being supplied as a parameter to the
	 * API call was already the active context.
	 * \deprecated
	 * This error return is deprecated as of CUDA 3.2. It is no longer an
	 * error to attempt to push the active context via ::cuCtxPushCurrent().
	 */
	{ "CUDA_ERROR_CONTEXT_ALREADY_CURRENT", 202 },

	/**
	 * This indicates that a map or register operation has failed.
	 */
	{ "CUDA_ERROR_MAP_FAILED", 205 },

	/**
	 * This indicates that an unmap or unregister operation has failed.
	 */
	{ "CUDA_ERROR_UNMAP_FAILED", 206 },

	/**
	 * This indicates that the specified array is currently mapped and thus
	 * cannot be destroyed.
	 */
	{ "CUDA_ERROR_ARRAY_IS_MAPPED", 207 },

	/**
	 * This indicates that the resource is already mapped.
	 */
	{ "CUDA_ERROR_ALREADY_MAPPED", 208 },

	/**
	 * This indicates that there is no kernel image available that is suitable
	 * for the device. This can occur when a user specifies code generation
	 * options for a particular CUDA source file that do not include the
	 * corresponding device configuration.
	 */
	{ "CUDA_ERROR_NO_BINARY_FOR_GPU", 209 },

	/**
	 * This indicates that a resource has already been acquired.
	 */
	{ "CUDA_ERROR_ALREADY_ACQUIRED", 210 },

	/**
	 * This indicates that a resource is not mapped.
	 */
	{ "CUDA_ERROR_NOT_MAPPED", 211 },

	/**
	 * This indicates that a mapped resource is not available for access as an
	 * array.
	 */
	{ "CUDA_ERROR_NOT_MAPPED_AS_ARRAY", 212 },

	/**
	 * This indicates that a mapped resource is not available for access as a
	 * pointer.
	 */
	{ "CUDA_ERROR_NOT_MAPPED_AS_POINTER", 213 },

	/**
	 * This indicates that an uncorrectable ECC error was detected during
	 * execution.
	 */
	{ "CUDA_ERROR_ECC_UNCORRECTABLE", 214 },

	/**
	 * This indicates that the ::CUlimit passed to the API call is not
	 * supported by the active device.
	 */
	{ "CUDA_ERROR_UNSUPPORTED_LIMIT", 215 },

	/**
	 * This indicates that the ::CUcontext passed to the API call can
	 * only be bound to a single CPU thread at a time but is already
	 * bound to a CPU thread.
	 */
	{ "CUDA_ERROR_CONTEXT_ALREADY_IN_USE", 216 },

	/**
	 * This indicates that an uncorrectable NVLink error was detected during the
	 * execution.
	 */
	{ "CUDA_ERROR_NVLINK_UNCORRECTABLE", 220 },

	/**
	 * This indicates that the device kernel source is invalid.
	 */
	{ "CUDA_ERROR_INVALID_SOURCE", 300 },

	/**
	 * This indicates that the file specified was not found.
	 */
	{ "CUDA_ERROR_FILE_NOT_FOUND", 301 },

	/**
	 * This indicates that a link to a shared object failed to resolve.
	 */
	{ "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND", 302 },

	/**
	 * This indicates that initialization of a shared object failed.
	 */
	{ "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED", 303 },

	/**
	 * This indicates that an OS call failed.
	 */
	{ "CUDA_ERROR_OPERATING_SYSTEM", 304 },


	/**
	 * This indicates that a resource handle passed to the API call was not
	 * valid. Resource handles are opaque types like ::CUstream and ::CUevent.
	 */
	{ "CUDA_ERROR_INVALID_HANDLE", 400 },


	/**
	 * This indicates that a named symbol was not found. Examples of symbols
	 * are global/constant variable names, texture names }, and surface names.
     */
    { "CUDA_ERROR_NOT_FOUND", 500 },


    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::CUDA_SUCCESS (which indicates completion). Calls that
     * may return this value include ::cuEventQuery() and ::cuStreamQuery().
     */
    { "CUDA_ERROR_NOT_READY", 600 },


    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. The context cannot be used }, so it must
     * be destroyed (and a new one should be created). All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    { "CUDA_ERROR_ILLEGAL_ADDRESS", 700 },

    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. This error usually indicates that the user has
     * attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register
     * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
     * when a 32-bit int is expected) is equivalent to passing too many
     * arguments and can also result in this error.
     */
    { "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES", 701 },

    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device attribute
     * ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. The
     * context cannot be used (and must be destroyed similar to
     * ::CUDA_ERROR_LAUNCH_FAILED). All existing device memory allocations from
     * this context are invalid and must be reconstructed if the program is to
     * continue using CUDA.
     */
    { "CUDA_ERROR_LAUNCH_TIMEOUT", 702 },

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    { "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING", 703 },

    /**
     * This error indicates that a call to ::cuCtxEnablePeerAccess() is
     * trying to re-enable peer access to a context which has already
     * had peer access to it enabled.
     */
    { "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED", 704 },

    /**
     * This error indicates that ::cuCtxDisablePeerAccess() is
     * trying to disable peer access which has not been enabled yet
     * via ::cuCtxEnablePeerAccess().
     */
    { "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED", 705 },

    /**
     * This error indicates that the primary context for the specified device
     * has already been initialized.
     */
    { "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE", 708 },

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::cuCtxDestroy }, or is a primary context which
     * has not yet been initialized.
     */
    { "CUDA_ERROR_CONTEXT_IS_DESTROYED", 709 },

    /**
     * A device-side assert triggered during kernel execution. The context
     * cannot be used anymore, and must be destroyed. All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    { "CUDA_ERROR_ASSERT", 710 },

    /**
     * This indicates that an unknown internal error has occurred.
     */
    { "CUDA_ERROR_UNKNOWN", 999 },
    { NULL, -1 }
};

// This is just a linear search through the array, since the error_id's are not
// always ocurring consecutively
const char *getCudaDrvErrorString(CUresult error_id)
{
	int index = 0;

	while ((sCudaDrvErrorString[index].error_id != (int)error_id) &&
			(sCudaDrvErrorString[index].error_id != -1))
	{
		index++;
	}

	if (sCudaDrvErrorString[index].error_id == (int)error_id)
		return (const char *)sCudaDrvErrorString[index].error_string;
	else
		return (const char *)"CUDA_ERROR not found!";
}

char *MQX_PRINT_MSG[PRINT_LEVELS] = {
    "S", "F", "E", "W", "I", "D",
};

void show_stackframe() {
  char **messages = (char **)NULL;
  int i, trace_size = 0;
  void *trace[32];

  trace_size = backtrace(trace, 32);
  messages = backtrace_symbols(trace, trace_size);
  mqx_print(INFO, "Printing stack frames:");
  for (i = 0; i < trace_size; ++i)
    mqx_print(INFO, "\t%s", messages[i]);
}

void panic(const char *msg) {
  mqx_print(FATAL, "%s", msg);
  show_stackframe();
  exit(-1);
}

#ifdef MQX_PRINT_BUFFER
char *mqx_print_buffer = NULL;
#define PRINT_BUFFER_SIZE (128L * 1024L * 1024L)
int mqx_print_lines = 0;
int mqx_print_head = 0;
struct spinlock mqx_print_lock;
#endif

void mqx_print_init() {
#ifdef MQX_PRINT_BUFFER
  int i;
  initlock(&mqx_print_lock);
  mqx_print_buffer = (char *)__libc_malloc(PRINT_BUFFER_SIZE);
  if (!mqx_print_buffer) {
    mqx_print(FATAL, "Failed to initialize mqx_print buffer");
    exit(-1);
  }
  for (i = 0; i < PRINT_BUFFER_SIZE; i += 4096)
    mqx_print_buffer[i] = 'x';
#endif
}

void mqx_print_fini() {
#ifdef MQX_PRINT_BUFFER
  int i, head = 0, len;
  if (mqx_print_buffer) {
    for (i = 0; i < mqx_print_lines; i++) {
      len = printf("%s", mqx_print_buffer + head);
      head += len + 1;
    }
    __libc_free(mqx_print_buffer);
    mqx_print_buffer = NULL;
  }
#endif
}

