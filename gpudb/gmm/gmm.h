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
#ifndef _MQX_H_
#define _MQX_H_

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "advice.h"

#define FLAG_PTARRAY 1 // Device memory pointer array

#ifdef __cplusplus
extern "C" {
#endif

// MQX extensions to CUDA runtime interfaces.
cudaError_t cudaMallocEx(void **devPtr, size_t size, int flags);
cudaError_t cudaAdvise(int which_arg, int advice);
cudaError_t cudaSetFunction(int func_index);

#ifdef __cplusplus
}
#endif

#endif // _MQX_H_
