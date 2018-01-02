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
#ifndef _MQX_INTERFACES_H_
#define _MQX_INTERFACES_H_

#define __USE_GNU       // _GNU_SOURCE
#include <dlfcn.h>      // For dlopen() and dlsym()
#include <stdio.h>
#include <stdlib.h>

#ifdef CUDAPATH
#define CUDA_CURT_PATH  (CUDAPATH "/lib64/libcudart.so")
#else
#define CUDA_CURT_PATH  "/usr/local/cuda/lib64/libcudart.so"
#endif

// Checks whether dlsym was successful
#define  CHECK_DLERROR()                          \
    do {                                          \
        char * __error;                           \
        if ((__error = dlerror()) != NULL) {      \
            fputs(__error, stderr);               \
            abort();                              \
        }                                         \
    } while(0)

// Gets the pointer to the default implementation of
// an API function, func, and stores it in var.
#define DEFAULT_API_POINTER(func, var)              \
    do {                                            \
        var = (typeof(var))dlsym(RTLD_NEXT, func);  \
        CHECK_DLERROR();                            \
    } while(0)

#endif  // _MQX_INTERFACES_H_
