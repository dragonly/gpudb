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
#ifndef __SERIALIZE_H_
#define __SERIALIZE_H_
#include <stdint.h>
#include "mps.h"

unsigned char* serialize_uint32(unsigned char*, const uint32_t);
unsigned char* deserialize_uint32(unsigned char* const, uint32_t*);
unsigned char* serialize_uint64(unsigned char*, const uint64_t);
unsigned char* deserialize_uint64(unsigned char* const, uint64_t*);
unsigned char* serialize_str(unsigned char*, const char* const, int);
unsigned char* deserialize_str(unsigned char* const, char*, int);
unsigned char* serialize_kernel_args(unsigned char*, const struct kernel_args);
unsigned char* deserialize_kernel_args(unsigned char* const, struct kernel_args*);
int kernel_args_bytes(const struct kernel_args);
#endif

