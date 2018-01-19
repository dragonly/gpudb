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

uint8_t* serialize_uint16(uint8_t*, const uint16_t);
uint8_t* deserialize_uint16(uint8_t*, uint16_t*);
uint8_t* serialize_uint32(uint8_t*, const uint32_t);
uint8_t* deserialize_uint32(uint8_t* const, uint32_t*);
uint8_t* serialize_uint64(uint8_t*, const uint64_t);
uint8_t* deserialize_uint64(uint8_t* const, uint64_t*);
uint8_t* serialize_str(uint8_t*, const uint8_t* const, int);
uint8_t* deserialize_str(uint8_t* const, uint8_t*, int);
uint8_t* serialize_kernel_args(uint8_t*, const struct kernel_args);
uint8_t* deserialize_kernel_args(uint8_t* const, struct kernel_args*);
uint8_t* serialize_mps_req(uint8_t*, const struct mps_req);
uint8_t* deserialize_mps_req(uint8_t* const, struct mps_req*);
int kernel_args_bytes(const struct kernel_args);
#endif

