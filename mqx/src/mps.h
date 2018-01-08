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
#ifndef __MPS_H_
#define __MPS_H_

#define SERVER_SOCKET_FILE "mqx_mps_server"
#define MAX_BUFFER_SIZE 102400

#define REQ_HOST_MALLOC             0
#define REQ_GPU_MALLOC              1
#define REQ_GPU_MEMCPY_HTOD_SYNC    2
#define REQ_GPU_MEMCPY_HTOD_ASYNC   3
#define REQ_GPU_MEMCPY_DTOH_SYNC    4
#define REQ_GPU_MEMCPY_DTOH_ASYNC   5
#define REQ_GPU_LAUNCH_KERNEL       6
#define REQ_GPU_SYNC                8
#define REQ_GPU_MEMFREE             9
#define REQ_GPU_MEMSET              10

#define MAX_ARG_SIZE  128
#define MAX_ARG_NUM   16

struct kernel_args {
  void *arg_info[MAX_ARG_NUM + 1];
  char args[MAX_ARG_SIZE];
};
struct mps_req {
  volatile int type;
};
#endif

