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
#include <stdint.h>
#include <cuda.h>

#define SERVER_SOCKET_FILE "mqx_mps_server"
#define MAX_BUFFER_SIZE 8192

#define MAX_ARG_SIZE  4096
#define MAX_ARG_NUM   16
#define MAX_CLIENTS   32

struct mps_client {
  uint16_t id;
  CUstream stream;
};
struct server_stats {
  struct mps_client clients[MAX_CLIENTS];
  // hard coded as 32 bit unsigned int for simplicity
  uint32_t clients_bitmap;
  uint16_t nclients;
};
struct kernel_args {
  void *arg_info[MAX_ARG_NUM + 1];
  uint16_t last_arg_len;
  char args[MAX_ARG_SIZE];
  uint16_t blocks_per_grid;
  uint16_t threads_per_block;
  uint16_t function_index;
};

#define REQ_HOST_MALLOC             0
#define REQ_GPU_LAUNCH_KERNEL       1
#define REQ_GPU_MALLOC              2
#define REQ_GPU_MEMCPY_HTOD_SYNC    3
#define REQ_GPU_MEMCPY_HTOD_ASYNC   4
#define REQ_GPU_MEMCPY_DTOH_SYNC    5
#define REQ_GPU_MEMCPY_DTOH_ASYNC   6
#define REQ_GPU_SYNC                7
#define REQ_GPU_MEMFREE             8
#define REQ_GPU_MEMSET              9
struct mps_req {
  uint16_t type;
  uint16_t len;
};

#define RES_OK   0
#define RES_FAIL 1
struct mps_res {
  uint16_t type;
};

#endif

