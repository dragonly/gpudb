/*
 * Copyright (c) 2014 Kaibo Wang (wkbjerry@gmail.com)
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
// The protocol between MQX clients and MQX server.

#ifndef _MQX_PROTOCOL_H_
#define _MQX_PROTOCOL_H_

#include <unistd.h>
#include <stdint.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "list.h"

/**
 * MPS related
 */
#define SERVER_SOCKET_FILE "/tmp/mqx_mps_server"
#define MAX_BUFFER_SIZE (1L * 1024L * 1024L)

#define MAX_ARG_SIZE  4096
#define MAX_ARG_NUM   16
#define MAX_CLIENTS   32
#define DMA_NBUF      2
#define DMA_BUF_SIZE  (512 * 1024)

struct mps_dma_channel {
  uint8_t ibuf;
  CUstream stream;
  void *stage_buf[DMA_NBUF];
  CUevent events[DMA_NBUF];
};
struct mps_karg_dptr_arg {
  struct mps_region *rgn;  // The region this argument points to
  uint64_t offset;         // Device pointer offset in the region
  uint32_t advice;         // Access advices
};
struct mps_karg {
  char is_dptr;
  union {
    struct mps_karg_dptr_arg dptr;
    void *ndptr;
  };
  size_t size;
  size_t argoff;
};
#define MAX_ARGS 32
struct client_kernel_conf {
  dim3 gridDim;
  dim3 blockDim;
  size_t sharedMem;
  uint8_t kstack[512];  // for primitive args
  void *ktop;
  struct mps_karg kargs[MAX_ARGS];
  uint32_t nargs;
  uint32_t advice_index[MAX_ARGS];
  uint32_t advices[MAX_ARGS];
  uint32_t nadvices;
  int32_t func_index;
};
#define MAX_CLIENT_REGIONS 1024
struct mps_client {
  uint16_t id;
  CUstream stream;
  struct mps_region *rgns[MAX_CLIENT_REGIONS];
  uint32_t nrgns;
  struct mps_dma_channel dma_htod;
  struct mps_dma_channel dma_dtoh;
  struct client_kernel_conf kconf;  // per-client kernel configurations
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
  uint8_t args[MAX_ARG_SIZE];
  uint16_t blocks_per_grid;
  uint16_t threads_per_block;
  uint16_t function_index;
};

enum mps_req_t {
  REQ_HOST_MALLOC = 0,
  REQ_CUDA_MALLOC,
  REQ_CUDA_MEMCPY,
  REQ_CUDA_MEMCPY_HTOD,
  REQ_CUDA_MEMCPY_DTOH,
  REQ_CUDA_MEMCPY_DTOD,
  REQ_CUDA_MEMCPY_HTOH,
  REQ_CUDA_MEMCPY_DEFAULT,
  REQ_CUDA_MEMFREE,
  REQ_CUDA_MEMSET,
  REQ_CUDA_ADVISE,
  REQ_SET_KERNEL_FUNCTION,
  REQ_CUDA_CONFIGURE_CALL,
  REQ_CUDA_SETUP_ARGUMENT,
  REQ_CUDA_LAUNCH_KERNEL,
  REQ_CUDA_MEM_GET_INFO,
  REQ_CUDA_DEVICE_SYNCHRONIZE,
  REQ_CUDA_EVENT_CREATE,
  REQ_CUDA_EVENT_ELAPSED_TIME,
  REQ_CUDA_EVENT_RECORD,
  REQ_CUDA_EVENT_SYNCHRONIZE,
  REQ_CUDA_GET_DEVICE,
  REQ_CUDA_GET_COLUMN_BLOCK_ADDRESS,
  REQ_CUDA_GET_COLUMN_BLOCK_HEADER,
  REQ_TEST_CUDA_LAUNCH_KERNEL,
  REQ_QUIT
};
#define MPS_REQ_SIZE 16
struct mps_req {
  uint16_t type;
  uint32_t len;
  uint16_t round; // rounds to send following payload
  uint64_t last_len; // length of last round of payload
};

struct global_context {
  struct mps_client mps_clients[MAX_CLIENTS];
  uint32_t mps_clients_bitmap;
  uint16_t mps_nclients;
  uint64_t gpumem_total;
  uint64_t gpumem_used;
  pthread_mutex_t client_mutex;
  struct list_head allocated_regions;
  pthread_mutex_t alloc_mutex;
  struct list_head attached_regions;
  pthread_mutex_t attach_mutex;
  pthread_mutex_t launch_mutex;
};

#endif

