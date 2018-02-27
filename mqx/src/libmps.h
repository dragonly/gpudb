/*
 * Copyright (c) 2017-2018 Yilong Li <liyilongko@gmail.com>
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
#ifndef _LIBMPS_H_
#define _LIBMPS_H_

#include <cuda.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include "list.h"

int send_large_buf(int socket, unsigned char *buf, uint32_t size);
int recv_large_buf(int socket, unsigned char *buf, uint32_t size);

// data types
#define BLOCKSIZE (8L * 1024L * 1024L)
#define BLOCKSHIFT 23
#define BLOCKMASK (~(BLOCKSIZE - 1))
#define NBLOCKS(size) (((size) + BLOCKSIZE - 1) >> BLOCKSHIFT)
#define BLOCKIDX(offset) (((uint64_t)(offset)) >> BLOCKSHIFT)
#define BLOCKUP(offset) (((offset) + BLOCKSIZE) & BLOCKMASK)
struct mps_block {
  uint8_t gpu_valid;
  uint8_t swap_valid;
};
typedef enum {
  DETACHED = 0, // not allocated with device memory
  ATTACHED,     // allocated with device memory
  EVICTED,      // evicted
  ZOMBIE        // waiting to be GC'ed
} mps_region_state_t;
// Device memory region flags.
#define FLAG_PTARRAY   1   // In mqx.h
#define FLAG_COW       2   // Copy-on-write
#define FLAG_MEMSET    4   // Lazy cudaMemset
#define FLAG_PERSIST   8   // will not be `cudaFree`ed during the whole lifetime of mpsserver, e.g. shared columns
#define FLAG_READONLY  16  // state will not change during the whole lifetime of mpsserver, now only used together with shared columns
struct mps_region {
  pthread_mutex_t mm_mutex;  // for shared region concurrency control, e.g. shared columns
  void *swap_addr;
  CUdeviceptr gpu_addr;
  // if this region contains a dptr array, then pta_addr contains swap addresses, which is consistent through the whole life of mpsserver
  // and the swap_addr will be filled with correct gpu address just in time before every kernel launch
  void *pta_addr; 
  mps_region_state_t state;
  struct mps_block *blocks;
  struct list_head entry_alloc;
  struct list_head entry_attach;
  size_t size;
  int32_t memset_value;
  uint32_t nblocks;
  uint32_t flags;
  uint32_t persist_swap_valid;
  uint32_t persist_gpu_valid;
  volatile uint32_t using_kernels;
  volatile uint32_t n_input;
  volatile uint32_t n_output;
  uint32_t advice;
  uint64_t evict_cost;
  uint64_t freq;
};
#define mqx_print_region(lvl, rgn, fmt, arg...)                                 \
  do {                                                              \
    if (lvl <= MQX_PRINT_LEVEL) {                                   \
      if (lvl > WARN) {                                             \
        fprintf(stdout, "%s %s: " fmt ": rgn(%p) swap(%p) gpu(%p) size(%zu) swap_valid(%d) gpu_valid(%d) (%s:%d)\n", MQX_PRINT_MSG[lvl], __func__, ##arg, (rgn), (rgn)->swap_addr, (void *)(rgn)->gpu_addr, (rgn)->size, (rgn)->blocks[0].swap_valid, (rgn)->blocks[0].gpu_valid, __FILE__, __LINE__); \
      } else {                                                      \
        fprintf(stderr, "%s %s: " fmt ": rgn(%p) swap(%p) gpu(%p) size(%zu) swap_valid(%d) gpu_valid(%d) (%s:%d)\n", MQX_PRINT_MSG[lvl], __func__, ##arg, (rgn), (rgn)->swap_addr, (void *)(rgn)->gpu_addr, (rgn)->size, (rgn)->blocks[0].swap_valid, (rgn)->blocks[0].gpu_valid, __FILE__, __LINE__); \
      }                                                             \
    }                                                               \
  } while (0)
#define RGN_PRINT_FMT(rgn) 

// column data file format header
struct columnHeader {
  long totalTupleNum; /* the total number of tuples in this column */
  long tupleNum;      /* the number of tuples in this block */
  long blockSize;     /* the size of the block in bytes */
  int blockTotal;     /* the total number of blocks that this column is divided into */
  int blockId;        /* the block id of the current block */
  int format;         /* the format of the current block */
  char padding[4060]; /* for futher use */
};
#define NCOLUMN 58
struct column_data {
  char name[32];
  struct columnHeader *headers;
  struct mps_region **prgns;
};

#endif

