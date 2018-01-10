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
#include <cuda_runtime_api.h>
#include "list.h"

#define BLOCKSIZE (8L * 1024L * 1024L)
#define BLOCKSHIFT 23
#define BLOCKMASK (~(BLOCKSIZE - 1))
#define NBLOCKS(size) (((size) + BLOCKSIZE - 1) >> BLOCKSHIFT)
#define BLOCKIDX(offset) (((uint64_t)(offset)) >> BLOCKSHIFT)
#define BLOCKUP(offset) (((offset) + BLOCKSIZE) & BLOCKMASK)
struct mps_block {
  uint8_t gpu_valid;
  uint8_t swap_valid;
  pthread_mutex_t mutex_lock;
};
typedef enum {
  DETACHED = 0, // not allocated with device memory
  ATTACHED,     // allocated with device memory
  FREEING,      // being freed
  EVICTING,     // being evicted
  ZOMBIE        // waiting to be GC'ed
} mps_region_state_t;
struct mps_region {
  size_t size;
  mps_region_state_t state;
  void *gpu_addr;
  void *swap_addr;
  struct mps_block *blocks;
  pthread_mutex_t mutex_lock;
  struct list_head entry_alloc;
  int32_t memset_value;
  uint32_t nblocks;
  uint32_t flags;
};

int mps_client_init();
cudaError_t mpsclient_cudaMalloc(void **devPtr, size_t size, uint32_t flags);
cudaError_t mps_cudaMalloc(void **devPtr, size_t size, uint32_t flags);
void add_allocated_region(struct mps_region *rgn);

