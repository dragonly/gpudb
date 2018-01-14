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

// data types
#define BLOCKSIZE (1L * 1024L * 1024L)
#define BLOCKSHIFT 20
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
  EVICTED,      // evicted
  ZOMBIE        // waiting to be GC'ed
} mps_region_state_t;
// Device memory region flags.
#define FLAG_PTARRAY  1  // In mqx.h
#define FLAG_COW      2  // Copy-on-write
#define FLAG_MEMSET   4  // Lazy cudaMemset
struct mps_region {
  pthread_mutex_t mm_mutex;
  mps_region_state_t state;
  void *gpu_addr;
  void *swap_addr;
  struct mps_block *blocks;
  struct list_head entry_alloc;
  struct list_head entry_attach;
  size_t size;
  int32_t memset_value;
  uint32_t nblocks;
  uint32_t flags;
  uint32_t using_kernels;
  uint32_t n_input;
  uint32_t n_output;
  uint64_t evict_cost;
};


cudaError_t mpsserver_cudaMalloc(void **devPtr, size_t size, uint32_t flags);
cudaError_t mpsserver_cudaFree(void *devPtr);
cudaError_t mpsserver_cudaMemcpy(struct mps_client *client, void *dst, void *src, size_t size, enum cudaMemcpyKind kind);
int recv_large_buf(int socket, unsigned char *buf, uint32_t size);
int dma_channel_init(struct mps_dma_channel *channel, int isHtoD);
void dma_channel_destroy(struct mps_dma_channel *channel);

