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
#include <cuda.h>

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
        fprintf(stdout, "%s %s: " fmt ": rgn(%p) swap(%p) gpu(%p) size(%zu) state(%d) flags(%d) (%s:%d)\n", MQX_PRINT_MSG[lvl], __func__, ##arg, (rgn), (rgn)->swap_addr, (void *)(rgn)->gpu_addr, (rgn)->size, (rgn)->state, (rgn)->flags, __FILE__, __LINE__); \
      } else {                                                      \
        fprintf(stderr, "%s %s: " fmt ": rgn(%p) swap(%p) gpu(%p) size(%zu) state(%d) flags(%d) (%s:%d)\n", MQX_PRINT_MSG[lvl], __func__, ##arg, (rgn), (rgn)->swap_addr, (void *)(rgn)->gpu_addr, (rgn)->size, (rgn)->state, (rgn)->flags, __FILE__, __LINE__); \
      }                                                             \
    }                                                               \
  } while (0)
#define RGN_PRINT_FMT(rgn) 

cudaError_t mpsserver_cudaMalloc(void **devPtr, size_t size, uint32_t flags);
cudaError_t mpsserver_cudaFree(void *devPtr);
cudaError_t mpsserver_cudaMemcpy(struct mps_client *client, void *dst, void *src, size_t size, enum cudaMemcpyKind kind);
cudaError_t mpsserver_cudaMemset(struct mps_client *client, void *dst, int32_t value, size_t size);
cudaError_t mpsserver_cudaAdvise(struct mps_client *client, int iarg, int advice);
cudaError_t mpsserver_cudaSetFunction(struct mps_client *client, int index);
cudaError_t mpsserver_cudaConfigureCall(struct mps_client *client, dim3 gridDim, dim3 blockDim, size_t sharedMem, CUstream stream);
cudaError_t mpsserver_cudaSetupArgument(struct mps_client *client, void *arg, size_t size, size_t offset);
cudaError_t mpsserver_cudaLaunchKernel(struct mps_client *client);
cudaError_t mpsserver_cudaMemGetInfo(size_t *free, size_t *total);
cudaError_t mpsserver_cudaDeviceSynchronize(struct mps_client *client);
cudaError_t mpsserver_cudaEventCreate(cudaEvent_t *event);
cudaError_t mpsserver_cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t mpsserver_cudaEventRecord(struct mps_client *client, cudaEvent_t event, cudaStream_t stream);
cudaError_t mpsserver_cudaEventSynchronize(cudaEvent_t event);
cudaError_t mpsserver_cudaGetDevice(int *device);
int dma_channel_init(struct mps_dma_channel *channel, int isHtoD);
void dma_channel_destroy(struct mps_dma_channel *channel);

