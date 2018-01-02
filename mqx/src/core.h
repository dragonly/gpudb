#ifndef _MQX_CORE_H_
#define _MQX_CORE_H_

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdint.h>
#include <sys/time.h>

#include "atomic.h"
#include "common.h"
#include "list.h"
#include "mqx.h" // For FLAG_PTARRAY
#include "spinlock.h"
#include "stats.h"

// State of a device memory region.
typedef enum region_state {
  STATE_DETACHED = 0, // not allocated with device memory
  STATE_ATTACHED,     // allocated with device memory
  STATE_FREEING,      // being freed
  STATE_EVICTING,     // being evicted
  STATE_ZOMBIE        // waiting to be GC'ed
} region_state_t;

// Data access advice.
// TODO: Data access advice could be very delicate. For example,
// if the application knows that a kernel only modifies part of
// a region, it can tell MQX that only that part is modified.
struct accadv {
  int advice;
};

// Device memory block.
struct block {
  int dev_valid;        // if data copy on device is valid
  int swp_valid;        // if data copy in host swap buffer is valid
  struct spinlock lock; // r/w lock
};

// Device memory region.
// A device memory region is a virtual memory area allocated by the user
// program through cudaMalloc. It is logically partitioned into an array
// of fixed-length device memory blocks. Due to lack of system support,
// all blocks must be attached/detached together. But the valid and dirty
// status of each block are maintained separately.
struct region {
  long size;            // Size of the region in bytes
  void *dev_addr;       // Device memory address
  void *swp_addr;       // Host swap buffer address
  void *pta_addr;       // Dptr array address
  void *cow_src;        // Copy-on-write source buffer address
  int memset_value;     // cudaMemset value
  int nr_blocks;        // Number of blocks (pages) in this region
  struct block *blocks; // Device memory blocks (pages)
  struct spinlock lock; // Lock controlling access to region state
  region_state_t state; // State of the region
  atomic_t c_pinned;    // Counter of kernels pinning the region
  atomic_t c_output;    // Counter of kernels using the region as OUTPUT
  atomic_t c_input;     // Counter of kernels using the region as INPUT

  int index;            // Index in the global list of attached regions
  int freq;             // Frequency of kernel accesses
  struct accadv accadv; // Data access advice
  int flags;

  struct list_head entry_alloc;  // Link to the local_alloc_list
  struct list_head entry_attach; // Link to the local_attach_list
};

// Maximum number of kernel arguments.
#define MAX_ARGS 32

// Device memory region flags.
// define FLAG_PTARRAY     1   // In mqx.h
#define FLAG_COW 2    // Copy-on-write
#define FLAG_MEMSET 4 // Lazy cudaMemset

// A kernel argument that is a device memory pointer
struct dptr_arg {
  struct region *r;  // The region this argument points to
  unsigned long off; // Device pointer offset in the region
  int advice;        // Access advices
  void *dptr;        // The actual device memory address
};

// A kernel argument
struct karg {
  char is_dptr;
  union {
    struct dptr_arg dptr;
    void *ndptr;
  };
  size_t size;
  size_t argoff;
};

// Kernel callback structure
struct kcb {
  struct region *rgns[MAX_ARGS]; // Regions referenced by the kernel
  int flags[MAX_ARGS];           // INPUT/OUTPUT flag of each region
  int nrgns;                     // Number of regions referenced
};

// Staging buffer amount and size.
// It seems nbufs=2, bufsize=512k, blocksize=8m performs very good.
#define DMA_NBUFS 2
#define DMA_BUFSIZE (512 * 1024)

// A DMA channel for either HtoD or DtoH data transfer
struct dma_channel {
  struct spinlock lock;
  cudaStream_t stream;           // The stream where DMA commands are issued
  int ibuf;                      // The next staging buffer available for use
  void *stage_bufs[DMA_NBUFS];   // Host-pinned staging buffers
  cudaEvent_t events[DMA_NBUFS]; // Events for syncing staging buffers
};

// The local MQX context
struct local_context {
  struct spinlock lock;         // Currently not useful
  latomic_t size_attached;      // Total size of attached mem regions
  struct list_head list_alloc;  // List of all allocated mem regions
  struct list_head list_attach; // LRU list of attached mem regions
  struct spinlock lock_alloc;
  struct spinlock lock_attach;
  struct dma_channel dma_htod; // HtoD DMA channel
  struct dma_channel dma_dtoh; // DtoH DMA channel
  cudaStream_t stream_kernel;  // The CUDA stream for kernel launches
  struct statistics stats;
};

// A victim region to be evicted
struct victim {
  struct region *r; // for a local victim
  int client;       // for a remote victim
  void *addr;       // virtual address of the remote victim
  struct list_head entry;
};

#define BLOCKSIZE (8L * 1024L * 1024L)
#define BLOCKSHIFT 23
#define BLOCKMASK (~(BLOCKSIZE - 1))

#define NRBLOCKS(size) (((size) + BLOCKSIZE - 1) >> BLOCKSHIFT)
#define BLOCKIDX(offset) (((unsigned long)(offset)) >> BLOCKSHIFT)
#define BLOCKUP(offset) (((offset) + BLOCKSIZE) & BLOCKMASK)

#define PAGESIZE (4 * 1024)
#define PAGESHIFT 12
#define PAGEMASK (PAGESIZE - 1)
#define PAGE_ALIGN_UP(addr) (((unsigned long)(addr) + PAGESIZE - 1) & (~PAGEMASK))
#define PAGE_ALIGN_DOWN(addr) ((unsigned long)(addr) & (~PAGEMASK))

#define region_pinned(r) atomic_read(&(r)->c_pinned)

// Make all host swap buffer or device memory blocks in a region invalid.
static inline void mk_region_inval(struct region *r, int swp) {
  int i;
  if (swp) {
    for (i = 0; i < r->nr_blocks; i++)
      r->blocks[i].swp_valid = 0;
  } else {
    for (i = 0; i < r->nr_blocks; i++)
      r->blocks[i].dev_valid = 0;
  }
}

// Make all host swap buffer or device memory blocks in a region valid.
static inline void mk_region_valid(struct region *r, int swp) {
  int i;
  if (swp) {
    for (i = 0; i < r->nr_blocks; i++)
      r->blocks[i].swp_valid = 1;
  } else {
    for (i = 0; i < r->nr_blocks; i++)
      r->blocks[i].dev_valid = 1;
  }
}

// Whether pointer p is included in pointer array a[0:n)
static inline int is_included(const void **a, int n, const void *p) {
  int i;
  for (i = 0; i < n; i++) {
    if (a[i] == p)
      return 1;
  }
  return 0;
}

// Calibrate requested malloc size to the actual size of allocated
// device memory space on the GPU. This calibration is based on NVIDIA
// GTX 580 GPU with commercial CUDA 5.0 driver.
// TODO: It is possible to automate the calibration for different GPUs
// and drivers.
static inline long size_cal(long size) {
  long size_calibrated = 0;
  if (size < 0)
    panic("illegal size calibrated");
  if (size == 0)
    size_calibrated = 0;
  else if (size > 0 && size <= 1024L * 1024L)
    size_calibrated = 1024L * 1024L;
  else // Align up to 128KiB boundary
    size_calibrated = (size + ((1 << 17) - 1)) & ((1 << 17) - 1);
  return size_calibrated;
}

int context_init();
void context_fini();
cudaError_t mqx_cudaMalloc(void **devPtr, size_t size, int flags);
cudaError_t mqx_cudaFree(void *devPtr);
cudaError_t mqx_cudaSetupArgument(const void *arg, size_t size, size_t offset);
cudaError_t mqx_cudaMemcpyHtoD(void *dst, const void *src, size_t count);
cudaError_t mqx_cudaMemcpyDtoH(void *dst, const void *src, size_t count);
cudaError_t mqx_cudaMemcpyDtoD(void *dst, const void *src, size_t count);
cudaError_t mqx_cudaMemcpyDefault(void *dst, const void *src, size_t count);
cudaError_t mqx_cudaMemGetInfo(size_t *free, size_t *total);
cudaError_t mqx_cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream);
cudaError_t mqx_cudaMemset(void *devPtr, int value, size_t count);
cudaError_t mqx_cudaLaunch(const char *entry);
void mqx_handle_cow(void *cow_begin, void *cow_end, struct region *r);

#endif
