#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>
#include <errno.h>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "common.h"
#include "protocol.h"
#include "serialize.h"
#include "libmpsserver.h"

// TODO: check all kinds of pointers
// TODO: statistics

static struct mps_region *find_region_alloc(struct global_context*, const void*);
static void add_allocated_region(struct mps_region*);
static void remove_allocated_region(struct mps_region *rgn);
static void add_attached_region(struct mps_region*);
static void remove_attached_region(struct mps_region *rgn);
static int free_region(struct mps_region*);

extern struct global_context *pglobal;

cudaError_t mpsserver_cudaFree(void *devPtr) {
  struct mps_region *rgn;
  if (!(rgn = find_region_alloc(pglobal, devPtr))) {
    mqx_print(ERROR, "invalid device pointer %p", devPtr);
    return cudaErrorInvalidDevicePointer;
  }
  if (free_region(rgn) != 0) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}
cudaError_t mpsserver_cudaMalloc(void **devPtr, size_t size, uint32_t flags) {
  if (size == 0) {
    mqx_print(WARN, "allocating 0 bytes");
  } else if (size > pglobal->mem_total) {
    return cudaErrorMemoryAllocation;
  }
  mqx_print(DEBUG, "allocate %zu bytes, %zu bytes free", size, pglobal->mem_total);

  struct mps_region *rgn;
  rgn = (struct mps_region *)calloc(1, sizeof(struct mps_region));
  rgn->swap_addr = malloc(size);
  rgn->gpu_addr = NULL;
  // TODO: FLAG_PTARRAY
  rgn->nblocks = NBLOCKS(size);
  rgn->blocks = (struct mps_block *)calloc(rgn->nblocks, sizeof(struct mps_block));
  rgn->size = size;
  rgn->state = DETACHED;
  rgn->flags = flags;
  rgn->using_kernels = 0;
  add_allocated_region(rgn);
  *devPtr = rgn->swap_addr;
  return cudaSuccess;
}

/**
 * helper functions
 */
static void add_allocated_region(struct mps_region *rgn) {
  pthread_mutex_lock(&pglobal->alloc_mutex);
  list_add(&rgn->entry_alloc, &pglobal->allocated_regions);
  pthread_mutex_unlock(&pglobal->alloc_mutex);
}
static void remove_allocated_region(struct mps_region *rgn) {
  pthread_mutex_lock(&pglobal->alloc_mutex);
  list_del(&rgn->entry_alloc);
  pthread_mutex_unlock(&pglobal->alloc_mutex);
}
static void add_attached_region(struct mps_region *rgn) {
  pthread_mutex_lock(&pglobal->attach_mutex);
  list_add(&rgn->entry_attach, &pglobal->attached_regions);
  pthread_mutex_unlock(&pglobal->attach_mutex);
}
static void remove_attached_region(struct mps_region *rgn) {
  pthread_mutex_lock(&pglobal->attach_mutex);
  list_del(&rgn->entry_attach);
  pthread_mutex_unlock(&pglobal->attach_mutex);
}
static struct mps_region *find_region_alloc(struct global_context *pglobal, const void *ptr) {
  uint64_t addr = (uint64_t)ptr;
  uint64_t start_addr, end_addr;
  struct list_head *pos;
  struct mps_region *rgn;
  uint8_t found = 0;
  
  pthread_mutex_lock(&pglobal->alloc_mutex);
  list_for_each(pos, &pglobal->allocated_regions) {
    rgn = list_entry(pos, struct mps_region, entry_alloc);
    if (rgn->state == ZOMBIE) {
      continue;
    }
    start_addr = (unsigned long)(rgn->swap_addr);
    end_addr = start_addr + (unsigned long)(rgn->size);
    if (addr >= start_addr && addr < end_addr) {
      found = 1;
      break;
    }
  }
  pthread_mutex_unlock(&pglobal->alloc_mutex);
  if (found)
    return rgn;
  else
    return NULL;
}
static int free_region(struct mps_region *rgn) {
begin:
  pthread_mutex_lock(&rgn->mm_mutex);
  switch (rgn->state) {
    // enough memory on gpu, just free it
    case ATTACHED:
      if (rgn->using_kernels == 0) {
        goto revoke_resources;
      }
      rgn->evict_cost = 0;
      pthread_mutex_unlock(&rgn->mm_mutex);
      // region still being used by kernel(s), waiting for the end.
      // Because column data is not passed in by socket, but allocated by the server,
      // and shared among client processes, this should not be `cudaFree`ed, thus must
      // not come here, which is not handled for now.
      sched_yield();
      goto begin;
    case EVICTED:
      mqx_print(DEBUG, "free a being evicted region");
      goto revoke_resources;
    case DETACHED:
      mqx_print(DEBUG, "free a detached region");
      checkCudaErrors(cuMemFree((CUdeviceptr)rgn->gpu_addr));
      rgn->gpu_addr = NULL;
      goto revoke_resources;
    case ZOMBIE:
      pthread_mutex_unlock(&rgn->mm_mutex);
      mqx_print(ERROR, "freeing a zombie region");
      return -1;
    default:
      pthread_mutex_unlock(&rgn->mm_mutex);
      mqx_print(FATAL, "no such region state");
      return -1;
  }
revoke_resources:
  // TODO: gc thread
  remove_attached_region(rgn);
  remove_allocated_region(rgn);
  free(rgn->blocks);
  rgn->blocks = NULL;
  free(rgn->swap_addr);
  rgn->swap_addr = NULL;
  rgn->state = ZOMBIE;
  pthread_mutex_unlock(&rgn->mm_mutex);
  return 0;
}
