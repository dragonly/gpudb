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

// foward declarations
static struct mps_region *find_allocated_region(struct global_context*, const void*);
static void add_allocated_region(struct mps_region*);
static void remove_allocated_region(struct mps_region *rgn);
static void add_attached_region(struct mps_region*);
static void remove_attached_region(struct mps_region *rgn);
static int free_region(struct mps_region*);
static cudaError_t mpsserver_cudaMemcpyHostToSwap(struct mps_client *client, void *dst, void *src, size_t size);
static cudaError_t mpsserver_cudaMemcpyDeviceToHost(struct mps_client *client, void *dst, void *src, size_t size);
static cudaError_t mpsserver_cudaMemcpyDeviceToDevice(struct mps_client *client, void *dst, void *src, size_t size);
static cudaError_t mpsserver_cudaMemcpyDefault(struct mps_client *client, void *dst, void *src, size_t size);
static cudaError_t mpsserver_HtoS(struct mps_client *client, struct mps_region *rgn, void *dst, void *src, size_t size);
static cudaError_t mpsserver_doHtoS(struct mps_client *client, struct mps_region *rgn, void *dst, void *src, size_t size);
static cudaError_t mpsserver_doHtoS_block(struct mps_client *client, struct mps_region*, uint64_t, void*, uint32_t, uint32_t, uint8_t, uint8_t*);
static cudaError_t mpsserver_DtoH(struct mps_client *client, struct mps_region *rgn, void *dst, void *src, size_t size);
static cudaError_t mpsserver_doDtoH(struct mps_client *client, struct mps_region *rgn, void *dst, void *src, size_t size);
static cudaError_t mpsserver_doDtoH_block(struct mps_client *client, struct mps_region*, uint64_t, void*, uint32_t, uint32_t, uint8_t, uint8_t*);
static cudaError_t mpsserver_sync_block(struct mps_client *client, struct mps_region *rgn, uint32_t iblock);
static cudaError_t mpsserver_sync_blockStoD(struct mps_client *client, void *dst, void *src, size_t size);
static cudaError_t mpsserver_sync_blockDtoS(struct mps_client *client, void *dst, void *src, size_t size);
static void update_region_evict_cost(struct mps_region *rgn);

extern struct global_context *pglobal;

inline int recv_large_buf(int socket, unsigned char *buf, uint32_t size) {
  unsigned char *pbuf = buf;
  uint32_t recved, rest_size;
  rest_size = size;
  do {
    recved = recv(socket, pbuf, rest_size, 0);
    if (recved < 0) {
      mqx_print(ERROR, "receiving from client socket: %s", strerror(errno));
      return -1;
    } else if (recved > 0) {
      pbuf += recved;
      rest_size -= recved;
    };
  } while (rest_size > 0);
  return 0;
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
  rgn->evict_cost = 0;
  add_allocated_region(rgn);
  *devPtr = rgn->swap_addr;
  return cudaSuccess;
}
cudaError_t mpsserver_cudaFree(void *devPtr) {
  struct mps_region *rgn;
  if (!(rgn = find_allocated_region(pglobal, devPtr))) {
    mqx_print(ERROR, "invalid device pointer %p", devPtr);
    return cudaErrorInvalidDevicePointer;
  }
  if (free_region(rgn) != 0) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}
cudaError_t mpsserver_cudaMemcpy(struct mps_client *client, void *dst, void *src, size_t size, enum cudaMemcpyKind kind) {
  switch (kind) {
    case cudaMemcpyHostToDevice:
      return mpsserver_cudaMemcpyHostToSwap(client, dst, src, size);
    case cudaMemcpyDeviceToHost:
      return mpsserver_cudaMemcpyDeviceToHost(client, dst, src, size);
    case cudaMemcpyDeviceToDevice:
      return mpsserver_cudaMemcpyDeviceToDevice(client, dst, src, size);
    case cudaMemcpyDefault:
      return mpsserver_cudaMemcpyDefault(client, dst, src, size);
    case cudaMemcpyHostToHost:
      return cudaErrorNotYetImplemented;
  }
  return cudaErrorInvalidMemcpyDirection;
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
static struct mps_region *find_allocated_region(struct global_context *pglobal, const void *ptr) {
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
static cudaError_t mpsserver_cudaMemcpyHostToSwap(struct mps_client *client, void *dst, void *src, size_t size) {
  struct mps_region *rgn;
  rgn = find_allocated_region(pglobal, dst);
  if (!rgn) {
    mqx_print(ERROR, "Host->Device: dst(%p) region found", dst);
    return cudaErrorInvalidDevicePointer;
  }
  ASSERT(rgn->state != ZOMBIE, "Host->Device: dst(%p) region already freed", dst);
  if (size == 0) {
    mqx_print(WARN, "Host->Device: copy ZERO bytes");
    return cudaSuccess;
  }
  if (dst + size > rgn->swap_addr + rgn->size) {
    mqx_print(ERROR, "Host->Device: copied range is out of dst(%p) swap region boundary, swap(%p), size(%ld)", dst, rgn->swap_addr, rgn->size);
    return cudaErrorInvalidValue;
  }
  return mpsserver_HtoS(client, rgn, dst, src, size);
}
static cudaError_t mpsserver_HtoS(struct mps_client *client, struct mps_region *rgn, void *dst, void *src, size_t size) {
  mqx_print(DEBUG, " HtoD: region{swap(%p) size(%ld) flags(%d), state(%d)}, dst(%p) src(%p) size(%ld)", rgn->swap_addr, rgn->size, rgn->flags, rgn->state, dst, src, size);
  // TODO: ptarray, memset
  cudaError_t ret = mpsserver_doHtoS(client, rgn, dst, src, size);
  if (ret == cudaSuccess) {
    update_region_evict_cost(rgn);
  }
  return ret;
}
static cudaError_t mpsserver_DtoH(struct mps_client *client, struct mps_region *rgn, void *dst, void *src, size_t size) {
  mqx_print(DEBUG, " DtoH: region{swap(%p) size(%ld) flags(%d), state(%d)}, dst(%p) src(%p) size(%ld)", rgn->swap_addr, rgn->size, rgn->flags, rgn->state, dst, src, size);
  // TODO: ptarray, memset
  cudaError_t ret = mpsserver_doDtoH(client, rgn, dst, src, size);
  if (ret == cudaSuccess) {
    update_region_evict_cost(rgn);
  }
  return ret;
}
static cudaError_t mpsserver_doHtoS(struct mps_client *client, struct mps_region *rgn, void *dst, void *src, size_t size) {
  uint64_t offset = (uint64_t)(dst - rgn->swap_addr);
  uint64_t psrc = (uint64_t)src;
  const uint64_t end_addr = (uint64_t)(offset + size);
  const uint32_t istart = BLOCKIDX(offset);
  const uint32_t iend = BLOCKIDX(end_addr - 1);
  // TODO: a cleaner solution, e.g. a while loop?
  uint8_t *skipped = (uint8_t *)calloc(iend - istart + 1, sizeof(uint8_t));
  uint32_t transfer_size;
  cudaError_t ret;
  for (int i = istart; i <= iend; i++) {
    transfer_size = min(BLOCKUP(offset), end_addr) - offset;
    ret = mpsserver_doHtoS_block(client, rgn, offset, (void *)psrc, transfer_size, i, 1, skipped + i - istart);
    if (ret != 0) {
      goto finish;
    }
    offset += transfer_size;
    psrc += transfer_size;
  }
  psrc = (uint64_t)src;
  offset = (uint64_t)(dst - rgn->swap_addr);
  for (int i = istart; i <= iend; i++) {
    transfer_size = min(BLOCKUP(offset), end_addr) - offset;
    if (skipped[i - istart]) {
      ret = mpsserver_doHtoS_block(client, rgn, offset, (void *)psrc, transfer_size, i, 0, NULL);
      if (ret != 0) {
        goto finish;
      }
    }
    offset += transfer_size;
    psrc += transfer_size;
  }
finish:
  free(skipped);
  return ret;
}
static cudaError_t mpsserver_doDtoH(struct mps_client *client, struct mps_region *rgn, void *dst, void *src, size_t size) {
  uint64_t offset = (uint64_t)(src - rgn->swap_addr);
  uint64_t pdst = (uint64_t)dst;
  const uint64_t end_addr = (uint64_t)(offset + size);
  const uint32_t istart = BLOCKIDX(offset);
  const uint32_t iend = BLOCKIDX(end_addr - 1);
  // TODO: a cleaner solution, e.g. a while loop?
  uint8_t *skipped = (uint8_t *)calloc(iend - istart + 1, sizeof(uint8_t));
  uint32_t transfer_size;
  cudaError_t ret;
  for (int i = istart; i <= iend; i++) {
    transfer_size = min(BLOCKUP(offset), end_addr) - offset;
    ret = mpsserver_doDtoH_block(client, rgn, offset, (void *)pdst, transfer_size, i, 1, skipped + i - istart);
    if (ret != 0) {
      goto finish;
    }
    offset += transfer_size;
    pdst += transfer_size;
  }
  pdst = (uint64_t)dst;
  offset = (uint64_t)(src - rgn->swap_addr);
  for (int i = istart; i <= iend; i++) {
    transfer_size = min(BLOCKUP(offset), end_addr) - offset;
    if (skipped[i - istart]) {
      ret = mpsserver_doHtoS_block(client, rgn, offset, (void *)pdst, transfer_size, i, 0, NULL);
      if (ret != 0) {
        goto finish;
      }
    }
    offset += transfer_size;
    pdst += transfer_size;
  }
finish:
  free(skipped);
  return ret;
}
static cudaError_t mpsserver_doHtoS_block(struct mps_client *client, struct mps_region *rgn, uint64_t offset, void *src, uint32_t transfer_size, uint32_t iblock, uint8_t skip_on_wait, uint8_t *skipped) {
  struct mps_block *blk = &rgn->blocks[iblock];
  ASSERT(BLOCKIDX(offset) == iblock, "  doHtoS_block: offset and block index do not match");

  if (skip_on_wait) {
    if (pthread_mutex_trylock(&rgn->mm_mutex)) {
      *skipped = 1;
      return cudaSuccess;
    }
  } else {
    pthread_mutex_lock(&rgn->mm_mutex);
  }

  cudaError_t ret;
  // gpu valid && swap invalid && the block transfer to swap is partial block
  // then the gpu block needs to be synced to swap block first, and then write
  // the other hopefully non-overlapping part to swap block
  if (!blk->swap_valid && blk->gpu_valid) {
    uint8_t partial_modification = (offset & (~BLOCKMASK)) ||
        ((transfer_size < BLOCKSIZE) && ((offset + transfer_size) < rgn->size));
    if (partial_modification) {
      ret = mpsserver_sync_block(client, rgn, iblock);
      if (ret != cudaSuccess) {
        if (skip_on_wait) {
          *skipped = 0;
        }
        return ret;
      }
    }
  }
  memcpy(rgn->swap_addr + offset, src, transfer_size);
  blk->swap_valid = 1;
  blk->gpu_valid = 0;
  pthread_mutex_unlock(&rgn->mm_mutex);
  return cudaSuccess;
}
static cudaError_t mpsserver_doDtoH_block(struct mps_client *client, struct mps_region *rgn, uint64_t offset, void *dst, uint32_t transfer_size, uint32_t iblock, uint8_t skip_on_wait, uint8_t *skipped) {
  struct mps_block *blk = &rgn->blocks[iblock];
  ASSERT(BLOCKIDX(offset) == iblock, "  doDtoH_block: offset and block index do not match");

  if (blk->swap_valid) {
    memcpy(dst, rgn->swap_addr + offset, transfer_size);
    if (skip_on_wait) {
      *skipped = 0;
    }
    return cudaSuccess;
  }

  if (skip_on_wait) {
    if (pthread_mutex_trylock(&rgn->mm_mutex)) {
      *skipped = 1;
      return cudaSuccess;
    }
  } else {
    pthread_mutex_lock(&rgn->mm_mutex);
  }

  // block swap invalid
  cudaError_t ret;
  if (!blk->gpu_valid) {
    mqx_print(FATAL, "   doDtoH_block: both swap & gpu memory are invalid");
    ret = cudaErrorUnknown;
    goto end;
  }
  // now swap invalid && gpu valid
  ret = mpsserver_sync_block(client, rgn, iblock);
  if (ret != cudaSuccess) {
    goto end;
  }
  memcpy(dst, rgn->swap_addr + offset, transfer_size);
end:
  pthread_mutex_unlock(&rgn->mm_mutex);
  return ret;
}
static cudaError_t mpsserver_sync_block(struct mps_client *client, struct mps_region *rgn, uint32_t iblock) {
  uint8_t swap_valid = rgn->blocks[iblock].swap_valid;
  uint8_t gpu_valid = rgn->blocks[iblock].gpu_valid;

  if ((gpu_valid && swap_valid) || (!gpu_valid && !swap_valid)) {
    return cudaSuccess;
  }
  ASSERT(rgn->swap_addr != NULL && rgn->gpu_addr, "swap(%p) or gpu(%p) is NULL", rgn->swap_addr, rgn->gpu_addr);
  mqx_print(DEBUG, "sync block: region(%p) block(%d) swap_valid(%d) gpu_valid(%d)", rgn, iblock, swap_valid, gpu_valid);

  while (rgn->n_output > 0) {
    sched_yield();
  }
  uint32_t offset = iblock * BLOCKSIZE;
  uint32_t size = min(offset + BLOCKSIZE, rgn->size) - offset;
  cudaError_t ret;
  if (gpu_valid && !swap_valid) {
    ret = mpsserver_sync_blockDtoS(client, rgn->swap_addr + offset, rgn->gpu_addr + offset, size);
    if (ret == cudaSuccess) {
      rgn->blocks[iblock].swap_valid = 1;
    }
  } else {
    while (rgn->n_input > 0) {
      sched_yield();
    }
    ret = mpsserver_sync_blockStoD(client, rgn->swap_addr + offset, rgn->gpu_addr + offset, size);
    if (ret == cudaSuccess) {
      rgn->blocks[iblock].gpu_valid = 1;
    }
  }
  return ret;
}
static cudaError_t mpsserver_sync_blockDtoS(struct mps_client *client, void *dst, void *src, size_t size) {
  struct mps_dma_channel *channel = &client->dma_htod;
  cudaError_t ret;
  uint64_t offset_dtos, offset_stoh, round_size, ibuflast;
  pthread_mutex_lock(&client->dma_mutex);
  offset_dtos = 0;
  ibuflast = channel->ibuf;
  // first initiate DMA to fully occupy all dma stage buffer
  while ((offset_dtos < size) && (offset_dtos < DMA_NBUF * DMA_BUF_SIZE)) {
    round_size = min(offset_dtos + DMA_BUF_SIZE, size) - offset_dtos;
    if ((ret = cudaMemcpyAsync(channel->stage_buf[channel->ibuf], src + offset_dtos, round_size, cudaMemcpyDeviceToHost, client->stream) != cudaSuccess)) {
      mqx_print(FATAL, "cudaMemcpyAsync failed in blockDtoH");
      goto end;
    }
    if (cudaEventRecord(channel->events[channel->ibuf], client->stream) != cudaSuccess) {
      mqx_print(FATAL, "cudaEventRecord failed in blockDtoH");
      goto end;
    }
    channel->ibuf = (channel->ibuf + 1) % DMA_NBUF;
    offset_dtos += round_size;
  }

  channel->ibuf = ibuflast;
  offset_stoh = 0;
  while (offset_stoh < size) {
    if ((ret = cudaEventSynchronize(channel->events[channel->ibuf])) != cudaSuccess) {
      mqx_print(FATAL, "cudaEventSynchronize failed in blockDtoH");
      goto end;
    }
    round_size = min(offset_stoh + DMA_BUF_SIZE, size) - offset_stoh;
    memcpy(dst + offset_stoh, channel->stage_buf[channel->ibuf], round_size);
    offset_stoh += round_size;

    if (offset_dtos < size) {
      round_size = min(offset_dtos + DMA_BUF_SIZE, size) - offset_dtos;
      if ((ret = cudaMemcpyAsync(channel->stage_buf[channel->ibuf], src + offset_dtos, round_size, cudaMemcpyDeviceToHost, client->stream) != cudaSuccess)) {
        mqx_print(FATAL, "cudaMemcpyAsync failed in blockDtoH");
        goto end;
      }
      if (cudaEventRecord(channel->events[channel->ibuf], client->stream) != cudaSuccess) {
        mqx_print(FATAL, "cudaEventRecord failed in blockDtoH");
        goto end;
      }
      offset_dtos += round_size;
    }
    channel->ibuf = (channel->ibuf + 1) % DMA_NBUF;
  }
end:
  pthread_mutex_unlock(&client->dma_mutex);
  return ret;
}
static cudaError_t mpsserver_sync_blockStoD(struct mps_client *client, void *dst, void *src, size_t size) {
  struct mps_dma_channel *channel = &client->dma_htod;
  cudaError_t ret;
  uint64_t offset, round_size, ibuflast;
  pthread_mutex_lock(&client->dma_mutex);
  offset = 0;
  while (offset < size) {
    if ((ret = cudaEventSynchronize(channel->events[channel->ibuf])) != cudaSuccess) {
      mqx_print(FATAL, "cudaEventSynchronize failed in blockHtoD");
      goto end;
    }
    round_size = min(offset + DMA_BUF_SIZE, size) - offset;
    memcpy(channel->stage_buf[channel->ibuf], src + offset, round_size);
    if ((ret = cudaMemcpyAsync(dst + offset, channel->stage_buf[channel->ibuf], round_size, cudaMemcpyHostToDevice, client->stream) != cudaSuccess)) {
      mqx_print(FATAL, "cudaMemcpyAsync failed in blockHtoD");
      goto end;
    }
    if (cudaEventRecord(channel->events[channel->ibuf], client->stream) != cudaSuccess) {
      mqx_print(FATAL, "cudaEventRecord failed in blockHtoD");
      goto end;
    }
    ibuflast = channel->ibuf;
    channel->ibuf = (channel->ibuf + 1) % DMA_NBUF;
    offset += round_size;
  }
  if ((ret = cudaEventSynchronize(channel->events[ibuflast])) != cudaSuccess) {
    mqx_print(FATAL, "cudaEventSynchronize failed in blockHtoD");
    goto end;
  }
end:
  pthread_mutex_unlock(&client->dma_mutex);
  return ret;
}
static cudaError_t mpsserver_cudaMemcpyDeviceToHost(struct mps_client *client, void *dst, void *src, size_t size) {
  struct mps_region *rgn;
  rgn = find_allocated_region(pglobal, src);
  if (!rgn) {
    mqx_print(ERROR, "Device->Host: src(%p) region found", src);
    return cudaErrorInvalidDevicePointer;
  }
  ASSERT(rgn->state != ZOMBIE, "Device->Host: src(%p) region already freed", src);
  if (size == 0) {
    mqx_print(WARN, "Device->Host: copy ZERO bytes");
    return cudaSuccess;
  }
  if (src + size > rgn->swap_addr + rgn->size) {
    mqx_print(ERROR, "Device->Host: copied range is out of src(%p) swap region boundary, swap(%p), size(%ld)", dst, rgn->swap_addr, rgn->size);
    return cudaErrorInvalidValue;
  }
  return mpsserver_DtoH(client, rgn, dst, src, size);
}
static cudaError_t mpsserver_cudaMemcpyDeviceToDevice(struct mps_client *client, void *dst, void *src, size_t size) {
  return cudaErrorNotYetImplemented;
}
static cudaError_t mpsserver_cudaMemcpyDefault(struct mps_client *client, void *dst, void *src, size_t size) {
  return cudaErrorNotYetImplemented;
}
int dma_channel_init(struct mps_dma_channel *channel, int isHtoD) {
  int i;
  channel->ibuf = 0;
  for (i = 0; i < DMA_NBUF; i++) {
    if (cudaHostAlloc(&channel->stage_buf[i], DMA_BUF_SIZE, isHtoD ? cudaHostAllocWriteCombined : cudaHostAllocDefault) != cudaSuccess) {
      mqx_print(FATAL, "Failed to create the staging buffer.");
      break;
    }
    if (cudaEventCreateWithFlags(&channel->events[i], cudaEventDisableTiming|cudaEventBlockingSync) != cudaSuccess) {
      mqx_print(FATAL, "Failed to create staging buffer sync barrier.");
      cudaFreeHost(channel->stage_buf[i]);
      break;
    }
  }
  if (i < DMA_NBUF) {
    while (--i >= 0) {
      cudaEventDestroy(channel->events[i]);
      cudaFreeHost(channel->stage_buf[i]);
    }
    return -1;
  }
  return 0;
}
void dma_channel_destroy(struct mps_dma_channel *channel) {
  for (int i = 0; i < DMA_NBUF; i++) {
    cudaEventDestroy(channel->events[i]);
    cudaFreeHost(channel->stage_buf[i]);
  }
}
static void update_region_evict_cost(struct mps_region *rgn) {
  uint64_t cost = 0;
  pthread_mutex_lock(&rgn->mm_mutex);
  if (rgn->flags & FLAG_PTARRAY) {
    cost = 0;
  } else {
    for (int i = 0; i < rgn->nblocks; i++) {
      if (rgn->blocks[i].gpu_valid && !rgn->blocks[i].swap_valid) {
        cost += min((i+1) * BLOCKSIZE, rgn->size) - i * BLOCKSIZE;
      }
    }
  }
  rgn->evict_cost = cost;
  pthread_mutex_unlock(&rgn->mm_mutex);
}

