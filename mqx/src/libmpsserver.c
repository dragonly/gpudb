#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>
#include <errno.h>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "advice.h"
#include "common.h"
#include "kernel_symbols.h"
#include "libmpsserver.h"
#include "mps_stats.h"
#include "protocol.h"
#include "serialize.h"


// foward declarations of internal functions
static void add_allocated_region(struct mps_region*);
static void remove_allocated_region(struct mps_region *rgn);
static void add_attached_region(struct mps_region*);
static void remove_attached_region(struct mps_region *rgn);
static uint64_t fetch_and_mark_regions(struct mps_client *client, struct mps_region ***prgns, uint32_t *pnrgns);
static cudaError_t free_region(struct mps_region*);
static cudaError_t attach_regions(struct mps_region **rgns, uint32_t nrgns);
static cudaError_t attach_region(struct mps_region *rgn);
static cudaError_t load_regions(struct mps_client *client, struct mps_region **rgns, uint32_t nrgns);
static cudaError_t load_region(struct mps_client *client, struct mps_region *rgn);
static cudaError_t load_region_memset(struct mps_client *client, struct mps_region *rgn);
static cudaError_t load_region_pta(struct mps_client *client, struct mps_region *rgn);
static cudaError_t mpsserver_cudaMemcpyHostToSwap(struct mps_client *client, void *dst, void *src, size_t size);
static cudaError_t mpsserver_cudaMemcpyDeviceToHost(struct mps_client *client, void *dst, void *src, size_t size);
static cudaError_t mpsserver_cudaMemcpyDeviceToDevice(struct mps_client *client, void *dst, void *src, size_t size);
static cudaError_t mpsserver_cudaMemcpyDefault(struct mps_client *client, void *dst, void *src, size_t size);
static cudaError_t mpsserver_doMemset(struct mps_client *client, struct mps_region *rgn, void *dst, int32_t value, size_t size);
static cudaError_t mpsserver_memset_block(struct mps_client *client, struct mps_region *rgn, uint64_t offset, int32_t value, size_t size, uint32_t iblock, uint8_t skip_on_wait, uint8_t *skipped);
static cudaError_t mpsserver_HtoS(struct mps_client *client, struct mps_region *rgn, void *dst, void *src, size_t size);
static cudaError_t mpsserver_doHtoS(struct mps_client *client, struct mps_region *rgn, void *dst, void *src, size_t size);
static cudaError_t mpsserver_doHtoS_pta(struct mps_region *rgn, void *dst, void *src, size_t size);
static cudaError_t mpsserver_doHtoS_block(struct mps_client *client, struct mps_region*, uint64_t, void*, uint32_t, uint32_t, uint8_t, uint8_t*);
static cudaError_t mpsserver_DtoH(struct mps_client *client, struct mps_region *rgn, void *dst, void *src, size_t size);
static cudaError_t mpsserver_doDtoH(struct mps_client *client, struct mps_region *rgn, void *dst, void *src, size_t size);
static cudaError_t mpsserver_doDtoS_pta(struct mps_region *rgn, void *dst, void *src, size_t size);
static cudaError_t mpsserver_doDtoH_block(struct mps_client *client, struct mps_region*, uint64_t, void*, uint32_t, uint32_t, uint8_t, uint8_t*);
static cudaError_t mpsserver_DtoD(struct mps_client *client, struct mps_region *rgn_dst, struct mps_region *rgn_src, void *dst, void *src, size_t size);
static cudaError_t mpsserver_sync_block(struct mps_client *client, struct mps_region *rgn, uint32_t iblock);
static cudaError_t mpsserver_sync_blockStoD(struct mps_client *client, CUdeviceptr dst, void *src, size_t size);
static cudaError_t mpsserver_sync_blockDtoS(struct mps_client *client, void *dst, CUdeviceptr src, size_t size);
static void update_region_evict_cost(struct mps_region *rgn);
static void kernel_finish_callback(CUstream stream, CUresult ret, void *data);
static inline void toggle_region_valid(struct mps_region *rgn, uint8_t is_swap, uint8_t value);
static inline uint8_t contains_ptr(const void **range, uint32_t len, const void *target);
static inline uint64_t calibrate_size(long size);
static inline uint64_t gpu_freemem();

// in mpsserver.c
extern struct global_context *pglobal;
extern CUfunction F_vectorAdd;
extern const char *datadir;
extern __thread struct mps_stats stats;

const char *_err_buf;
extern struct column_data shared_columns[NCOLUMN];

void mpsserver_printStats() {
#ifdef MPS_CONFIG_STATS
  mqx_print(STAT, "================= STATS =================");
  //mqx_print(STAT, "load_region time:         %.3lf ms", stats.time_load_region);
  //mqx_print(STAT, "load_region_pta time:     %.3lf ms", stats.time_load_region_pta);
  //mqx_print(STAT, "load_region_memset time:  %.3lf ms", stats.time_load_region_memset);
  //mqx_print(STAT, "load_region total time:   %.3lf ms", stats.time_load_region + stats.time_load_region_pta + stats.time_load_region_memset);
  mqx_print(STAT, "sync_block time:          %.3lf ms", stats.time_sync_block);
  mqx_print(STAT, "syncStoD time:            %.3lf ms", stats.time_syncStoD);
  mqx_print(STAT, "syncDtoS time:            %.3lf ms", stats.time_syncDtoS);
  //mqx_print(STAT, "syncDtoS1 time:           %.3lf ms", stats.time_syncDtoS1);
  mqx_print(STAT, "memcpyDtoD time:          %.3lf ms", stats.time_memcpyDtoD);
  mqx_print(STAT, "=========================================");
#endif
}

cudaError_t mpsserver_getColumnBlockAddress(void **devPtr, const char *colname, uint32_t iblock) {
  for (int i = 0; i < NCOLUMN; i++) {
    if (strcmp(colname, shared_columns[i].name) == 0) {
      *devPtr = shared_columns[i].prgns[iblock]->swap_addr;
      if (*devPtr == NULL) {
        mqx_print(FATAL, "column %s(%d) is NULL", colname, iblock);
        return cudaErrorUnknown;
      }
      mqx_print(DEBUG, "%s(%d) address(%p)", colname, iblock, *devPtr);
      return cudaSuccess;
    }
  }
  mqx_print(WARN, "column not found (%s)", colname);
  return cudaErrorInvalidValue;
}
cudaError_t mpsserver_getColumnBlockHeader(struct columnHeader **pheader, const char *colname, uint32_t iblock) {
  for (int i = 0; i < NCOLUMN; i++) {
    if (strcmp(colname, shared_columns[i].name) == 0) {
      *pheader = &shared_columns[i].headers[iblock];
      struct columnHeader *ph = *pheader;
      mqx_print(DEBUG, "%s totalTupleNum(%lu) tupleNum(%lu) blockSize(%lu) blockTotal(%d) blockId(%d) format(%d)", colname, ph->totalTupleNum, ph->tupleNum, ph->blockSize, ph->blockTotal, ph->blockId, ph->format);
      return cudaSuccess;
    }
  }
  mqx_print(WARN, "column not found (%s)", colname);
  return cudaErrorInvalidValue;
}
// NOTE: destroy all regions which are `cudaMalloc`ed by this client thread.
void client_destroy(struct mps_client *client) {
  struct mps_region *rgn;
  int destroyed = 0;
  for (int i = 0; i < client->nrgns; i++) {
    rgn = client->rgns[i];
    if (rgn->state != ZOMBIE) {
      free_region(rgn);
      destroyed++;
    }
    free(rgn);
  }
  mqx_print(INFO, "destroy %d regions", destroyed);

  struct list_head *_pos;
  struct mps_region *_rgn;
  int _nrgns = 0;
  list_for_each(_pos, &pglobal->allocated_regions) {
    //_rgn = list_entry(_pos, struct mps_region, entry_alloc);
    //printf("%p ", _rgn->swap_addr);
    _nrgns++;
  }
  mqx_print(INFO, "%d allocated regions in total", _nrgns);
  _nrgns = 0;
  int _persist = 0;
  list_for_each(_pos, &pglobal->attached_regions) {
    _rgn = list_entry(_pos, struct mps_region, entry_attach);
    //printf("%p ", _rgn->swap_addr);
    if ((_rgn->flags & FLAG_PERSIST) == 1) {
      _persist++;
    }
    _nrgns++;
  }
  mqx_print(INFO, "%d attached regions in total, %d PERSIST", _nrgns, _persist);
}
cudaError_t mpsserver_cudaMalloc(struct mps_client *client, void **devPtr, size_t size, uint32_t flags) {
  if (size == 0) {
    mqx_print(WARN, "allocating 0 bytes");
  } else if (size > pglobal->gpumem_total) {
    mqx_print(ERROR, "out of memory");
    return cudaErrorMemoryAllocation;
  }
  //mqx_print(DEBUG, "allocate %zu bytes, %zu bytes free", size, pglobal->gpumem_total);

  struct mps_region *rgn;
  rgn = (struct mps_region *)calloc(1, sizeof(struct mps_region));
  if (flags & FLAG_PTARRAY) {
    if (size % sizeof(void *)) {
      mqx_print(ERROR, "device pointer array size (%lu) not aligned", size);
      return cudaErrorInvalidValue;
    }
    rgn->pta_addr = calloc(1, size);
  }
  // TODO: use block level mutex on block operations, and maybe make mutexes as ptrs for better print in gdb
  pthread_mutex_init(&rgn->mm_mutex, NULL);
  rgn->swap_addr = calloc(1, size);
  rgn->gpu_addr = 0;
  rgn->state = DETACHED;
  rgn->nblocks = NBLOCKS(size);
  rgn->blocks = (struct mps_block *)calloc(rgn->nblocks, sizeof(struct mps_block));
  INIT_LIST_HEAD(&rgn->entry_alloc);
  INIT_LIST_HEAD(&rgn->entry_attach);
  rgn->size = size;
  rgn->memset_value = 0;
  rgn->flags = flags;
  rgn->using_kernels = 0;
  rgn->n_input = 0;
  rgn->n_output = 0;
  rgn->advice = 0;
  rgn->evict_cost = 0;
  rgn->freq = 0;
  add_allocated_region(rgn);
  if (client != NULL) {
    client->rgns[client->nrgns++] = rgn;
  } else {
    mqx_print(WARN, "client can only be NULL for preloaded columns");
  }
  *devPtr = rgn->swap_addr;
  mqx_print(DEBUG, "devPtr(%p) size(%zu)", *devPtr, size);
  return cudaSuccess;
}
cudaError_t mpsserver_cudaMemGetInfo(size_t *free, size_t *total) {
  return cuMemGetInfo(free, total);
}
cudaError_t mpsserver_cudaDeviceSynchronize(struct mps_client *client) {
  return cuStreamSynchronize(client->stream);
}
cudaError_t mpsserver_cudaEventCreate(cudaEvent_t *event) {
  return cuEventCreate(event, CU_EVENT_DEFAULT);
}
cudaError_t mpsserver_cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
  return cuEventElapsedTime(ms, start, end);
}
cudaError_t mpsserver_cudaEventRecord(struct mps_client *client, cudaEvent_t event, cudaStream_t stream) {
  // TODO: multiple stream support for each cient process
  return cuEventRecord(event, client->stream);
}
cudaError_t mpsserver_cudaEventSynchronize(cudaEvent_t event) {
  return cuEventSynchronize(event);
}
cudaError_t mpsserver_cudaGetDevice(int *device) {
  // NOTE: only handles single device
  return cuDeviceGet(device, 0);
}
cudaError_t mpsserver_cudaFree(void *devPtr) {
  struct mps_region *rgn;
  cudaError_t ret = cudaSuccess;
  if (!(rgn = find_allocated_region(pglobal, devPtr))) {
    mqx_print(ERROR, "invalid device pointer %p", devPtr);
    return cudaErrorInvalidDevicePointer;
  }
  if ((ret = free_region(rgn) != cudaSuccess)) {
    return ret;
  }
  return cudaSuccess;
}
cudaError_t mpsserver_cudaMemcpy(struct mps_client *client, void *dst, void *src, size_t size, enum cudaMemcpyKind kind) {
  cudaError_t ret = cudaSuccess;
  switch (kind) {
    case cudaMemcpyHostToDevice:
      ret = mpsserver_cudaMemcpyHostToSwap(client, dst, src, size);
      break;
    case cudaMemcpyDeviceToHost:
      ret = mpsserver_cudaMemcpyDeviceToHost(client, dst, src, size);
      break;
    case cudaMemcpyDeviceToDevice:
      ret = mpsserver_cudaMemcpyDeviceToDevice(client, dst, src, size);
      break;
    case cudaMemcpyDefault:
      ret = mpsserver_cudaMemcpyDefault(client, dst, src, size);
      break;
    case cudaMemcpyHostToHost:
      ret = cudaErrorNotYetImplemented;
      break;
    default:
      ret = cudaErrorInvalidMemcpyDirection;
  }
  return ret;
}
cudaError_t mpsserver_cudaMemset(struct mps_client *client, void *dst, int32_t value, size_t size) {
  if (size == 0) {
    mqx_print(WARN, "size == 0");
    return cudaSuccess;
  }
  struct mps_region *rgn;
  if ((rgn = find_allocated_region(pglobal, dst)) == NULL) {
    mqx_print(ERROR, "cannot find region swap(%p)", dst);
    return cudaErrorInvalidDevicePointer;
  }
  ASSERT(rgn->state != ZOMBIE, "ZOMBIE region swap(%p)", dst);
  if (dst + size > rgn->swap_addr + rgn->size) {
    mqx_print(ERROR, "out of region range");
    return cudaErrorInvalidValue;
  }
  
  if (rgn->flags & FLAG_PTARRAY) {
    uint64_t offset = dst - rgn->swap_addr;
    memset(rgn->pta_addr + offset, value, size);
    return cudaSuccess;
  }
  if (size == rgn->size) { // memset whole region, do it lazily, only set FLAG_MEMSET
    rgn->flags |= FLAG_MEMSET;
    rgn->memset_value = value;
    toggle_region_valid(rgn, 0, 0);
    toggle_region_valid(rgn, 1, 0);
    update_region_evict_cost(rgn);
    return cudaSuccess;
  }
  // partial memset, finish the former whole-region lazy memset's non-overlapping parts
  // honestly this part is hardly useful...
  mqx_print_region(DEBUG, rgn, "partial memset");
  cudaError_t ret = cudaSuccess;
  if (rgn->flags & FLAG_MEMSET) {
    uint64_t offset = dst - rgn->swap_addr;
    // left part
    if (offset > 0) {
      ret = mpsserver_doMemset(client, rgn, rgn->swap_addr, rgn->memset_value, offset);
      if (ret != cudaSuccess) {
        return ret;
      }
    }
    // right part
    offset += size;
    if (offset < rgn->size) {
      ret = mpsserver_doMemset(client, rgn, rgn->swap_addr + offset, rgn->memset_value, rgn->size - offset);
      if (ret != cudaSuccess) {
        return ret;
      }
    }
    rgn->flags &= ~FLAG_MEMSET;
    rgn->memset_value = 0;
  }
  ret = mpsserver_doMemset(client, rgn, dst, value, size);
  if (ret == cudaSuccess) {
    update_region_evict_cost(rgn);
  }
  return ret;
}
static cudaError_t mpsserver_doMemset(struct mps_client *client, struct mps_region *rgn, void *dst, int32_t value, size_t size) {
  uint64_t offset = dst - rgn->swap_addr;
  const uint64_t end_offset = offset + size;
  const uint32_t istart = BLOCKIDX(offset);
  const uint32_t iend = BLOCKIDX(end_offset - 1);
  uint8_t *skipped = calloc(1, iend - istart + 1);
  uint32_t transfer_size;
  cudaError_t ret = cudaSuccess;
  for (uint32_t i = istart; i < iend; i++) {
    transfer_size = min(BLOCKUP(offset), end_offset) - offset;
    ret = mpsserver_memset_block(client, rgn, offset, value, transfer_size, i, 1, skipped + i - istart);
    if (ret != cudaSuccess) {
      goto end;
    }
    offset += size;
  }
  offset = dst - rgn->swap_addr;
  for (uint32_t i = istart; i < iend; i++) {
    transfer_size = min(BLOCKUP(offset), end_offset) - offset;
    if (skipped[i - istart]) {
      ret = mpsserver_memset_block(client, rgn, offset, value, transfer_size, i, 1, skipped + i - istart);
      if (ret != cudaSuccess) {
        goto end;
      }
    }
    offset += size;
  }
end:
  free(skipped);
  return ret;
}
// NOTE: this is almost the same as doHtoS_block/doDtoH_block
static cudaError_t mpsserver_memset_block(struct mps_client *client, struct mps_region *rgn, uint64_t offset, int32_t value, size_t transfer_size, uint32_t iblock, uint8_t skip_on_wait, uint8_t *skipped) {
  //if (skip_on_wait) {
  //  if (pthread_mutex_trylock(&rgn->mm_mutex)) {
  //    *skipped = 1;
  //    return cudaSuccess;
  //  }
  //} else {
  //  pthread_mutex_lock(&rgn->mm_mutex);
  //}

  struct mps_block *blk = &rgn->blocks[iblock];
  ASSERT(BLOCKIDX(offset) == iblock, "  doHtoS_block: offset and block index do not match");

  cudaError_t ret = cudaSuccess;
  // gpu valid && swap invalid && the block transfer to swap is partial block
  // then the gpu block needs to be synced to swap block first, and then write
  // the other hopefully non-overlapping part to swap block
  if (!blk->swap_valid && blk->gpu_valid) {
    uint8_t partial_modification = (offset & (~BLOCKMASK)) ||
        ((transfer_size < BLOCKSIZE) && ((offset + transfer_size) < rgn->size));
    if (partial_modification) {
      if ((ret = mpsserver_sync_block(client, rgn, iblock)) != cudaSuccess) {
        if (skip_on_wait) {
          *skipped = 0;
        }
        return ret;
      }
    }
  }
  memset(rgn->swap_addr + offset, value, transfer_size);
  blk->swap_valid = 1;
  blk->gpu_valid = 0;
  //pthread_mutex_unlock(&rgn->mm_mutex);
  return cudaSuccess;
}
cudaError_t mpsserver_cudaSetFunction(struct mps_client *client, int index) {
  client->kconf.func_index = index;
  mqx_print(DEBUG, "<%s>[%d]", fname_table[index], index);
  return cudaSuccess;
}
cudaError_t mpsserver_cudaAdvise(struct mps_client *client, int iarg, int advice) {
  int i;
  struct client_kernel_conf *kconf = &client->kconf;
  mqx_print(DEBUG, "iarg(%d) advice(%d)", iarg, advice);

  if (iarg >= 0 && iarg < MAX_ARGS) {
    for (i = 0; i < kconf->nadvices; i++) {
      if (kconf->advice_index[i] == iarg)
        break;
    }
    if (i == kconf->nadvices) {
      kconf->advice_index[kconf->nadvices] = iarg;
      kconf->advices[kconf->nadvices] = advice;
      kconf->nadvices++;
    } else {
      kconf->advices[i] |= advice;
    }
  } else {
    mqx_print(ERROR, "max cudaAdvise argument exceeded(%d)", iarg);
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}
cudaError_t mpsserver_cudaConfigureCall(struct mps_client *client, dim3 gDim, dim3 bDim, size_t shMem, CUstream stream) {
  struct client_kernel_conf *kconf = &client->kconf;
  mqx_print(DEBUG, "<<<(%d %d %d), (%d %d %d), %lu, %p>>>", gDim.x, gDim.y, gDim.z, bDim.x, bDim.y, bDim.z, shMem, stream);
  kconf->nargs = 0;
  kconf->ktop = (void *)kconf->kstack;
  kconf->gridDim = gDim;
  kconf->blockDim = bDim;
  kconf->sharedMem = shMem;
  // TODO: use custom stream
  // NOTE: some kernels in are grid/block dimension dependent (and hard to refactor sadly :(
  //       hopefully it is only about 0.5% of the whole kernel execution time.
  //       we change other kernels' gridDim.x for now
  //       prescan<bool, bool>, uniformAdd, scan_int, scan_other,
  // uint32_t f = kconf->func_index;
  // if (f != 110 && f != 111 && f != 112 && f != 113 && f != 130 && f != 134 && f != 135 && f != 136 && f != 137) {
  //   if (kconf->gridDim.x > 28) {
  //     kconf->gridDim.x = 28;
  //   }
  // } else {
  //   mqx_print(DEBUG, "scanImpl, not change gridDim.x(%d)", kconf->gridDim.x);
  // }
  return cudaSuccess;
}
cudaError_t mpsserver_cudaSetupArgument(struct mps_client *client, void *parg, size_t size, size_t offset) {
  struct mps_region *rgn;
  uint8_t is_dptr = 0;
  uint32_t iadv = 0;
  struct client_kernel_conf *kconf = &client->kconf;
  mqx_print(DEBUG, "(%d) size(%lu) offset(%lu)", kconf->nargs, size, offset);

  // FIXME: this is buggy, because in this setting, any primitive arguments must NOT be `cudaAdvise`d
  if (kconf->nadvices > 0) {
    for (iadv = 0; iadv < kconf->nadvices; iadv++) {
      if (kconf->advice_index[iadv] == kconf->nargs) {
        break;
      }
    }
    if (iadv < kconf->nadvices) {
      if (size != sizeof(void *)) {
        mqx_print(ERROR, "(%d) Argument size (%lu) does not match size of dptr", kconf->nargs, size);
        return cudaErrorInvalidValue;
      }
      rgn = find_allocated_region(pglobal, *(void **)parg);
      if (!rgn) {
        mqx_print(ERROR, "(%d) Cannot find region containing %p", kconf->nargs, parg);
        pthread_exit(NULL);
        return cudaErrorInvalidValue;
      }
      is_dptr = 1;
    }
  } else if (size == sizeof(void *)) {
    mqx_print(WARN, "trying to parse dptr argument automatically, which is ERROR-PRONE!");
    rgn = find_allocated_region(pglobal, *(void **)parg);
    is_dptr = (rgn != NULL);
  } else {
    mqx_print(DEBUG, "primitive type argument");
  }

  if (is_dptr) {
    kconf->kargs[kconf->nargs].dptr.rgn = rgn;
    kconf->kargs[kconf->nargs].dptr.offset = (unsigned long)(*(void **)parg - rgn->swap_addr);
    if (kconf->nadvices > 0) {
      kconf->kargs[kconf->nargs].dptr.advice = kconf->advices[iadv];
    } else {
      kconf->kargs[kconf->nargs].dptr.advice = CADV_DEFAULT | CADV_PTADEFAULT;
    }
    mqx_print_region(DEBUG, rgn, "argument is dptr");
  } else {
    // This argument is not a device memory pointer.
    // XXX: Currently we ignore the case that CUDA runtime might
    // stop pushing arguments due to some errors.
    memcpy(kconf->ktop, parg, size);
    kconf->kargs[kconf->nargs].ndptr = kconf->ktop;
    kconf->ktop += size;
  }
  kconf->kargs[kconf->nargs].is_dptr = is_dptr;
  kconf->kargs[kconf->nargs].size = size;
  // NOTE: argoff is not used, because cudaSetupArgument somehow think a uint64 following a uint32 should have a `8` offset from the former argument, which is wierd for cuLaunchKernel
  kconf->kargs[kconf->nargs].argoff = -1; 
  kconf->nargs++;
  free(parg);
  return cudaSuccess;
}
struct kernel_callback {
  struct mps_region *rgns[MAX_ARGS];
  uint32_t advices[MAX_ARGS];
  uint32_t nrgns;
  uint32_t func_index;
};
cudaError_t mpsserver_cudaLaunchKernel(struct mps_client *client) {
  struct client_kernel_conf *kconf = &client->kconf;
  cudaError_t ret = cudaSuccess;
  struct mps_region **rgns = NULL;
  uint32_t nrgns = 0;
  uint64_t gpumem_required = 0;
  mqx_print(DEBUG, "start preparing to launch kernel <%s>[%d]", fname_table[kconf->func_index], kconf->func_index);

  // nrgns can be 0 for some kernels only using registers
  // advices are marked on regions
  gpumem_required = fetch_and_mark_regions(client, &rgns, &nrgns);
  if (gpumem_required < 0) {
    mqx_print(ERROR, "failed to get required regions");
    ret = cudaErrorUnknown;
    goto launch_fail;
  } else if (gpumem_required > pglobal->gpumem_total) {
    mqx_print(ERROR, "out of memory (%ld required/%ld total)", gpumem_required, pglobal->gpumem_total);
    ret = cudaErrorInvalidConfiguration;
    goto launch_fail;
  }

attach:
  //pthread_mutex_lock(&pglobal->attach_mutex);
  ret = attach_regions(rgns, nrgns);
  //pthread_mutex_unlock(&pglobal->attach_mutex);
  if (ret != cudaSuccess) {
    if (ret == cudaErrorLaunchTimeout) {
      //sched_yield();
      goto attach;
    } else {
      mqx_print(FATAL, "attach regions failed (%d)", ret);
      goto launch_fail;
    }
  }

  ret = load_regions(client, rgns, nrgns);
  if (ret != cudaSuccess) {
    mqx_print(FATAL, "load regions failed (%d)", ret);
    goto launch_fail;
  }

  CUfunction kernel = fsym_table[kconf->func_index];
  if (kconf->func_index == 233) {
    kernel = F_vectorAdd;
  }
  ASSERT(kernel != NULL, "null function %s (%d)", fname_table[kconf->func_index], kconf->func_index);
  ASSERT(nrgns < MAX_ARGS, "nrgns(%d) >= MAX_ARGS(%d)", nrgns, MAX_ARGS);
  struct kernel_callback *kcb;
  kcb = (struct kernel_callback *)calloc(1, sizeof(struct kernel_callback));
  kcb->func_index = kconf->func_index;
  if (nrgns > 0) {
    memcpy(kcb->rgns, rgns, sizeof(void *) * nrgns);
  }
  for (int i = 0; i < nrgns; i++) {
    kcb->advices[i] = rgns[i]->advice;
    if (kcb->advices[i] & CADV_OUTPUT) {
      __sync_fetch_and_add(&rgns[i]->n_output, 1);
    }
    if (kcb->advices[i] & CADV_INPUT) {
      __sync_fetch_and_add(&rgns[i]->n_input, 1);
    }
  }
  kcb->nrgns = nrgns;

  cudaError_t cret;
  uint8_t *kernel_args_buf = malloc(sizeof(void *) * kconf->nargs);
  void **kernel_args_ptr = malloc(sizeof(void *) * kconf->nargs);
  uint32_t offset = 0;
  for (int i = 0; i < kconf->nargs; i++) {
    if (kconf->kargs[i].is_dptr) {
      // this should copy the dptr
      memcpy(kernel_args_buf + offset, &kconf->kargs[i].dptr.rgn->gpu_addr, kconf->kargs[i].size);
    } else {
      // and this should copy the value pointed by ndptr
      memcpy(kernel_args_buf + offset, kconf->kargs[i].ndptr, kconf->kargs[i].size);
    }
    *((void **)kernel_args_ptr + i) = kernel_args_buf + offset;
    offset += kconf->kargs[i].size;
  }
  checkCudaErrors(cuStreamSynchronize(client->stream));
  if ((cret = cuLaunchKernel(kernel, kconf->gridDim.x, kconf->gridDim.y, kconf->gridDim.z, kconf->blockDim.x, kconf->blockDim.y, kconf->blockDim.z, kconf->sharedMem, client->stream, kernel_args_ptr, 0)) != cudaSuccess) {
    cuGetErrorString(cret, &_err_buf);
    mqx_print(FATAL, "cuLaunchKernel failed: %s(%d)", _err_buf, cret);
    for (int i = 0; i < nrgns; i++) {
      if (kcb->advices[i] & CADV_OUTPUT) {
        __sync_fetch_and_sub(&rgns[i]->n_output, 1);
      }
      if (kcb->advices[i] & CADV_INPUT) {
        __sync_fetch_and_sub(&rgns[i]->n_input, 1);
      }
    }
    free(kcb);
    free(kernel_args_buf);
    free(kernel_args_ptr);
    return cret;
  }
  cuStreamAddCallback(client->stream, kernel_finish_callback, (void *)kcb, 0);
  checkCudaErrors(cuStreamSynchronize(client->stream));
  // update clients position in LRU list
  //
  // NOTE: `kcb` cannot be freed here!!! it should be the work of `kernel_finish_callback`
  free(kernel_args_buf);
  free(kernel_args_ptr);
  if (kconf->func_index != 233) {
    mqx_print(DEBUG, "launch kernel <%s>[%d] succeeded", fname_table[kconf->func_index], kconf->func_index);
  } else {
    mqx_print(DEBUG, "launch kernel <vectorAdd> succeeded");
  }

launch_fail:
  if (rgns) {
    free(rgns);
  }
  kconf->func_index = -1;
  kconf->nadvices = 0;
  kconf->nargs = 0;
  kconf->ktop = (void *)kconf->kstack;
  return ret;
}
static cudaError_t load_regions(struct mps_client *client, struct mps_region **rgns, uint32_t nrgns) {
  cudaError_t ret = cudaSuccess;
  struct mps_region *rgn;
  mqx_print(DEBUG, "loading regions, %d in total", nrgns);
  for (uint32_t i = 0; i < nrgns; i++) {
    rgn = rgns[i];
    if (rgn->advice & CADV_INPUT) {
      if ((ret = load_region(client, rgn)) != cudaSuccess) {
        cuGetErrorString(ret, &_err_buf);
        mqx_print_region(ERROR, rgn, "load_region failed, %s(%d)", _err_buf, ret);
        return ret;
      }
    }
    if (rgn->advice & CADV_OUTPUT) {
      if (rgn->flags & FLAG_MEMSET) {
        if ((ret = load_region_memset(client, rgn)) != cudaSuccess) {
          cuGetErrorString(ret, &_err_buf);
          mqx_print_region(ERROR, rgn, "load_region_memset failed, %s(%d).", _err_buf, ret);
          return ret;
        }
      }
      toggle_region_valid(rgn, 1/*swap*/, 0);
      toggle_region_valid(rgn, 0/*gpu*/, 1);
    }
    update_region_evict_cost(rgn);
    rgn->freq++;
  }
  return cudaSuccess;
}
static cudaError_t load_region(struct mps_client *client, struct mps_region *rgn) {
  cudaError_t ret = cudaSuccess;
  mqx_print_region(DEBUG, rgn, "start loading region");
  double time;
  stats_time_begin();
  if (rgn->flags & FLAG_MEMSET) {
    ret = load_region_memset(client, rgn);
    stats_time_end(time);
    stats.time_load_region_memset += time;
  } else if (rgn->flags & FLAG_PTARRAY) {
    ret = load_region_pta(client, rgn);
    stats_time_end(time);
    stats.time_load_region_pta += time;
  } else {
    for (uint32_t i = 0; i < NBLOCKS(rgn->size); i++) {
      if (!rgn->blocks[i].gpu_valid) {
        if ((ret = mpsserver_sync_block(client, rgn, i)) != cudaSuccess) {
          mqx_print(ERROR, "mpsserver_sync_block failed");
          return ret;
        }
      }
    }
    stats_time_end(time);
    stats.time_load_region += time;
    mqx_print(DEBUG, "region is in sync with gpu");
  }
  return ret;
}
static cudaError_t load_region_memset(struct mps_client *client, struct mps_region *rgn) {
  cudaError_t ret = cudaSuccess;
  if ((ret = cuMemsetD8Async((CUdeviceptr)rgn->gpu_addr, rgn->memset_value, rgn->size, client->stream)) != cudaSuccess) {
    cuGetErrorString(ret, &_err_buf);
    mqx_print(ERROR, "cuMemsetD8Async failed: %s(%d)", _err_buf, ret);
    return ret;
  }
  rgn->flags &= ~FLAG_MEMSET;
  rgn->memset_value = 0;
  toggle_region_valid(rgn, 0/*gpu*/, 1);
  return cudaSuccess;
}
static cudaError_t load_region_pta(struct mps_client *client, struct mps_region *rgn) {
  mqx_print_region(DEBUG, rgn, "loading dptr array region and all sub-regions");
  void **pdptr = rgn->pta_addr; // swap address of pointers
  void **pend = rgn->pta_addr + rgn->size;
  uint64_t offset = 0;
  struct mps_region *rtmp;
  while (pdptr < pend) {
    rtmp = find_allocated_region(pglobal, *pdptr);
    if (rtmp == NULL) {
      mqx_print(ERROR, "cannot find region for dptr(%p)", *pdptr);
      return cudaErrorInvalidDevicePointer;
    }
    // populate the swap_addr of dptr array region with gpu_addr(after attached) + offset
    *(void **)(rgn->swap_addr + offset) = (void *)rtmp->gpu_addr + (*pdptr - rtmp->swap_addr);
    offset += sizeof(void *);
    pdptr++;
  }
  mqx_print(DEBUG, "dptr array region swap loaded");
  cudaError_t ret = cudaSuccess;
  // toggle validity to sync in swap->gpu direction
  toggle_region_valid(rgn, 1/*swap*/, 1);
  toggle_region_valid(rgn, 0/*gpu*/, 0);
  for (uint32_t i = 0; i < rgn->nblocks; i++) {
    if ((ret = mpsserver_sync_block(client, rgn, i)) != cudaSuccess) {
      mqx_print(ERROR, "mpsserver_sync_block failed");
      return ret;
    }
  }
  mqx_print(DEBUG, "dptr arary region is in sync with gpu");
  return cudaSuccess;
}
static void kernel_finish_callback(CUstream stream, CUresult ret, void *data) {
  struct kernel_callback *kcb = (struct kernel_callback *)data;
  if (ret != CUDA_SUCCESS) {
    cuGetErrorString(ret, &_err_buf);
    mqx_print(ERROR, "kernel <%s>[%d] execution failed: %s(%d)", fname_table[kcb->func_index], kcb->func_index, _err_buf, ret);
  } else {
    mqx_print(DEBUG, "kernel <%s>[%d] execution succeeded", fname_table[kcb->func_index], kcb->func_index);
  }
  for (int i = 0; i < kcb->nrgns; i++) {
    if (!(kcb->advices[i] & CADV_MASK)) {
      continue;
    }
    if (kcb->advices[i] & CADV_OUTPUT) {
      __sync_fetch_and_sub(&kcb->rgns[i]->n_output, 1);
    }
    if (kcb->advices[i] & CADV_INPUT) {
      __sync_fetch_and_sub(&kcb->rgns[i]->n_input, 1);
    }
    __sync_fetch_and_sub(&kcb->rgns[i]->using_kernels, 1);
  }
  // FIXME: `free` can still cause deadlock because of double lock, if this handler is intercepting while the thread is `malloc`ing or `free`ing
  free(kcb);
  mqx_print(DEBUG, "kernel callback cleaned up");
}

/**
 * helper functions
 */
static cudaError_t attach_regions(struct mps_region **rgns, uint32_t nrgns) {
  if (nrgns == 0) {
    return cudaSuccess;
  }
  if (nrgns < 0 || (nrgns > 0 && rgns == NULL)) {
    mqx_print(ERROR, "invalid value, rgns(%p) nrgns(%d)", rgns, nrgns);
  }
  mqx_print(DEBUG, "attaching regions, %d in total", nrgns);

  cudaError_t ret = cudaSuccess;
  struct mps_region *rgn;
  for (uint32_t i = 0; i < nrgns; i++) {
    rgn = rgns[i];
    if (rgn->state == ZOMBIE) {
      mqx_print(ERROR, "region(%p) is in ZOMBIE state", rgn);
      ret = cudaErrorInvalidValue;
      goto fail;
    }
    ret = attach_region(rgn);
    if (ret != cudaSuccess) {
      goto fail;
    }
  }

  mqx_print(DEBUG, "successfully attached %d regions", nrgns);
  return ret;
fail:
// TODO: detach all regions on failure
  return ret;
}
// TODO: this is copied directly from libmqx and is said for GTX580, further inspection needed
static inline uint64_t calibrate_size(long size) {
  uint64_t size_calibrated = 0;
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
static cudaError_t attach_region(struct mps_region *rgn) {
  mqx_print_region(DEBUG, rgn, "attaching region");
  //pthread_mutex_lock(&rgn->mm_mutex);
  if (rgn->state == ATTACHED) {
    // TODO: update region position in LRU list
    __sync_fetch_and_add(&rgn->using_kernels, 1);
    mqx_print_region(DEBUG, rgn, "region already attached, using kernels(%d)", rgn->using_kernels);
    //pthread_mutex_unlock(&rgn->mm_mutex);
    return cudaSuccess;
  }
  if (rgn->state == ZOMBIE) {
    mqx_print_region(ERROR, rgn, "attaching a zombie region");
    //pthread_mutex_unlock(&rgn->mm_mutex);
    return cudaErrorInvalidValue;
  }
  // EVICTED || DETACHED
  mqx_print(DEBUG, "calibrate size(%ld), gpumem total(%ld), gpumem used(%ld)", calibrate_size(rgn->size), pglobal->gpumem_total, pglobal->gpumem_used);
  cudaError_t ret = cudaSuccess;
  if (calibrate_size(rgn->size) <= gpu_freemem()) {
    if ((ret = cuMemAlloc((CUdeviceptr *)&(rgn->gpu_addr), rgn->size)) != cudaSuccess) {
      cuGetErrorString(ret, &_err_buf);
      mqx_print(DEBUG, "cuMemAlloc failed, %s(%d)", _err_buf, ret);
      //pthread_mutex_unlock(&rgn->mm_mutex);
      return ret;
    }
  }
  // TODO: if out of memory, evict and try to allocate gpu memory again
  mqx_print(DEBUG, "cuMemAlloc %lu bytes @ %p", rgn->size, (void *)rgn->gpu_addr);
  // TODO: maintain this value
  //pglobal->gpumem_used += calibrate_size(rgn->size);
  __sync_fetch_and_add(&rgn->using_kernels, 1);
  rgn->state = ATTACHED;
  add_attached_region(rgn);
  //pthread_mutex_unlock(&rgn->mm_mutex);
  toggle_region_valid(rgn, 0/*gpu*/, 0/*invalid*/);
  return ret;
}
static inline void toggle_region_valid(struct mps_region *rgn, uint8_t is_swap, uint8_t value) {
  //pthread_mutex_lock(&rgn->mm_mutex);
  //if (rgn->flags & FLAG_READONLY) {
  //  mqx_print_region(DEBUG, rgn, "toggling a READONLY region");
  //  return;
  //}
  for (uint32_t i = 0; i < rgn->nblocks; i++) {
    if (is_swap) {
      rgn->blocks[i].swap_valid = value;
    } else {
      rgn->blocks[i].gpu_valid = value;
    }
  }
  //pthread_mutex_unlock(&rgn->mm_mutex);
}
static inline uint64_t gpu_freemem() {
  return pglobal->gpumem_total - pglobal->gpumem_used;
}
static inline uint8_t contains_ptr(const void **range, uint32_t len, const void *target) {
  for (uint32_t i = 0; i < len; i++) {
    if (target == range[i]) {
      return 1;
    }
  }
  return 0;
}
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
struct mps_region *find_allocated_region(struct global_context *pglobal, const void *ptr) {
  uint64_t addr = (uint64_t)ptr;
  uint64_t start_addr, end_addr;
  struct list_head *pos;
  struct mps_region *rgn;
  uint8_t found = 0;
  
  pthread_mutex_lock(&pglobal->alloc_mutex);
  list_for_each(pos, &pglobal->allocated_regions) {
    rgn = list_entry(pos, struct mps_region, entry_alloc);
    if (rgn->state == ZOMBIE) {
      mqx_print_region(ERROR, rgn, "found ZOMBIE region");
      continue;
    }
    start_addr = (uint64_t)(rgn->swap_addr);
    end_addr = start_addr + (uint64_t)(rgn->size);
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
static uint64_t fetch_and_mark_regions(struct mps_client *client, struct mps_region ***prgns, uint32_t *pnrgns) {
  struct client_kernel_conf *kconf = &client->kconf;
  struct mps_region **rgns, *rgn;
  uint32_t nrgns_upper_bound = 0;
  ASSERT(kconf->nadvices >= 0 && kconf->nadvices <= MAX_ARGS, "invalid number of advices(%d)", kconf->nadvices);
  ASSERT(kconf->nargs >= 0 && kconf->nargs <= MAX_ARGS, "invalid number of arguments(%d)", kconf->nargs);

  // get upper bound of unique regions number
  for (int i = 0; i < kconf->nargs; i++) {
    if (kconf->kargs[i].is_dptr) {
      nrgns_upper_bound++;
      rgn = kconf->kargs[i].dptr.rgn;
      if (rgn->flags & FLAG_PTARRAY) {
        nrgns_upper_bound += rgn->size / sizeof(void *);
      }
    }
  }
  rgns = malloc(sizeof(void *) * nrgns_upper_bound);

  // fetch and mark unique referenced regions
  uint32_t nrgns = 0;
  uint64_t mem_total = 0;
  for (int i = 0; i < kconf->nargs; i++) {
    if (kconf->kargs[i].is_dptr) {
      rgn = kconf->kargs[i].dptr.rgn;
      if (!contains_ptr((const void **)rgns, nrgns, (const void *)rgn)) {
        mqx_print_region(DEBUG, rgn, "new region(%d), offset(%zu)", i, kconf->kargs[i].dptr.offset);
        rgn->advice = kconf->kargs[i].dptr.advice & CADV_MASK;
        rgns[nrgns++] = rgn;
        mem_total += rgn->size;
      } else {
        mqx_print_region(DEBUG, rgn, "re-referenced region(%d), offset(%zu)", i, kconf->kargs[i].dptr.offset);
        rgn->advice |= kconf->kargs[i].dptr.advice & CADV_MASK;
      }
      if (rgn->flags & FLAG_READONLY) {
        if (rgn->advice & CADV_OUTPUT) {
          mqx_print_region(ERROR, rgn, "output to a READONLY region");
          pthread_exit(NULL);
        }
      }

      if (rgn->flags & FLAG_PTARRAY) {
        void **pdptr = (void **)rgn->pta_addr;
        void **pend = (void **)(rgn->pta_addr + rgn->size);
        if (rgn->advice != CADV_INPUT) {
          mqx_print(WARN, "dptr array MUST be INPUT-ONLY, correcting it");
          rgn->advice = CADV_INPUT;
        }
        mem_total += rgn->size;
        while (pdptr < pend) {
          rgn = find_allocated_region(pglobal, *pdptr);
          if (rgn == NULL) {
            mqx_print(ERROR, "cannot find region for dptr(%p)", *pdptr);
            return cudaErrorInvalidDevicePointer;
          }
          if (!contains_ptr((const void **)rgns, nrgns, (const void *)rgn)) {
            mqx_print_region(DEBUG, rgn, "new pta region, offset(%zu)", kconf->kargs[i].dptr.offset);
            //rgn->advice = kconf->kargs[i].dptr.advice & CADV_MASK;
            rgns[nrgns++] = rgn;
            rgn->advice = ((kconf->kargs[i].dptr.advice & CADV_PTAINPUT) ? CADV_INPUT : 0) | ((kconf->kargs[i].dptr.advice & CADV_PTAOUTPUT) ? CADV_OUTPUT : 0);
            mem_total += rgn->size;
          } else {
            mqx_print_region(DEBUG, rgn, "re-referenced pta region, offset(%zu)", kconf->kargs[i].dptr.offset);
            rgn->advice |= ((kconf->kargs[i].dptr.advice & CADV_PTAINPUT) ? CADV_INPUT : 0) | ((kconf->kargs[i].dptr.advice & CADV_PTAOUTPUT) ? CADV_OUTPUT : 0);
          }
          if (rgn->flags & FLAG_READONLY) {
            if (rgn->advice & CADV_OUTPUT) {
              mqx_print_region(ERROR, rgn, "output to a READONLY region");
              pthread_exit(NULL);
            }
          }
          pdptr++;
        }
      } // FLAG_PTARRAY
    } // argument is dptr
  } // foreach argument

  *pnrgns = nrgns;
  if (nrgns > 0) {
    *prgns = rgns;
  } else {
    free(rgns);
    *prgns = NULL;
  }
  return mem_total;
}
// NOTE: may be concurrent executed by multiple threads on exit
static cudaError_t free_region(struct mps_region *rgn) {
  uint8_t yielded = 0;
  if (rgn->flags & FLAG_PERSIST) {
    mqx_print_region(DEBUG, rgn, "freeing a PERSIST region");
    return cudaSuccess;
  }
begin:
  //pthread_mutex_lock(&rgn->mm_mutex);
  switch (rgn->state) {
    // enough memory on gpu, just free it
    case ATTACHED:
      if (rgn->using_kernels == 0) {
        mqx_print_region(DEBUG, rgn, "freeing an ATTACHED but not used region");
        checkCudaErrors(cuMemFree((CUdeviceptr)rgn->gpu_addr));
        goto revoke_resources;
      }
      rgn->evict_cost = 0;
      //pthread_mutex_unlock(&rgn->mm_mutex);
      // region still being used by kernel(s), waiting for the end.
      // Because column data is not passed in by socket, but allocated by the server,
      // and shared among client processes, this should not be `cudaFree`ed, thus must
      // not come here, which is not handled for now.
      if (!yielded) {
        mqx_print_region(DEBUG, rgn, "region is attached, and being used, yield until region state change. using_kernels(%d) n_input(%d) n_output(%d)", rgn->using_kernels, rgn->n_input, rgn->n_output);
      }
      yielded = 1;
      //sched_yield();
      goto begin;
    case EVICTED:
      mqx_print(DEBUG, "freeing an EVICTED region");
      goto revoke_resources;
    case DETACHED:
      mqx_print(DEBUG, "freeing a DETACHED region");
      goto revoke_resources;
    case ZOMBIE:
      //pthread_mutex_unlock(&rgn->mm_mutex);
      mqx_print(ERROR, "freeing a ZOMBIE region");
      // TODO: ensure this only happends on shared regions(columns data), and means that some other thread has already freed this column when no kernel is using it
      return cudaSuccess;
  }
revoke_resources:
  remove_attached_region(rgn);
  remove_allocated_region(rgn);
  free(rgn->blocks);
  rgn->blocks = NULL;
  free(rgn->swap_addr);
  rgn->swap_addr = NULL;
  rgn->gpu_addr = 0;
  rgn->state = ZOMBIE;
  //pthread_mutex_unlock(&rgn->mm_mutex);
  pthread_mutex_destroy(&rgn->mm_mutex);

  //int _nrgns = 0;
  //struct list_head *_pos;
  //struct mps_region *_rgn;
  //mqx_print(DEBUG, "allocated regions");
  //list_for_each(_pos, &pglobal->allocated_regions) {
  //  _rgn = list_entry(_pos, struct mps_region, entry_alloc);
  //  printf("%p ", _rgn->swap_addr);
  //  _nrgns++;
  //}
  //printf("\n%d in total\n", _nrgns);
  //_nrgns = 0;
  //mqx_print(DEBUG, "attached regions");
  //list_for_each(_pos, &pglobal->attached_regions) {
  //  _rgn = list_entry(_pos, struct mps_region, entry_attach);
  //  printf("%p ", _rgn->swap_addr);
  //  _nrgns++;
  //}
  //printf("\n%d in total\n", _nrgns);

  //NOTE: do not free(rgn), leave it for client_destroy()
  return cudaSuccess;
}
static cudaError_t mpsserver_cudaMemcpyHostToSwap(struct mps_client *client, void *dst, void *src, size_t size) {
  struct mps_region *rgn;
  rgn = find_allocated_region(pglobal, dst);
  if (!rgn) {
    mqx_print(ERROR, "Host->Device, dst(%p) region not found", dst);
    return cudaErrorInvalidDevicePointer;
  }
  ASSERT(rgn->state != ZOMBIE, "Host->Device: dst(%p) region already freed", dst);
  if (size == 0) {
    mqx_print(WARN, "Host->Device, copy ZERO bytes");
    return cudaSuccess;
  }
  if (dst + size > rgn->swap_addr + rgn->size) {
    mqx_print(ERROR, "Host->Device, copied range is out of dst(%p) swap region boundary, swap(%p), size(%ld)", dst, rgn->swap_addr, rgn->size);
    return cudaErrorInvalidValue;
  }
  if ((rgn->flags & FLAG_READONLY) && rgn->persist_swap_valid) {
    return cudaSuccess;
  }
  return mpsserver_HtoS(client, rgn, dst, src, size);
}
static cudaError_t mpsserver_HtoS(struct mps_client *client, struct mps_region *rgn, void *dst, void *src, size_t size) {
  mqx_print_region(DEBUG, rgn, "start");
  if (rgn->flags & FLAG_PTARRAY) {
    return mpsserver_doHtoS_pta(rgn, dst, src, size);
  }
  cudaError_t ret = cudaSuccess;
  // finish the former whole-region lazy memset's non-overlapping parts
  if (rgn->flags & FLAG_MEMSET) {
    uint64_t offset = dst - rgn->swap_addr;
    // left part
    if (offset > 0) {
      ret = mpsserver_doMemset(client, rgn, rgn->swap_addr, rgn->memset_value, offset);
      if (ret != cudaSuccess) {
        return ret;
      }
    }
    // right part
    offset += size;
    if (offset < rgn->size) {
      ret = mpsserver_doMemset(client, rgn, rgn->swap_addr + offset, rgn->memset_value, rgn->size - offset);
      if (ret != cudaSuccess) {
        return ret;
      }
    }
    rgn->flags &= ~FLAG_MEMSET;
    rgn->memset_value = 0;
  }
  ret = mpsserver_doHtoS(client, rgn, dst, src, size);
  if (ret == cudaSuccess) {
    update_region_evict_cost(rgn);
  }
  return ret;
}
static cudaError_t mpsserver_DtoH(struct mps_client *client, struct mps_region *rgn, void *dst, void *src, size_t size) {
  mqx_print_region(DEBUG, rgn, "dst(%p) src(%p) size(%ld)", dst, src, size);
  cudaError_t ret = cudaSuccess;
  if (rgn->flags & FLAG_PTARRAY) {
    return mpsserver_doDtoS_pta(rgn, dst, src, size);
  }
  if (rgn->flags & FLAG_MEMSET) {
    memset(dst, rgn->memset_value, size);
    return cudaSuccess;
  }
  ret = mpsserver_doDtoH(client, rgn, dst, src, size);
  if (ret == cudaSuccess) {
    update_region_evict_cost(rgn);
  }
  return ret;
}
static cudaError_t mpsserver_doHtoS_pta(struct mps_region *rgn, void *dst, void *src, size_t size) {
  uint64_t offset = dst - rgn->swap_addr;
  if (offset % sizeof(void *)) {
    mqx_print(ERROR, "offset(%lu) not aligned", offset);
    return cudaErrorInvalidValue;
  }
  if (size % sizeof(void *)) {
    mqx_print(ERROR, "size(%lu) not aligned", size);
    return cudaErrorInvalidValue;
  }
  memcpy(rgn->pta_addr + offset, src, size);
  return cudaSuccess;
}
static cudaError_t mpsserver_doDtoS_pta(struct mps_region *rgn, void *dst, void *src, size_t size) {
  uint64_t offset = src - rgn->swap_addr;
  if (offset % sizeof(void *)) {
    mqx_print(ERROR, "offset(%lu) not aligned", offset);
    return cudaErrorInvalidValue;
  }
  if (size % sizeof(void *)) {
    mqx_print(ERROR, "size(%lu) not aligned", size);
    return cudaErrorInvalidValue;
  }
  memcpy(dst, rgn->pta_addr + offset, size);
  return cudaSuccess;
}
static cudaError_t mpsserver_doHtoS(struct mps_client *client, struct mps_region *rgn, void *dst, void *src, size_t size) {
  uint64_t offset = (uint64_t)(dst - rgn->swap_addr);
  uint64_t psrc = (uint64_t)src;
  const uint64_t end_offset = (uint64_t)(offset + size);
  const uint32_t istart = BLOCKIDX(offset);
  const uint32_t iend = BLOCKIDX(end_offset - 1);
  // TODO: a cleaner solution, e.g. a while loop?
  uint8_t *skipped = (uint8_t *)calloc(iend - istart + 1, sizeof(uint8_t));
  uint32_t transfer_size;
  cudaError_t ret = cudaSuccess;
  for (int i = istart; i <= iend; i++) {
    transfer_size = min(BLOCKUP(offset), end_offset) - offset;
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
    transfer_size = min(BLOCKUP(offset), end_offset) - offset;
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
  void *pdst = dst;  // dst is the server buf
  const uint64_t end_offset = (uint64_t)(offset + size);
  const uint32_t istart = BLOCKIDX(offset);
  const uint32_t iend = BLOCKIDX(end_offset - 1);
  // TODO: a cleaner solution, e.g. a while loop?
  uint8_t *skipped = (uint8_t *)calloc(iend - istart + 1, sizeof(uint8_t));
  uint32_t transfer_size;
  cudaError_t ret = cudaSuccess;
  for (int i = istart; i <= iend; i++) {
    transfer_size = min(BLOCKUP(offset), end_offset) - offset;
    ret = mpsserver_doDtoH_block(client, rgn, offset, pdst, transfer_size, i, 1, skipped + i - istart);
    if (ret != 0) {
      goto finish;
    }
    offset += transfer_size;
    pdst += transfer_size;
  }
  pdst = dst;
  offset = (uint64_t)(src - rgn->swap_addr);
  for (int i = istart; i <= iend; i++) {
    transfer_size = min(BLOCKUP(offset), end_offset) - offset;
    if (skipped[i - istart]) {
      ret = mpsserver_doDtoH_block(client, rgn, offset, pdst, transfer_size, i, 0, NULL);
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
  //if (skip_on_wait) {
  //  if (pthread_mutex_trylock(&rgn->mm_mutex)) {
  //    *skipped = 1;
  //    return cudaSuccess;
  //  }
  //} else {
  //  pthread_mutex_lock(&rgn->mm_mutex);
  //}

  struct mps_block *blk = &rgn->blocks[iblock];
  ASSERT(BLOCKIDX(offset) == iblock, "  doHtoS_block: offset and block index do not match");

  cudaError_t ret = cudaSuccess;
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
  //pthread_mutex_unlock(&rgn->mm_mutex);
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

  //if (skip_on_wait) {
  //  if (pthread_mutex_trylock(&rgn->mm_mutex)) {
  //    *skipped = 1;
  //    return cudaSuccess;
  //  }
  //} else {
  //  pthread_mutex_lock(&rgn->mm_mutex);
  //}

  // block swap invalid
  cudaError_t ret = cudaSuccess;
  if (!blk->gpu_valid) {
    mqx_print(FATAL, "both swap & gpu memory are invalid");
    ret = cudaErrorUnknown;
    goto end;
  }
  // swap invalid && gpu valid
  ret = mpsserver_sync_block(client, rgn, iblock);
  if (ret != cudaSuccess) {
    goto end;
  }
  mqx_print(DEBUG, "doDtoH_block synced");
  memcpy(dst, rgn->swap_addr + offset, transfer_size);
end:
  //pthread_mutex_unlock(&rgn->mm_mutex);
  return ret;
}
static cudaError_t mpsserver_sync_block(struct mps_client *client, struct mps_region *rgn, uint32_t iblock) {
  uint8_t swap_valid = rgn->blocks[iblock].swap_valid;
  uint8_t gpu_valid = rgn->blocks[iblock].gpu_valid;
  double time;
  stats_time_begin();

  if ((gpu_valid && swap_valid) || (!gpu_valid && !swap_valid)) {
    return cudaSuccess;
  }
  ASSERT(rgn->swap_addr != NULL && rgn->gpu_addr, "swap(%p) or gpu(%p) is NULL", rgn->swap_addr, (void *)rgn->gpu_addr);

  { int i = 0;
  while (rgn->n_output > 0) {
    //sched_yield();
    i++;
  }
  }
  uint32_t offset = iblock * BLOCKSIZE;
  uint32_t size = min(offset + BLOCKSIZE, rgn->size) - offset;
  cudaError_t ret = cudaSuccess;
  if (gpu_valid && !swap_valid) {
    mqx_print(DEBUG, "syncing block DtoS: rgn(%p) block(%d) offset(%d) size(%d) swap_valid(%d) gpu_valid(%d)", rgn, iblock, offset, size, swap_valid, gpu_valid);
    ret = mpsserver_sync_blockDtoS(client, rgn->swap_addr + offset, rgn->gpu_addr + offset, size);
    if (ret == cudaSuccess) {
      rgn->blocks[iblock].swap_valid = 1;
    }
  } else {
    { int i = 0;
    while (rgn->n_input > 0) {
      //sched_yield();
      i++;
    }
    }
    mqx_print(DEBUG, "syncing block StoD: rgn(%p) block(%d) offset(%d) size(%d) swap_valid(%d) gpu_valid(%d)", rgn, iblock, offset, size, swap_valid, gpu_valid);
    ret = mpsserver_sync_blockStoD(client, rgn->gpu_addr + offset, rgn->swap_addr + offset, size);
    if (ret == cudaSuccess) {
      rgn->blocks[iblock].gpu_valid = 1;
    }
  }
  stats_time_end(time);
  stats.time_sync_block += time;
  return ret;
}
static cudaError_t mpsserver_sync_blockDtoS(struct mps_client *client, void *dst, CUdeviceptr src, size_t size) {
  double time;
  stats_time_begin();
  pthread_mutex_lock(&pglobal->dma_dtoh.mutex);
  struct mps_dma_channel *channel = &pglobal->dma_dtoh;
  cudaError_t ret = cudaSuccess;
  uint64_t offset_dtos, offset_stoh, round_size, ibuflast;
  offset_dtos = 0;
  ibuflast = channel->ibuf;
  // first initiate DMA to fully occupy all dma stage buffer
  while ((offset_dtos < size) && (offset_dtos < DMA_NBUF * DMA_BUF_SIZE)) {
    round_size = min(offset_dtos + DMA_BUF_SIZE, size) - offset_dtos;
    if ((ret = cuMemcpyDtoHAsync(channel->stage_buf[channel->ibuf], src + offset_dtos, round_size, channel->stream) != CUDA_SUCCESS)) {
      mqx_print(FATAL, "cuMemcpyDtoHAsync failed in blockDtoH");
      goto end;
    }
    if (cuEventRecord(channel->events[channel->ibuf], channel->stream) != CUDA_SUCCESS) {
      mqx_print(FATAL, "cuEventRecord failed in blockDtoH");
      goto end;
    }
    channel->ibuf = (channel->ibuf + 1) % DMA_NBUF;
    offset_dtos += round_size;
  }
  //cuEventSynchronize(channel->events[0]);
  //cuEventSynchronize(channel->events[1]);
  //stats_time_end(time);
  //stats.time_syncDtoS1 += time;

  channel->ibuf = ibuflast;
  offset_stoh = 0;
  while (offset_stoh < size) {
    if ((ret = cuEventSynchronize(channel->events[channel->ibuf])) != cudaSuccess) {
      mqx_print(FATAL, "cuEventSynchronize failed in blockDtoH");
      goto end;
    }
    struct timeval t3, t4;
    gettimeofday(&t3, NULL);
    round_size = min(offset_stoh + DMA_BUF_SIZE, size) - offset_stoh;
    memcpy(dst + offset_stoh, channel->stage_buf[channel->ibuf], round_size);
    gettimeofday(&t4, NULL);
    offset_stoh += round_size;
    stats.time_syncDtoS1 += TDIFFMS(t3, t4);

    if (offset_dtos < size) {
      round_size = min(offset_dtos + DMA_BUF_SIZE, size) - offset_dtos;
      if ((ret = cuMemcpyDtoHAsync(channel->stage_buf[channel->ibuf], src + offset_dtos, round_size, channel->stream) != CUDA_SUCCESS)) {
        mqx_print(FATAL, "cuMemcpyDtoHAsync failed in blockDtoH");
        goto end;
      }
      if (cuEventRecord(channel->events[channel->ibuf], channel->stream) != CUDA_SUCCESS) {
        mqx_print(FATAL, "cuEventRecord failed in blockDtoH");
        goto end;
      }
      offset_dtos += round_size;
    }
    channel->ibuf = (channel->ibuf + 1) % DMA_NBUF;
  }
end:
  pthread_mutex_unlock(&pglobal->dma_dtoh.mutex);
  stats_time_end(time);
  stats.time_syncDtoS += time;
  return ret;
}
static cudaError_t mpsserver_sync_blockStoD(struct mps_client *client, CUdeviceptr dst, void *src, size_t size) {
  double time;
  stats_time_begin();
  pthread_mutex_lock(&pglobal->dma_htod.mutex);
  struct mps_dma_channel *channel = &pglobal->dma_htod;
  enum cudaError_enum ret;
  uint64_t offset, round_size, ibuflast;
  offset = 0;
  while (offset < size) {
    if ((ret = cuEventSynchronize(channel->events[channel->ibuf])) != CUDA_SUCCESS) {
      mqx_print(FATAL, "cuEventSynchronize failed in blockHtoD");
      goto end;
    }
    round_size = min(offset + DMA_BUF_SIZE, size) - offset;
    memcpy(channel->stage_buf[channel->ibuf], src + offset, round_size);
    if ((ret = cuMemcpyHtoDAsync(dst + offset, channel->stage_buf[channel->ibuf], round_size, channel->stream) != CUDA_SUCCESS)) {
      mqx_print(FATAL, "cuMemcpyHtoDAsync failed in blockHtoD");
      goto end;
    }
    if (cuEventRecord(channel->events[channel->ibuf], channel->stream) != CUDA_SUCCESS) {
      mqx_print(FATAL, "cuEventRecord failed in blockHtoD");
      goto end;
    }
    ibuflast = channel->ibuf;
    channel->ibuf = (channel->ibuf + 1) % DMA_NBUF;
    offset += round_size;
  }
  if ((ret = cuEventSynchronize(channel->events[ibuflast])) != CUDA_SUCCESS) {
    mqx_print(FATAL, "cuEventSynchronize failed in blockHtoD");
    goto end;
  }
end:
  pthread_mutex_unlock(&pglobal->dma_htod.mutex);
  stats_time_end(time);
  stats.time_syncStoD += time;
  return ret;
}
static cudaError_t mpsserver_cudaMemcpyDeviceToHost(struct mps_client *client, void *dst, void *src, size_t size) {
  struct mps_region *rgn;
  // find swap addr `src`
  rgn = find_allocated_region(pglobal, src);
  if (!rgn) {
    mqx_print(ERROR, "Device->Host, src(%p) region found", src);
    return cudaErrorInvalidDevicePointer;
  }
  ASSERT(rgn->state != ZOMBIE, "Device->Host: src(%p) region already freed", src);
  if (size == 0) {
    mqx_print(WARN, "Device->Host, copy ZERO bytes");
    return cudaSuccess;
  }
  if (src + size > rgn->swap_addr + rgn->size) {
    mqx_print(ERROR, "Device->Host, copied range is out of src(%p) swap region boundary, swap(%p), size(%ld)", dst, rgn->swap_addr, rgn->size);
    return cudaErrorInvalidValue;
  }
  if (rgn->flags & FLAG_READONLY) {
    mqx_print_region(ERROR, rgn, "overwriting a READONLY region");
    return cudaErrorInvalidDevicePointer;
  }
  return mpsserver_DtoH(client, rgn, dst, src, size);
}
static cudaError_t mpsserver_cudaMemcpyDeviceToDevice(struct mps_client *client, void *dst, void *src, size_t size) {
  struct mps_region *rgn_dst, *rgn_src;
  double time;
  stats_time_begin();

  if (size == 0) {
    mqx_print(WARN, "copy zero byte");
    return cudaSuccess;
  }

  if ((rgn_src = find_allocated_region(pglobal, src)) == NULL) {
    mqx_print(ERROR, "Device->Device, region not found for src(%p)", src);
    return cudaErrorInvalidDevicePointer;
  }
  if (rgn_src->state == ZOMBIE) {
    mqx_print_region(ERROR, rgn_src, "src region is a ZOMBIE");
    return cudaErrorInvalidValue;
  }
  if (rgn_src->state == DETACHED) {
    mqx_print_region(ERROR, rgn_src, "src region is DETACHED");
    return cudaErrorInvalidValue;
  }
  if (src + size > rgn_src->swap_addr + rgn_src->size) {
    mqx_print_region(ERROR, rgn_src, "src(%p) range is too long, size(%lu)", src, size);
    return cudaErrorInvalidValue;
  }

  if ((rgn_dst = find_allocated_region(pglobal, dst)) == NULL) {
    mqx_print(ERROR, "region not found for dst(%p)", dst);
    return cudaErrorInvalidDevicePointer;
  }
  if (rgn_dst->state == ZOMBIE) {
    mqx_print_region(ERROR, rgn_dst, "dst region is a ZOMBIE");
    return cudaErrorInvalidValue;
  }
  if (dst + size > rgn_dst->swap_addr + rgn_dst->size) {
    mqx_print_region(ERROR, rgn_dst, "dst(%p) range is too long, size(%lu)", dst, size);
    return cudaErrorInvalidValue;
  }
  if (rgn_dst->flags & FLAG_READONLY) {
    mqx_print_region(ERROR, rgn_dst, "overwriting a READONLY region");
    return cudaErrorInvalidDevicePointer;
  }
  cudaError_t ret = mpsserver_DtoD(client, rgn_dst, rgn_src, dst, src, size);
  stats_time_end(time);
  stats.time_memcpyDtoD += time;
  return ret;
}
// NOTE: data should be copied from src to dst directly on GPU
// TODO: double check correctness
static cudaError_t mpsserver_DtoD(struct mps_client *client, struct mps_region *rgn_dst, struct mps_region *rgn_src, void *dst, void *src, size_t size) {
  struct mps_block *blk;
  cudaError_t ret = cudaSuccess;
  //pthread_mutex_lock(&rgn_dst->mm_mutex);
  //pthread_mutex_lock(&rgn_src->mm_mutex);
  if (rgn_dst->state == DETACHED) {
    //pthread_mutex_unlock(&rgn_dst->mm_mutex);
    if ((ret = attach_region(rgn_dst)) != cudaSuccess) {
      mqx_print(ERROR, "attach region failed");
      goto fail;
    }
    mqx_print(DEBUG, "rgn_dst is DETACHED, may have only just been `cudaMalloc`ed, so attach it");
    //pthread_mutex_lock(&rgn_dst->mm_mutex);
    // NOTE: using_kernel is increased after a successful region attachment, but no kernel is actually using it, and it cannot be decreased by any kernel callback because it is attached here, just for DtoD memcpy but not before a kernel launch. so maybe we can decrease it by one here
    __sync_fetch_and_sub(&rgn_dst->using_kernels, 1);
  }
  if (rgn_dst->flags & FLAG_MEMSET) {
    if ((ret = load_region_memset(client, rgn_dst)) != cudaSuccess) {
      goto fail;
    }
  } else {
    for (uint32_t i = 0; i < rgn_dst->nblocks; i++) {
      blk = &rgn_dst->blocks[i];
      if (!blk->gpu_valid) {
        // !gpu_valid && !swap_valid condition is cleared in branches above, thus swap_valid here
        mqx_print_region(DEBUG, rgn_dst, "sync dst in DtoD");
        if ((ret = mpsserver_sync_block(client, rgn_dst, i)) != cudaSuccess) {
          mqx_print(ERROR, "sync block failed");
          goto fail;
        }
      }
    }
  }
  if (rgn_src->flags & FLAG_MEMSET) {
    if ((ret = load_region_memset(client, rgn_src)) != cudaSuccess) {
      goto fail;
    }
  } else {
    for (uint32_t i = 0; i < rgn_src->nblocks; i++) {
      blk = &rgn_src->blocks[i];
      if (!blk->gpu_valid) { // !gpu_valid && !swap_valid condition is cleared in memset branch above, thus swap_valid here
        mqx_print_region(DEBUG, rgn_src, "sync src in DtoD");
        if ((ret = mpsserver_sync_block(client, rgn_src, i)) != cudaSuccess) {
          mqx_print(ERROR, "sync block failed");
          goto fail;
        }
      }
    }
  }

  uint64_t offset_src = src - rgn_src->swap_addr;
  uint64_t offset_dst = dst - rgn_dst->swap_addr;
  if ((ret = cuMemcpyDtoDAsync(rgn_dst->gpu_addr + offset_dst, rgn_src->gpu_addr + offset_src, size, client->stream)) != cudaSuccess) {
    mqx_print(ERROR, "cuMemcpyDtoDAsync failed, ret(%d)", ret);
    goto fail;
  }
  //pthread_mutex_unlock(&rgn_dst->mm_mutex);
  //pthread_mutex_unlock(&rgn_src->mm_mutex);
  toggle_region_valid(rgn_dst, 0/*gpu*/, 1);
  toggle_region_valid(rgn_dst, 1/*swap*/, 0);
  return ret;
fail:
  //pthread_mutex_unlock(&rgn_dst->mm_mutex);
  //pthread_mutex_unlock(&rgn_src->mm_mutex);
  return ret;
}
static cudaError_t mpsserver_cudaMemcpyDefault(struct mps_client *client, void *dst, void *src, size_t size) {
  return cudaErrorNotYetImplemented;
}
// TODO: change runtime api to driver api calls for dma_channel
int dma_channel_init(struct mps_dma_channel *channel, int isHtoD) {
  int i;
  channel->ibuf = 0;
  for (i = 0; i < DMA_NBUF; i++) {
    if (cuMemHostAlloc(&channel->stage_buf[i], DMA_BUF_SIZE, isHtoD ? CU_MEMHOSTALLOC_WRITECOMBINED : 0) != CUDA_SUCCESS) {
      mqx_print(FATAL, "failed to create the staging buffer.");
      break;
    }
    // KILLME: FIXME: this is SOOOOO AWKWARD because memcpy from pinned memory is TOOOOO SLOW in block_syncDtoH, check it later
    //channel->stage_buf[i] = malloc(DMA_BUF_SIZE);
    if (cuEventCreate(&channel->events[i], CU_EVENT_DISABLE_TIMING|CU_EVENT_BLOCKING_SYNC) != CUDA_SUCCESS) {
      mqx_print(FATAL, "failed to create staging buffer sync barrier.");
      cuMemFreeHost(channel->stage_buf[i]);
      break;
    }
  }
  if (i < DMA_NBUF) {
    while (--i >= 0) {
      cuEventDestroy(channel->events[i]);
      cuMemFreeHost(channel->stage_buf[i]);
    }
    return -1;
  }
  cuStreamCreate(&channel->stream, CU_STREAM_DEFAULT);
  pthread_mutex_init(&channel->mutex, NULL);
  return 0;
}
void dma_channel_destroy(struct mps_dma_channel *channel) {
  for (int i = 0; i < DMA_NBUF; i++) {
    cuEventDestroy(channel->events[i]);
    cuMemFreeHost(channel->stage_buf[i]);
  }
  cuStreamDestroy(channel->stream);
  pthread_mutex_destroy(&channel->mutex);
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

