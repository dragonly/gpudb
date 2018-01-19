#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>
#include <errno.h>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "advice.h"
#include "common.h"
#include "protocol.h"
#include "serialize.h"
#include "libmpsserver.h"
#include "kernel_symbols.h"

// TODO: check all kinds of pointers
// TODO: statistics

// foward declarations
static struct mps_region *find_allocated_region(struct global_context*, const void*);
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
static inline void toggle_region_valid(struct mps_region *rgn, uint8_t is_swap, uint8_t value);
static inline uint64_t calibrate_size(long size);
static inline uint64_t gpu_freemem();
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
static cudaError_t mpsserver_sync_blockStoD(struct mps_client *client, CUdeviceptr dst, void *src, size_t size);
static cudaError_t mpsserver_sync_blockDtoS(struct mps_client *client, void *dst, CUdeviceptr src, size_t size);
static void update_region_evict_cost(struct mps_region *rgn);
static void kernel_finish_callback(CUstream stream, CUresult ret, void *data);
static inline uint8_t contains_ptr(const void **range, uint32_t len, const void *target);

// in mpsserver.c
extern struct global_context *pglobal;
extern CUfunction F_vectorAdd;

cudaError_t mpsserver_cudaMalloc(void **devPtr, size_t size, uint32_t flags) {
  if (size == 0) {
    mqx_print(WARN, "allocating 0 bytes");
  } else if (size > pglobal->mem_total) {
    return cudaErrorMemoryAllocation;
  }
  //mqx_print(DEBUG, "allocate %zu bytes, %zu bytes free", size, pglobal->mem_total);

  struct mps_region *rgn;
  rgn = (struct mps_region *)calloc(1, sizeof(struct mps_region));
  rgn->swap_addr = malloc(size);
  rgn->gpu_addr = 0;
  // TODO: FLAG_PTARRAY
  rgn->nblocks = NBLOCKS(size);
  rgn->blocks = (struct mps_block *)calloc(rgn->nblocks, sizeof(struct mps_block));
  for (int i = 0; i < rgn->nblocks; i++) {
    pthread_mutex_init(&rgn->blocks[i].mutex_lock, NULL);
  }
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
  cudaError_t ret;
  mqx_print_region(DEBUG, rgn, "mpsserver_cudaFree");
  if ((ret = free_region(rgn) != cudaSuccess)) {
    return ret;
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

cudaError_t mpsserver_cudaSetFunction(struct mps_client *client, int index) {
  client->kconf.func_index = index;
  return cudaSuccess;
}
cudaError_t mpsserver_cudaAdvise(struct mps_client *client, int iarg, int advice) {
  int i;
  struct client_kernel_conf *kconf = &client->kconf;
  mqx_print(DEBUG, "cudaAdvise: %d %d", iarg, advice);

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
  mqx_print(DEBUG, "cudaConfigureCall: <<<(%d %d %d), (%d %d %d), %lu, %p>>>", gDim.x, gDim.y, gDim.z, bDim.x, bDim.y, bDim.z, shMem, stream);
  kconf->nargs = 0;
  kconf->ktop = (void *)kconf->kstack;
  kconf->gridDim = gDim;
  kconf->blockDim = bDim;
  kconf->sharedMem = shMem;
  // TODO: use custom stream
  return cudaSuccess;
}
cudaError_t mpsserver_cudaSetupArgument(struct mps_client *client, void *parg, size_t size, size_t offset) {
  struct mps_region *rgn;
  uint8_t is_dptr = 0;
  uint32_t iadv = 0;
  struct client_kernel_conf *kconf = &client->kconf;
  mqx_print(DEBUG, "cudaSetupArgument %d: size(%lu) offset(%lu)", kconf->nargs, size, offset);

  // FIXME: this is buggy, because in this setting, any primitive arguments must NOT be `cudaAdvise`d
  if (kconf->nadvices > 0) {
    for (iadv = 0; iadv < kconf->nadvices; iadv++) {
      if (kconf->advice_index[iadv] == kconf->nargs) {
        break;
      }
    }
    if (iadv < kconf->nadvices) {
      if (size != sizeof(void *)) {
        mqx_print(ERROR, "cudaSetupArgument (%d): Argument size (%lu) does not match size of dptr", kconf->nargs, size);
        return cudaErrorInvalidValue;
      }
      rgn = find_allocated_region(pglobal, *(void **)parg);
      if (!rgn) {
        mqx_print(ERROR, "cudaSetupArgument (%d): Cannot find region containing %p", kconf->nargs, parg);
        return cudaErrorInvalidValue;
      }
      is_dptr = 1;
    }
  } else if (size == sizeof(void *)) {
    mqx_print(WARN, "trying to parse dptr argument automatically, which is ERROR-PRONE!");
    rgn = find_allocated_region(pglobal, *(void **)parg);
    is_dptr = (rgn != NULL);
  } else {
    mqx_print(DEBUG, "cudaSetupArgument: primitive type argument");
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
};
cudaError_t mpsserver_cudaLaunchKernel(struct mps_client *client) {
  struct client_kernel_conf *kconf = &client->kconf;
  cudaError_t ret = cudaSuccess;
  struct mps_region **rgns = NULL;
  uint32_t nrgns = 0;
  uint64_t gpumem_required = 0;
  mqx_print(DEBUG, "start launching kernel");

  // nrgns can be 0 for some kernels only using registers
  // advices are marked on regions
  gpumem_required = fetch_and_mark_regions(client, &rgns, &nrgns);
  if (gpumem_required < 0) {
    mqx_print(ERROR, "cudaLaunch: failed to get required regions");
    ret = cudaErrorUnknown;
    goto launch_fail;
  } else if (gpumem_required > pglobal->mem_total) {
    mqx_print(ERROR, "cudaLaunch: out of memory (%ld required/%ld total)", gpumem_required, pglobal->mem_total);
    ret = cudaErrorInvalidConfiguration;
    goto launch_fail;
  }

attach:
  pthread_mutex_lock(&pglobal->kernel_launch_mutex);
  ret = attach_regions(rgns, nrgns);
  if (ret != cudaSuccess) {
    if (ret == cudaErrorLaunchTimeout) {
      sched_yield();
      goto attach;
    } else {
      mqx_print(FATAL, "cudaLaunch: attach regions failed (%d)", ret);
      goto launch_fail;
    }
  }

  ret = load_regions(client, rgns, nrgns);
  if (ret != cudaSuccess) {
    mqx_print(FATAL, "cudaLaunch: load regions failed (%d)", ret);
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
  if (nrgns > 0) {
    memcpy(kcb->rgns, rgns, sizeof(void *) * nrgns);
  }
  for (int i = 0; i < nrgns; i++) {
    kcb->advices[i] = rgns[i]->advice;
    if (kcb->advices[i] & CADV_OUTPUT) {
      rgns[i]->n_output++;
    }
    if (kcb->advices[i] & CADV_INPUT) {
      rgns[i]->n_input++;
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
    mqx_print(FATAL, "cuLaunchKernel failed: %s(%d)", cudaGetErrorString(cret), cret);
    for (int i = 0; i < nrgns; i++) {
      if (kcb->advices[i] & CADV_OUTPUT) {
        rgns[i]->n_output--;
      }
      if (kcb->advices[i] & CADV_INPUT) {
        rgns[i]->n_input--;
      }
    }
    free(kcb);
    free(kernel_args_buf);
    free(kernel_args_ptr);
    pthread_mutex_unlock(&pglobal->kernel_launch_mutex);
    return cret;
  }
  cuStreamAddCallback(client->stream, kernel_finish_callback, (void *)kcb, 0);
  // checkCudaErrors(cuStreamSynchronize(client->stream));
  // update clients position in LRU list
  //
  // NOTE: `kcb` cannot be freed here!!! it should be the work of `kernel_finish_callback`
  free(kernel_args_buf);
  free(kernel_args_ptr);
  pthread_mutex_unlock(&pglobal->kernel_launch_mutex);
  if (kconf->func_index != 233) {
    mqx_print(DEBUG, "launch kernel <%s> succeeded", fname_table[kconf->func_index]);
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
  cudaError_t ret;
  struct mps_region *rgn;
  mqx_print(DEBUG, "loading regions, %d in total", nrgns);
  for (uint32_t i = 0; i < nrgns; i++) {
    rgn = rgns[i];
    if (rgn->advice & CADV_INPUT) {
      if ((ret = load_region(client, rgn)) != cudaSuccess) {
        mqx_print_region(DEBUG, rgn, "load_region failed, %s(%d).", cudaGetErrorString(ret), ret);
        return ret;
      }
    }
    if (rgn->advice & CADV_OUTPUT) {
      if (rgn->flags & FLAG_MEMSET) {
        if ((ret = load_region_memset(client, rgn)) != cudaSuccess) {
          mqx_print_region(DEBUG, rgn, "load_region_memset failed, %s(%d).", cudaGetErrorString(ret), ret);
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
  cudaError_t ret;
  if (rgn->flags & FLAG_MEMSET) {
    ret = load_region_memset(client, rgn);
  } else if (rgn->flags & FLAG_PTARRAY) {
    // TODO
    ret = cudaErrorNotYetImplemented;
  } else {
    mqx_print_region(DEBUG, rgn, "loading region");
    for (uint32_t i = 0; i < NBLOCKS(rgn->size); i++) {
      if (!rgn->blocks[i].gpu_valid) {
        if ((ret = mpsserver_sync_block(client, rgn, i)) != cudaSuccess) {
          // TODO: print __FILE__ and __func__ in levels severer than WARN
          mqx_print(ERROR, "mpsserver_sync_block failed");
          return ret;
        }
      }
    }
    mqx_print(DEBUG, "region loaded");
  }
  return ret;
}
static cudaError_t load_region_memset(struct mps_client *client, struct mps_region *rgn) {
  cudaError_t ret;
  if ((ret = cuMemsetD32Async((CUdeviceptr)rgn->gpu_addr, rgn->memset_value, rgn->size, client->stream)) != cudaSuccess) {
    mqx_print(ERROR, "cuda memset failed: %s(%d)", cudaGetErrorString(ret), ret);
    return ret;
  }
  rgn->flags &= ~FLAG_MEMSET;
  rgn->memset_value = 0;
  toggle_region_valid(rgn, 0/*gpu*/, 1);
  return cudaSuccess;
}
static void kernel_finish_callback(CUstream stream, CUresult ret, void *data) {
  struct kernel_callback *kcb = (struct kernel_callback *)data;
  if (ret != CUDA_SUCCESS) {
    mqx_print(ERROR, "kernel execution failed: %s(%d)", cudaGetErrorString(ret), ret);
  } else {
    mqx_print(DEBUG, "kernel execution succeeded");
  }
  for (int i = 0; i < kcb->nrgns; i++) {
    if (!(kcb->advices[i] & CADV_MASK)) {
      continue;
    }
    pthread_mutex_lock(&kcb->rgns[i]->mm_mutex);
    if (kcb->advices[i] & CADV_OUTPUT) {
      kcb->rgns[i]->n_output--;
    }
    if (kcb->advices[i] & CADV_INPUT) {
      kcb->rgns[i]->n_input--;
    }
    kcb->rgns[i]->using_kernels--;
    pthread_mutex_unlock(&kcb->rgns[i]->mm_mutex);
  }
  free(kcb);
}

/**
 * helper functions
 */
static cudaError_t attach_regions(struct mps_region **rgns, uint32_t nrgns) {
  if (nrgns == 0) {
    return cudaSuccess;
  }
  if (nrgns < 0 || (nrgns > 0 && rgns == NULL)) {
    mqx_print(ERROR, "invalid value: rgns(%p) nrgns(%d)", rgns, nrgns);
  }
  mqx_print(DEBUG, "attaching regions, %d in total", nrgns);

  cudaError_t ret;
  struct mps_region *rgn;
  for (uint32_t i = 0; i < nrgns; i++) {
    rgn = rgns[i];
    if (rgn->state == ZOMBIE) {
      mqx_print(ERROR, "attach regions: region(%p) is in ZOMBIE state", rgn);
      ret = cudaErrorInvalidValue;
      goto fail;
    }
    pthread_mutex_lock(&rgn->mm_mutex);
    ret = attach_region(rgn);
    pthread_mutex_unlock(&rgn->mm_mutex);
    if (ret != cudaSuccess) {
      goto fail;
    }
  }

  mqx_print(DEBUG, "successfully attached regions, %d in total", nrgns);
  return ret;
fail:
// TODO: detach all regions
  return ret;
}
// TODO: this is copied directly from mqx and is said for GTX580, further inspection needed
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
  if (rgn->state == ATTACHED) {
    // TODO: update region position in LRU list
    rgn->using_kernels += 1;
    return cudaSuccess;
  }
  if (rgn->state != DETACHED) {
    mqx_print_region(ERROR, rgn, "attaching a non-detached region");
    return cudaErrorInvalidValue;
  }
  cudaError_t ret;
  if (calibrate_size(rgn->size) <= gpu_freemem()) {
    if ((ret = cuMemAlloc((CUdeviceptr *)&(rgn->gpu_addr), rgn->size)) == cudaSuccess) {
      mqx_print(DEBUG, "cuMemAlloc: %lu@%p", rgn->size, (void *)rgn->gpu_addr);
      goto success;
    } else {
      mqx_print(DEBUG, "attach region: cuMemAlloc failed, %s(%d)", cudaGetErrorString(ret), ret);
      if (ret == cudaErrorLaunchFailure) {
        return ret;
      }
    }
  }
  // TODO: try to evict and allocate gpu memory again
success:
  pglobal->gpumem_used += calibrate_size(rgn->size);
  rgn->using_kernels += 1;
  toggle_region_valid(rgn, 0/*gpu*/, 0/*invalid*/);
  rgn->state = ATTACHED;
  add_attached_region(rgn);
  return ret;
}
static inline void toggle_region_valid(struct mps_region *rgn, uint8_t is_swap, uint8_t value) {
  for (uint32_t i = 0; i < rgn->nblocks; i++) {
    if (is_swap) {
      rgn->blocks[i].swap_valid = value;
    } else {
      rgn->blocks[i].gpu_valid = value;
    }
  }
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
static uint64_t fetch_and_mark_regions(struct mps_client *client, struct mps_region ***prgns, uint32_t *pnrgns) {
  struct client_kernel_conf *kconf = &client->kconf;
  struct mps_region **rgns, *rgn;
  uint32_t nrgns = 0;
  ASSERT(kconf->nadvices >= 0 && kconf->nadvices <= MAX_ARGS, "invalid number of advices(%d)", kconf->nadvices);
  ASSERT(kconf->nargs >= 0 && kconf->nargs <= MAX_ARGS, "invalid number of arguments(%d)", kconf->nargs);

  // get upper bound of unique regions number
  for (int i = 0; i < kconf->nargs; i++) {
    if (kconf->kargs[i].is_dptr) {
      nrgns++;
      rgn = kconf->kargs[i].dptr.rgn;
      if (rgn->flags & FLAG_PTARRAY) {
        nrgns += rgn->size / sizeof(void *);
      }
    }
  }
  rgns = malloc(sizeof(struct mps_region *) * nrgns);
  CHECK_POINTER(rgns);

  // fetch and mark unique referenced regions
  nrgns = 0;
  uint64_t mem_total = 0;
  for (int i = 0; i < kconf->nargs; i++) {
    if (kconf->kargs[i].is_dptr) {
      rgn = kconf->kargs[i].dptr.rgn;
      if (!contains_ptr((const void **)rgns, nrgns, (const void *)rgn)) {
        mqx_print_region(DEBUG, rgn, "new referenced region, offset(%zu)", kconf->kargs[i].dptr.offset);
        rgn->advice = kconf->kargs[i].dptr.advice & CADV_MASK;
        rgns[nrgns++] = rgn;
        mem_total += rgn->size;
      } else {
        mqx_print_region(DEBUG, rgn, "re-referenced region, offset(%zu)", kconf->kargs[i].dptr.offset);
        rgn->advice |= kconf->kargs[i].dptr.advice & CADV_MASK;
      }

      if (rgn->flags & FLAG_PTARRAY) {
      } // TODO: FLAG_PTARRAY
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
static cudaError_t free_region(struct mps_region *rgn) {
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
      rgn->gpu_addr = 0;
      goto revoke_resources;
    case ZOMBIE:
      pthread_mutex_unlock(&rgn->mm_mutex);
      mqx_print(ERROR, "freeing a zombie region");
      return cudaErrorInvalidValue;
    default:
      pthread_mutex_unlock(&rgn->mm_mutex);
      mqx_print(FATAL, "no such region state");
      return cudaErrorInvalidValue;
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
  return cudaSuccess;
}
static cudaError_t mpsserver_cudaMemcpyHostToSwap(struct mps_client *client, void *dst, void *src, size_t size) {
  struct mps_region *rgn;
  rgn = find_allocated_region(pglobal, dst);
  if (!rgn) {
    mqx_print(ERROR, "Host->Device: dst(%p) region not found", dst);
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
  mqx_print(DEBUG, " HtoS: region{swap(%p) size(%ld) flags(%d), state(%d)}, dst(%p) src(%p) size(%ld)", rgn->swap_addr, rgn->size, rgn->flags, rgn->state, dst, src, size);
  // TODO: ptarray, memset
  cudaError_t ret = mpsserver_doHtoS(client, rgn, dst, src, size);
  if (ret == cudaSuccess) {
    update_region_evict_cost(rgn);
  }
  return ret;
}
static cudaError_t mpsserver_DtoH(struct mps_client *client, struct mps_region *rgn, void *dst, void *src, size_t size) {
  mqx_print_region(DEBUG, rgn, " DtoH: dst(%p) src(%p) size(%ld)", dst, src, size);
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
  void *pdst = dst;  // dst is the server buf
  const uint64_t end_addr = (uint64_t)(offset + size);
  const uint32_t istart = BLOCKIDX(offset);
  const uint32_t iend = BLOCKIDX(end_addr - 1);
  // TODO: a cleaner solution, e.g. a while loop?
  uint8_t *skipped = (uint8_t *)calloc(iend - istart + 1, sizeof(uint8_t));
  uint32_t transfer_size;
  cudaError_t ret;
  for (int i = istart; i <= iend; i++) {
    transfer_size = min(BLOCKUP(offset), end_addr) - offset;
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
    transfer_size = min(BLOCKUP(offset), end_addr) - offset;
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
  // swap invalid && gpu valid
  ret = mpsserver_sync_block(client, rgn, iblock);
  if (ret != cudaSuccess) {
    goto end;
  }
  mqx_print(DEBUG, " doDtoH_block synced");
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
  ASSERT(rgn->swap_addr != NULL && rgn->gpu_addr, "swap(%p) or gpu(%p) is NULL", rgn->swap_addr, (void *)rgn->gpu_addr);

  while (rgn->n_output > 0) {
    sched_yield();
  }
  uint32_t offset = iblock * BLOCKSIZE;
  uint32_t size = min(offset + BLOCKSIZE, rgn->size) - offset;
  cudaError_t ret;
  if (gpu_valid && !swap_valid) {
    mqx_print(DEBUG, "sync block DtoS: rgn(%p) block(%d) offset(%d) size(%d) swap_valid(%d) gpu_valid(%d)", rgn, iblock, offset, size, swap_valid, gpu_valid);
    ret = mpsserver_sync_blockDtoS(client, rgn->swap_addr + offset, rgn->gpu_addr + offset, size);
    if (ret == cudaSuccess) {
      rgn->blocks[iblock].swap_valid = 1;
    }
  } else {
    while (rgn->n_input > 0) {
      sched_yield();
    }
    mqx_print(DEBUG, "sync block StoD: rgn(%p) block(%d) offset(%d) size(%d) swap_valid(%d) gpu_valid(%d)", rgn, iblock, offset, size, swap_valid, gpu_valid);
    ret = mpsserver_sync_blockStoD(client, rgn->gpu_addr + offset, rgn->swap_addr + offset, size);
    if (ret == cudaSuccess) {
      rgn->blocks[iblock].gpu_valid = 1;
    }
  }
  return ret;
}
static cudaError_t mpsserver_sync_blockDtoS(struct mps_client *client, void *dst, CUdeviceptr src, size_t size) {
  struct mps_dma_channel *channel = &client->dma_htod;
  cudaError_t ret;
  uint64_t offset_dtos, offset_stoh, round_size, ibuflast;
  pthread_mutex_lock(&client->dma_mutex);
  offset_dtos = 0;
  ibuflast = channel->ibuf;
  // first initiate DMA to fully occupy all dma stage buffer
  while ((offset_dtos < size) && (offset_dtos < DMA_NBUF * DMA_BUF_SIZE)) {
    round_size = min(offset_dtos + DMA_BUF_SIZE, size) - offset_dtos;
    if ((ret = cuMemcpyDtoHAsync(channel->stage_buf[channel->ibuf], src + offset_dtos, round_size, client->stream) != CUDA_SUCCESS)) {
      mqx_print(FATAL, "cuMemcpyDtoHAsync failed in blockDtoH");
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
      if ((ret = cuMemcpyDtoHAsync(channel->stage_buf[channel->ibuf], src + offset_dtos, round_size, client->stream) != CUDA_SUCCESS)) {
        mqx_print(FATAL, "cuMemcpyDtoHAsync failed in blockDtoH");
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
static cudaError_t mpsserver_sync_blockStoD(struct mps_client *client, CUdeviceptr dst, void *src, size_t size) {
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
    if ((ret = cuMemcpyHtoDAsync(dst + offset, channel->stage_buf[channel->ibuf], round_size, client->stream) != CUDA_SUCCESS)) {
      mqx_print(FATAL, "cuMemcpyHtoDAsync failed in blockHtoD");
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
  // find swap addr `src`
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

