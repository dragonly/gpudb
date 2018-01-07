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
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <sched.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>

#include "advice.h"
#include "client.h"
#include "common.h"
#include "core.h"
#include "cow.h"
#include "libc_hooks.h"
#include "msq.h"
#include "replacement.h"

// CUDA runtime API handlers, defined in mqx_interfaces.c.
extern cudaError_t (*nv_cudaMalloc)(void **, size_t);
extern cudaError_t (*nv_cudaFree)(void *);
cudaError_t (*nv_cudaMemcpy)(void *, const void *, size_t, enum cudaMemcpyKind);
extern cudaError_t (*nv_cudaMemcpyAsync)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
extern cudaError_t (*nv_cudaStreamCreate)(cudaStream_t *);
extern cudaError_t (*nv_cudaStreamDestroy)(cudaStream_t);
#ifdef MQX_CONFIG_STAT_KERNEL_TIME
extern cudaError_t (*nv_cudaStreamSynchronize)(cudaStream_t);
#endif
extern cudaError_t (*nv_cudaMemGetInfo)(size_t *, size_t *);
extern cudaError_t (*nv_cudaSetupArgument)(const void *, size_t, size_t);
extern cudaError_t (*nv_cudaConfigureCall)(dim3, dim3, size_t, cudaStream_t);
extern cudaError_t (*nv_cudaMemsetAsync)(void *, int, size_t, cudaStream_t);
extern cudaError_t (*nv_cudaLaunch)(const void *);
extern cudaError_t (*nv_cudaStreamAddCallback)(cudaStream_t, cudaStreamCallback_t, void *, unsigned int);

static int mqx_free(struct region *r);
static int mqx_htod(struct region *r, void *dst, const void *src, size_t count);
static int mqx_htod_cow(struct region *r, void *dst, const void *src, size_t count);
// mqx_dtoh is used in debug.c. So export it.
int mqx_dtoh(struct region *r, void *dst, const void *src, size_t count);
static int mqx_dtod(struct region *rd, struct region *rs, void *dst, const void *src, size_t count);
static int mqx_do_memset(struct region *r, void *dst, int value, size_t count);
static int mqx_memset(struct region *r, void *dst, int value, size_t count);
static int mqx_attach(struct region **rgns, int n);
static int mqx_load(struct region **rgns, int nrgns);
static int mqx_launch(const char *entry, struct region **rgns, int nrgns);
struct region *region_lookup(struct local_context *ctx, const void *ptr);
static int block_sync(struct region *r, int block);
static int region_attach(struct region *r, int pin, struct region **excls, int nexcl);

// Per-process MQX context.
struct local_context *local_ctx = NULL;

static void local_alloc_list_add(struct region *r) {
  acquire(&local_ctx->lock_alloc);
  list_add(&r->entry_alloc, &local_ctx->list_alloc);
  release(&local_ctx->lock_alloc);
}

static void local_alloc_list_del(struct region *r) {
  acquire(&local_ctx->lock_alloc);
  list_del(&r->entry_alloc);
  release(&local_ctx->lock_alloc);
}

static void attach_list_add(struct region *r) {
  acquire(&local_ctx->lock_attach);
  list_add(&r->entry_attach, &local_ctx->list_attach);
  release(&local_ctx->lock_attach);
  global_attach_list_add(r);
}

static void attach_list_del(struct region *r) {
  global_attach_list_del(r);
  acquire(&local_ctx->lock_attach);
  list_del(&r->entry_attach);
  release(&local_ctx->lock_attach);
}

static void attach_list_mov(struct region *r) {
  acquire(&local_ctx->lock_attach);
  list_move(&r->entry_attach, &local_ctx->list_attach);
  release(&local_ctx->lock_attach);
  global_attach_list_mov(r);
}

static inline void region_pin(struct region *r) {
  int pinned = atomic_inc(&r->c_pinned);
  if (pinned == 0)
    inc_detachable_size(-size_cal(r->size));
  if (r->state == STATE_ATTACHED)
    global_attach_list_pin(r);
}

static inline void region_unpin(struct region *r) {
  int pinned;
  global_attach_list_unpin(r);
  pinned = atomic_dec(&r->c_pinned);
  if (pinned == 1)
    inc_detachable_size(size_cal(r->size));
}

static int dma_channel_init(struct dma_channel *chan, int htod) {
  long nfail = 0;
  int i;

  // cudaStreamCreate (and other CUDA runtime functions) will fail if
  // the device memory space has been depleted by other processes at
  // the moment. This should not prevent the local context from being
  // initialized, because once we join the global MQX context, free
  // device memory space can be made available for the user program.
  // To get over this transient period, we keep trying until the
  // function succeeds. Once one API succeeds, the following ones will
  // very likely succeed too. So this effectively addresses the problem,
  // at least for now.
  while (nv_cudaStreamCreate(&chan->stream) != cudaSuccess) {
    if (++nfail == 1)
      mqx_print(WARN, "Failed to create DMA stream; keep trying.");
    if (nfail > (LONG_MAX >> 1)) {
      mqx_print(FATAL, "Failed to create DMA stream after %ld trails.", nfail);
      return -1;
    }
  }

  initlock(&chan->lock);
  chan->ibuf = 0;
  for (i = 0; i < DMA_NBUFS; i++) {
    if (cudaHostAlloc(&chan->stage_bufs[i], DMA_BUFSIZE, htod ? cudaHostAllocWriteCombined : cudaHostAllocDefault) !=
        cudaSuccess) {
      mqx_print(FATAL, "Failed to create the staging buffer.");
      break;
    }
    if (cudaEventCreateWithFlags(&chan->events[i], cudaEventDisableTiming) != cudaSuccess) {
      mqx_print(FATAL, "Failed to create staging buffer sync barrier.");
      cudaFreeHost(chan->stage_bufs[i]);
      break;
    }
  }
  if (i < DMA_NBUFS) {
    while (--i >= 0) {
      cudaEventDestroy(chan->events[i]);
      cudaFreeHost(chan->stage_bufs[i]);
    }
    return -1;
  }

  return 0;
}

static void dma_channel_fini(struct dma_channel *chan) {
  int i;
  for (i = 0; i < DMA_NBUFS; i++) {
    cudaEventDestroy(chan->events[i]);
    cudaFreeHost(chan->stage_bufs[i]);
  }
  nv_cudaStreamDestroy(chan->stream);
}

static void dma_begin(struct dma_channel *chan) { acquire(&chan->lock); }

static void dma_end(struct dma_channel *chan) { release(&chan->lock); }

// Initializes local MQX context.
int context_init() {
  if (local_ctx != NULL) {
    mqx_print(ERROR, "Local context already exists!");
    return -1;
  }
  local_ctx = (struct local_context *)__libc_malloc(sizeof(*local_ctx));
  if (!local_ctx) {
    mqx_print(FATAL, "Failed to allocate memory for local context: %s", strerror(errno));
    return -1;
  }

  initlock(&local_ctx->lock);
  atomic_setl(&local_ctx->size_attached, 0L);
  INIT_LIST_HEAD(&local_ctx->list_alloc);
  INIT_LIST_HEAD(&local_ctx->list_attach);
  initlock(&local_ctx->lock_alloc);
  initlock(&local_ctx->lock_attach);
  if (dma_channel_init(&local_ctx->dma_htod, 1)) {
    mqx_print(FATAL, "Failed to create HtoD DMA channel.");
    __libc_free(local_ctx);
    local_ctx = NULL;
    return -1;
  }
  if (dma_channel_init(&local_ctx->dma_dtoh, 0)) {
    mqx_print(FATAL, "Failed to create DtoH DMA channel.");
    dma_channel_fini(&local_ctx->dma_htod);
    __libc_free(local_ctx);
    local_ctx = NULL;
    return -1;
  }
  if (nv_cudaStreamCreate(&local_ctx->stream_kernel) != cudaSuccess) {
    mqx_print(FATAL, "Failed to create kernel stream.");
    dma_channel_fini(&local_ctx->dma_dtoh);
    dma_channel_fini(&local_ctx->dma_htod);
    __libc_free(local_ctx);
    local_ctx = NULL;
    return -1;
  }
  stats_init(&local_ctx->stats);

  return 0;
}

void context_fini() {
  if (local_ctx != NULL) {
    // Release all dangling device memory regions.
    while (!list_empty(&local_ctx->list_alloc)) {
      struct list_head *p = local_ctx->list_alloc.next;
      struct region *r = list_entry(p, struct region, entry_alloc);
      if (!mqx_free(r))
        list_move_tail(p, &local_ctx->list_alloc);
    }
    // Destroy local context.
    dma_channel_fini(&local_ctx->dma_dtoh);
    dma_channel_fini(&local_ctx->dma_htod);
    nv_cudaStreamDestroy(local_ctx->stream_kernel);
    stats_print(&local_ctx->stats);
    __libc_free(local_ctx);
    local_ctx = NULL;
  }
}

// Allocates a new device memory region. Device memory space is not
// immediately allocated until the region is to be used in a kernel.
// Instead, a host swapping buffer area is allocated in the caller
// process's virtual memory, and its starting address is returned
// to the user program as the unique identifier of the region.
cudaError_t mqx_cudaMalloc(void **devPtr, size_t size, int flags) {
  struct region *r;

  mqx_print(DEBUG, "cudaMalloc begins: size(%lu) flags(0x%x)", size, flags);
  if (size > dev_mem_size()) {
    mqx_print(ERROR, "cudaMalloc size (%lu) too large (max %ld).", size, dev_mem_size());
    return cudaErrorInvalidValue;
  }
  if (size == 0) {
    // NOTE: libc and cudart do not consider malloc of size 0 as
    // an error, so we do not return error here either.
    mqx_print(WARN, "cudaMalloc size is zero.");
  }

  r = (struct region *)calloc(1, sizeof(*r));
  if (!r) {
    mqx_print(FATAL,
              "Failed to allocate host memory for a new "
              "region structure: %s.",
              strerror(errno));
    return cudaErrorMemoryAllocation;
  }
  r->swp_addr = __libc_malloc(size);
  if (!r->swp_addr) {
    mqx_print(FATAL,
              "Failed to allocate host swapping buffer "
              "for a new region: %s.",
              strerror(errno));
    __libc_free(r);
    return cudaErrorMemoryAllocation;
  }

  // Device memory regions containing device memory pointers are special.
  // They can only be modified from CPU, and are read-only by GPU kernels.
  // This constraint is made because the REAL pointer values may change
  // each time before a kernel is launched, when the device memory region
  // pointed to by a pointer may be attached and its starting device memory
  // address may have changed.
  if (flags & FLAG_PTARRAY) {
    if (size % sizeof(void *)) {
      mqx_print(ERROR, "Device pointer array size (%lu) not aligned.", size);
      __libc_free(r->swp_addr);
      __libc_free(r);
      return cudaErrorInvalidValue;
    }
    r->pta_addr = calloc(1, size);
    if (!r->pta_addr) {
      mqx_print(FATAL,
                "Failed to allocate host memory for a new device "
                "pointer array: %s.",
                strerror(errno));
      __libc_free(r->swp_addr);
      __libc_free(r);
      return cudaErrorMemoryAllocation;
    }
    mqx_print(DEBUG, "pta_addr (%p) alloced for %p", r->pta_addr, r);
  }

  r->nr_blocks = NRBLOCKS(size);
  r->blocks = (struct block *)calloc(r->nr_blocks, sizeof(struct block));
  if (!r->blocks) {
    mqx_print(FATAL, "Failed to allocate host memory for a new block array: %s.", strerror(errno));
    if (r->pta_addr)
      __libc_free(r->pta_addr);
    __libc_free(r->swp_addr);
    __libc_free(r);
    return cudaErrorMemoryAllocation;
  }

  r->size = (long)size;
  r->state = STATE_DETACHED;
  r->flags = flags;
  r->index = -1;
  local_alloc_list_add(r);
  stats_inc_alloc(&local_ctx->stats, size);
  mqx_print(DEBUG, "cudaMalloc ends: r(%p) swp(%p) pta(%p) nr_blocks(%d)",
            r, r->swp_addr, r->pta_addr, r->nr_blocks);

  *devPtr = r->swp_addr;
  return cudaSuccess;
}

cudaError_t mqx_cudaFree(void *devPtr) {
  struct region *r;

  if (!(r = region_lookup(local_ctx, devPtr))) {
    mqx_print(ERROR, "Cannot find a region containing address %p for freeing.", devPtr);
    return cudaErrorInvalidDevicePointer;
  }
  if (mqx_free(r) < 0)
    return cudaErrorUnknown;
  stats_inc_freed(&local_ctx->stats, r->size);

  return cudaSuccess;
}

cudaError_t mqx_cudaMemcpyHtoD(void *dst, const void *src, size_t count) {
  struct region *r;

  r = region_lookup(local_ctx, dst);
  if (!r) {
    mqx_print(ERROR,
              "HtoD: cannot find device memory region containing "
              "destination address %p",
              dst);
    return cudaErrorInvalidDevicePointer;
  }
  CHECK(r->state != STATE_FREEING && r->state != STATE_ZOMBIE, "HtoD: destination region already freed!");
  if (count == 0) {
    mqx_print(WARN, "HtoD: number of bytes to be transferred is zero");
    return cudaSuccess;
  }
  if (dst + count > r->swp_addr + r->size) {
    mqx_print(ERROR, "HtoD: copy is out of region boundary");
    return cudaErrorInvalidValue;
  }

  stats_time_begin();
  if (mqx_htod(r, dst, src, count) < 0)
    return cudaErrorUnknown;
  stats_time_end(&local_ctx->stats, time_htod);
  stats_inc(&local_ctx->stats, bytes_htod, count);

  return cudaSuccess;
}

cudaError_t mqx_cudaMemcpyDtoH(void *dst, const void *src, size_t count) {
  struct region *r;

  r = region_lookup(local_ctx, src);
  if (!r) {
    mqx_print(ERROR, "DtoH: cannot find device memory region containing source address %p", src);
    return cudaErrorInvalidDevicePointer;
  }
  CHECK(r->state != STATE_FREEING && r->state != STATE_ZOMBIE, "DtoH: region already freed");
  if (count == 0) {
    mqx_print(WARN, "DtoH: number of bytes to be transferred is zero");
    return cudaSuccess;
  }
  if (src + count > r->swp_addr + r->size) {
    mqx_print(ERROR, "DtoH: copy is out of region boundary");
    return cudaErrorInvalidValue;
  }

  stats_time_begin();
  if (mqx_dtoh(r, dst, src, count) < 0)
    return cudaErrorUnknown;
  stats_time_end(&local_ctx->stats, time_dtoh);
  stats_inc(&local_ctx->stats, bytes_dtoh, count);

  return cudaSuccess;
}

cudaError_t mqx_cudaMemcpyDtoD(void *dst, const void *src, size_t count) {
  struct region *rs, *rd;

  rs = region_lookup(local_ctx, src);
  if (!rs) {
    mqx_print(ERROR,
              "DtoD: cannot find the device memory region "
              "containing source address %p",
              src);
    return cudaErrorInvalidDevicePointer;
  }
  CHECK(rs->state != STATE_FREEING && rs->state != STATE_ZOMBIE, "Source device memory region already freed");
  rd = region_lookup(local_ctx, dst);
  if (!rd) {
    mqx_print(ERROR,
              "DtoD: cannot find the device memory region "
              "containing destination address %p",
              src);
    return cudaErrorInvalidDevicePointer;
  }
  CHECK(rd->state != STATE_FREEING && rd->state != STATE_ZOMBIE, "Destination device memory region already freed");
  if (count == 0) {
    mqx_print(WARN, "DtoD: number of bytes to be copied is zero");
    return cudaSuccess;
  }
  if (src + count > rs->swp_addr + rs->size) {
    mqx_print(ERROR, "DtoD: memory copy is out of source boundary");
    return cudaErrorInvalidValue;
  }
  if (dst + count > rd->swp_addr + rd->size) {
    mqx_print(ERROR, "DtoD: memory copy is out of destination boundary");
    return cudaErrorInvalidValue;
  }

  stats_time_begin();
  if (mqx_dtod(rd, rs, dst, src, count) < 0)
    return cudaErrorUnknown;
  stats_time_end(&local_ctx->stats, time_dtod);
  stats_inc(&local_ctx->stats, bytes_dtod, count);

  return cudaSuccess;
}

cudaError_t mqx_cudaMemcpyDefault(void *dst, const void *src, size_t count) {
  struct region *rs, *rd;

  rs = region_lookup(local_ctx, src);
  rd = region_lookup(local_ctx, dst);
  if (!rs && rd)
    return mqx_cudaMemcpyHtoD(dst, src, count);
  else if (rs && !rd)
    return mqx_cudaMemcpyDtoH(dst, src, count);
  else if (rs && rd)
    return mqx_cudaMemcpyDtoD(dst, src, count);
  else
    return nv_cudaMemcpy(dst, src, count, cudaMemcpyHostToHost);
}

cudaError_t mqx_cudaMemset(void *devPtr, int value, size_t count) {
  struct region *r;

  if (count == 0) {
    mqx_print(WARN, "memset: number of bytes to be set is zero.");
    return cudaSuccess;
  }
  r = region_lookup(local_ctx, devPtr);
  if (!r) {
    mqx_print(ERROR, "memset: cannot find region containing %p", devPtr);
    return cudaErrorInvalidDevicePointer;
  }
  CHECK(r->state != STATE_FREEING && r->state != STATE_ZOMBIE, "memset: region already freed");
  if (devPtr + count > r->swp_addr + r->size) {
    mqx_print(ERROR, "memset: out of region boundary");
    return cudaErrorInvalidValue;
  }

  stats_time_begin();
  if (mqx_memset(r, devPtr, value, count) < 0)
    return cudaErrorUnknown;
  stats_time_end(&local_ctx->stats, time_memset);
  stats_inc(&local_ctx->stats, bytes_memset, count);

  return cudaSuccess;
}

cudaError_t mqx_cudaMemGetInfo(size_t *size_free, size_t *size_total) {
  *size_free = (size_t)dev_mem_free();
  *size_total = (size_t)dev_mem_size();
  mqx_print(DEBUG, "cudaMemGetInfo: free(%lu) total(%lu)", *size_free, *size_total);
  return cudaSuccess;
}

// Data reference and access advices passed for a kernel launch.
// Set by cudaAdvice() in interfaces.c.
// TODO: have to reset nadvices in case of cudaLaunch errors.
extern int advice_refs[MAX_ARGS];
extern int advice_accs[MAX_ARGS];
extern int nadvices;
extern int func_index;

// The arguments for the following kernel to be launched.
// TODO: should prepare the following structures for each stream.
static dim3 grid;
static dim3 block;
static size_t shared;
static cudaStream_t stream_issue = 0;

static unsigned char kstack[512];   // Temporary kernel argument stack
static void *ktop = (void *)kstack; // Stack top
static struct karg kargs[MAX_ARGS];
static int nargs = 0;

// TODO: Currently, %stream_issue is always set to local_ctx->stream_kernel.
// This is not the best solution because it forbids kernels from being
// issued to different streams, which is required for, e.g., concurrent
// kernel executions.
// A better design is to prepare a kernel callback queue in local_ctx->kcb
// for each possible stream ; kernel callbacks are registered in the queues
// they are issued to. This both maintains the correctness of kernel callbacks
// and retains the capability that kernels can be issued to multiple streams.
cudaError_t mqx_cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
  mqx_print(DEBUG, "cudaConfigureCall: (%d %d %d) (%d %d %d) %lu %p", gridDim.x, gridDim.y, gridDim.z, blockDim.x,
            blockDim.y, blockDim.z, sharedMem, stream);
  nargs = 0;
  ktop = (void *)kstack;
  grid = gridDim;
  block = blockDim;
  shared = sharedMem;
  stream_issue = local_ctx->stream_kernel;
  return cudaSuccess;
}

// CUDA pushes kernel arguments from left to right. For example, for a kernel
//              k(a, b, c)
// , a will be pushed on top of the stack, followed by b, and finally c.
// %offset gives the actual offset of an argument in the call stack,
// rather than which argument is being pushed. %size is the size of the
// argument being pushed. Note that argument offset will be aligned based
// on the type of argument (e.g., a (void *)-type argument's offset has to
// be aligned to sizeof(void *)).
//
// Test whether the argument being set up is a device memory pointer.
// No matter it is a device memory pointer or not, we postpone
// its pushing until later when cudaLaunch is called.
// Use reference advices if given. Otherwise, parse automatically
// (but parsing errors are possible, e.g. when the user passes a
// long argument whose value happens to lay within some region's
// host swap buffer area).
cudaError_t mqx_cudaSetupArgument(const void *arg, size_t size, size_t offset) {
  struct region *r;
  int is_dptr = 0;
  int iadv = 0;

  mqx_print(DEBUG, "cudaSetupArgument (%d): size(%lu) offset(%lu)", nargs, size, offset);

  if (nadvices > 0) {
    for (iadv = 0; iadv < nadvices; iadv++) {
      if (advice_refs[iadv] == nargs)
        break;
    }
    if (iadv < nadvices) {
      if (size != sizeof(void *)) {
        mqx_print(ERROR,
                  "cudaSetupArgument (%d): Argument size (%lu) "
                  "does not match size of dptr",
                  nargs, size);
        return cudaErrorUnknown;
      }
      r = region_lookup(local_ctx, *(void **)arg);
      if (!r) {
        mqx_print(ERROR,
                  "cudaSetupArgument (%d): Cannot find region "
                  "containing %p",
                  nargs, *(void **)arg);
        return cudaErrorUnknown;
      }
      is_dptr = 1;
    }
  } else if (size == sizeof(void *)) {
    mqx_print(WARN, "Trying to parse dptr argument automatically; "
                    "this is ERROR prone!!!");
    r = region_lookup(local_ctx, *(void **)arg);
    if (r)
      is_dptr = 1;
  }

  if (is_dptr) {
    kargs[nargs].dptr.r = r;
    kargs[nargs].dptr.off = (unsigned long)(*(void **)arg - r->swp_addr);
    if (nadvices > 0)
      kargs[nargs].dptr.advice = advice_accs[iadv];
    else
      kargs[nargs].dptr.advice = CADV_DEFAULT | CADV_PTADEFAULT;
    mqx_print(DEBUG, "Argument is dptr: r(%p %p %ld %d %d)", r, r->swp_addr, r->size, r->flags, r->state);
  } else {
    // This argument is not a device memory pointer.
    // XXX: Currently we ignore the case that CUDA runtime might
    // stop pushing arguments due to some errors.
    memcpy(ktop, arg, size);
    kargs[nargs].ndptr = ktop;
    ktop += size;
  }
  kargs[nargs].is_dptr = is_dptr;
  kargs[nargs].size = size;
  kargs[nargs].argoff = offset;
  nargs++;

  return cudaSuccess;
}

static long regions_referenced(struct region ***prgns, int *pnrgns) {
  struct region **rgns, *r;
  int i, nrgns = 0;
  long total = 0;

  CHECK(nadvices <= MAX_ARGS && nadvices >= 0, "nadvices");
  CHECK(nargs <= MAX_ARGS && nargs >= 0, "nargs");

  // Get the upper bound of the number of unique regions.
  for (i = 0; i < nargs; i++) {
    if (kargs[i].is_dptr) {
      nrgns++;
      r = kargs[i].dptr.r;
      if (r->flags & FLAG_PTARRAY)
        nrgns += r->size / sizeof(void *);
    }
  }

  rgns = (struct region **)__libc_malloc(sizeof(*rgns) * nrgns);
  if (!rgns) {
    mqx_print(FATAL, "Failed to allocate memory for region array: %s", strerror(errno));
    return -1;
  }
  nrgns = 0;

  // Now set the regions to be referenced.
  for (i = 0; i < nargs; i++) {
    if (kargs[i].is_dptr) {
      r = kargs[i].dptr.r;
      if (!is_included((const void **)rgns, nrgns, (const void *)r)) {
        mqx_print(DEBUG,
                  "New referenced region(%p %p %ld %d %d) "
                  "off(%lu)",
                  r, r->swp_addr, r->size, r->flags, r->state, kargs[i].dptr.off);
        rgns[nrgns++] = r;
        r->accadv.advice = kargs[i].dptr.advice & CADV_MASK;
        total += r->size;
      } else {
        mqx_print(DEBUG,
                  "Existing referenced region"
                  "(%p %p %ld %d %d) off(%lu)",
                  r, r->swp_addr, r->size, r->flags, r->state, kargs[i].dptr.off);
        r->accadv.advice |= kargs[i].dptr.advice & CADV_MASK;
      }

      if (r->flags & FLAG_PTARRAY) {
        void **pdptr = (void **)(r->pta_addr);
        void **pend = (void **)(r->pta_addr + r->size);
        if (r->accadv.advice != CADV_INPUT) {
          mqx_print(ERROR, "Dptr array must be input-only!");
          r->accadv.advice = CADV_INPUT;
        }
        // For each device memory pointer contained in this region.
        while (pdptr < pend) {
          r = region_lookup(local_ctx, *pdptr);
          if (!r) {
            mqx_print(WARN,
                      "Cannot find region for dptr "
                      "%p (%d)",
                      *pdptr, i);
            pdptr++;
            continue;
          }
          if (!is_included((const void **)rgns, nrgns, (const void *)r)) {
            mqx_print(DEBUG,
                      "\tNew referenced region"
                      "(%p %p %ld %d %d) off(%lu)",
                      r, r->swp_addr, r->size, r->flags, r->state, kargs[i].dptr.off);
            rgns[nrgns++] = r;
            r->accadv.advice = ((kargs[i].dptr.advice & CADV_PTAINPUT) ? CADV_INPUT : 0) |
                               ((kargs[i].dptr.advice & CADV_PTAOUTPUT) ? CADV_OUTPUT : 0);
            total += r->size;
          } else {
            mqx_print(DEBUG,
                      "\tExisting referenced region"
                      "(%p %p %ld %d %d) off(%lu)",
                      r, r->swp_addr, r->size, r->flags, r->state, kargs[i].dptr.off);
            r->accadv.advice |= ((kargs[i].dptr.advice & CADV_PTAINPUT) ? CADV_INPUT : 0) |
                                ((kargs[i].dptr.advice & CADV_PTAOUTPUT) ? CADV_OUTPUT : 0);
          }
          pdptr++;
        } // For each dptr
      }   // If FLAG_PTARRAY
    }
  }

  *pnrgns = nrgns;
  if (nrgns > 0)
    *prgns = rgns;
  else {
    __libc_free(rgns);
    *prgns = NULL;
  }
  return total;
}

// It is possible that multiple device pointers fall into
// the same region. So we first need to get the list of
// unique regions referenced by the kernel being launched.
// Then, load all referenced regions. At any time, only one
// context is allowed to load regions for kernel launch.
// This is to avoid deadlocks/livelocks caused by memory
// contentions from simultaneous kernel launches.
// TODO: Kernel scheduling.
cudaError_t mqx_cudaLaunch(const char *entry) {
  cudaError_t ret = cudaSuccess;
  struct region **rgns = NULL;
  int nrgns = 0;
  long total_devmem_required = 0;
  int i, ldret;

  mqx_print(DEBUG, "cudaLaunch");

  // NOTE: It is possible that nrgns == 0 when regions_referenced
  // returns. Consider a kernel that only uses registers for example.
  total_devmem_required = regions_referenced(&rgns, &nrgns);
  if (total_devmem_required < 0) {
    mqx_print(ERROR, "cudaLaunch: Failed to get referenced regions.");
    ret = cudaErrorUnknown;
    goto finish;
  } else if (total_devmem_required > dev_mem_size()) {
    mqx_print(ERROR, "cudaLaunch: Kernel requires too much memory (%ld).", total_devmem_required);
    ret = cudaErrorInvalidConfiguration;
    goto finish;
  }

reload:
  stats_time_begin();
  launch_wait();
  stats_time_end(&local_ctx->stats, time_sync_attach);
  stats_time_begin();
  ldret = mqx_attach(rgns, nrgns);
  stats_time_end(&local_ctx->stats, time_attach);
  launch_signal();
  if (ldret > 0) { // retry later
    sched_yield();
    goto reload;
  } else if (ldret < 0) { // fatal load error, quit launching
    mqx_print(ERROR, "cudaLaunch: attach failed (%d)", ldret);
    ret = cudaErrorUnknown;
    goto finish;
  }

  // Transfer region data to device memory if they contain kernel INPUT,
  // and handle OUTPUT advice. Note that, by this moment, all regions
  // pointed by each dptr array has been attached and pinned to
  // device memory.
  // TODO: What if the launch below failed? Partial modification?
  stats_time_begin();
  ldret = mqx_load(rgns, nrgns);
  stats_time_end(&local_ctx->stats, time_load);
  if (ldret < 0) {
    mqx_print(ERROR, "cudaLaunch: load failed (%d)", ldret);
    for (i = 0; i < nrgns; i++)
      region_unpin(rgns[i]);
    ret = cudaErrorUnknown;
    goto finish;
  }

  // Configure and push all kernel arguments.
  if (nv_cudaConfigureCall(grid, block, shared, stream_issue) != cudaSuccess) {
    mqx_print(ERROR, "cudaLaunch: cudaConfigureCall failed");
    for (i = 0; i < nrgns; i++)
      region_unpin(rgns[i]);
    ret = cudaErrorUnknown;
    goto finish;
  }
  for (i = 0; i < nargs; i++) {
    if (kargs[i].is_dptr) {
      kargs[i].dptr.dptr = kargs[i].dptr.r->dev_addr + kargs[i].dptr.off;
      nv_cudaSetupArgument(&kargs[i].dptr.dptr, kargs[i].size, kargs[i].argoff);
    } else
      nv_cudaSetupArgument(kargs[i].ndptr, kargs[i].size, kargs[i].argoff);
  }

  // Now we can launch the kernel.
  if (mqx_launch(entry, rgns, nrgns) < 0) {
    for (i = 0; i < nrgns; i++)
      region_unpin(rgns[i]);
    ret = cudaErrorUnknown;
  }
  stats_inc(&local_ctx->stats, count_kernels, 1);

finish:
  if (rgns)
    __libc_free(rgns);
  func_index = -1;
  nadvices = 0;
  nargs = 0;
  ktop = (void *)kstack;
  return ret;
}

static void mqx_free_check_cow(struct region *r) {
  if (r->flags & FLAG_COW) {
    void *begin_aligned = (void *)PAGE_ALIGN_UP(r->cow_src);
    void *end_aligned = (void *)PAGE_ALIGN_DOWN(r->cow_src + r->size);
    if (del_cow_range(begin_aligned, (unsigned long)(end_aligned - begin_aligned), r) == -1) {
      mqx_print(FATAL, "Failed to recover protection for COW region.");
    }
    r->flags &= ~FLAG_COW;
    r->cow_src = NULL;
  }
}

// The return value of this function indicates whether the region
// was released immediately: 0 - not released yet; 1 - released.
static int mqx_free(struct region *r) {
  mqx_print(DEBUG, "Freeing region: r(%p) swp(%p) size(%ld) flags(%d) state(%d).",
            r, r->swp_addr, r->size, r->flags, r->state);

// Properly inspect/set region state.
set_state:
  acquire(&r->lock);
  switch (r->state) {
  case STATE_ATTACHED:
    if (!region_pinned(r))
      attach_list_del(r);
    else {
      release(&r->lock);
      // This region is to be freed, so its cost is zero.
      region_set_cost_evict(r, 0);
      sched_yield();
      goto set_state;
    }
    break;
  case STATE_EVICTING:
    mqx_free_check_cow(r);
    // Inform the evictor that this region is being freed.
    r->state = STATE_FREEING;
    release(&r->lock);
    mqx_print(DEBUG, "Region set to FREEING.");
    return 0;
  case STATE_FREEING:
    // The evictor has not seen STATE_FREEING yet.
    release(&r->lock);
    sched_yield();
    goto set_state;
  default: // STATE_DETACHED or STATE_ZOMBIE
    break;
  }
  release(&r->lock);

  mqx_free_check_cow(r);
  local_alloc_list_del(r);
  if (r->blocks)
    __libc_free(r->blocks);
  if (r->pta_addr)
    __libc_free(r->pta_addr);
  if (r->swp_addr)
    __libc_free(r->swp_addr);
  if (r->dev_addr) {
    nv_cudaFree(r->dev_addr);
    atomic_subl(&local_ctx->size_attached, size_cal(r->size));
    inc_attached_size(-size_cal(r->size));
    inc_detachable_size(-size_cal(r->size));
  }
  __libc_free(r);
  mqx_print(DEBUG, "Region freed.");
  return 1;
}

static int mqx_memcpy_dtoh(void *dst, const void *src, unsigned long size) {
  struct dma_channel *chan = &local_ctx->dma_dtoh;
  unsigned long off_dtos, // Device to staging buffer
      off_stou,           // Staging buffer to user buffer
      delta;
  int ret = 0, ibuf_old;

  dma_begin(chan);

  // First issue DtoH commands for all staging buffers
  ibuf_old = chan->ibuf;
  off_dtos = 0;
  while (off_dtos < size && off_dtos < DMA_NBUFS * DMA_BUFSIZE) {
    delta = min(off_dtos + DMA_BUFSIZE, size) - off_dtos;
    if (nv_cudaMemcpyAsync(chan->stage_bufs[chan->ibuf], src + off_dtos, delta, cudaMemcpyDeviceToHost, chan->stream) !=
        cudaSuccess) {
      mqx_print(FATAL, "nv_cudaMemcpyAsync failed in dtoh");
      ret = -1;
      goto finish;
    }
    if (cudaEventRecord(chan->events[chan->ibuf], chan->stream) != cudaSuccess) {
      mqx_print(FATAL, "cudaEventRecord failed in dtoh");
      ret = -1;
      goto finish;
    }
    chan->ibuf = (chan->ibuf + 1) % DMA_NBUFS;
    off_dtos += delta;
  }

  // Now copy data to user buffer, meanwhile issuing the
  // rest DtoH commands if any.
  chan->ibuf = ibuf_old;
  off_stou = 0;
  while (off_stou < size) {
    if (cudaEventSynchronize(chan->events[chan->ibuf]) != cudaSuccess) {
      mqx_print(FATAL, "cudaEventSynchronize failed in dtoh");
      ret = -1;
      goto finish;
    }
    delta = min(off_stou + DMA_BUFSIZE, size) - off_stou;
    memcpy(dst + off_stou, chan->stage_bufs[chan->ibuf], delta);
    off_stou += delta;

    if (off_dtos < size) {
      delta = min(off_dtos + DMA_BUFSIZE, size) - off_dtos;
      if (nv_cudaMemcpyAsync(chan->stage_bufs[chan->ibuf], src + off_dtos, delta, cudaMemcpyDeviceToHost,
                             chan->stream) != cudaSuccess) {
        mqx_print(FATAL, "nv_cudaMemcpyAsync failed in dtoh");
        ret = -1;
        goto finish;
      }
      if (cudaEventRecord(chan->events[chan->ibuf], chan->stream) != cudaSuccess) {
        mqx_print(FATAL, "cudaEventRecord failed in dtoh");
        ret = -1;
        goto finish;
      }
      off_dtos += delta;
    }
    chan->ibuf = (chan->ibuf + 1) % DMA_NBUFS;
  }

finish:
  dma_end(chan);
  return ret;
}

static int mqx_memcpy_htod(void *dst, const void *src, unsigned long size) {
  struct dma_channel *chan = &local_ctx->dma_htod;
  unsigned long off, delta;
  int ret = 0, ilast;

  dma_begin(chan);

  off = 0;
  while (off < size) {
    if (cudaEventSynchronize(chan->events[chan->ibuf]) != cudaSuccess) {
      mqx_print(FATAL, "cudaEventSynchronize failed in htod");
      ret = -1;
      goto finish;
    }

    delta = min(off + DMA_BUFSIZE, size) - off;
    memcpy(chan->stage_bufs[chan->ibuf], src + off, delta);
    if (nv_cudaMemcpyAsync(dst + off, chan->stage_bufs[chan->ibuf], delta, cudaMemcpyHostToDevice, chan->stream) !=
        cudaSuccess) {
      mqx_print(FATAL, "nv_cudaMemcpyAsync failed in htod");
      ret = -1;
      goto finish;
    }
    if (cudaEventRecord(chan->events[chan->ibuf], chan->stream) != cudaSuccess) {
      mqx_print(FATAL, "cudaEventRecord failed in htod");
      ret = -1;
      goto finish;
    }

    ilast = chan->ibuf; // Remember the last event
    chan->ibuf = (chan->ibuf + 1) % DMA_NBUFS;
    off += delta;
  }

  // Syncing the last HtoD command
  if (cudaEventSynchronize(chan->events[ilast]) != cudaSuccess) {
    mqx_print(FATAL, "cudaEventSynchronize failed in htod");
    ret = -1;
  }

finish:
  dma_end(chan);
  return ret;
}

// Sync the host and device copies of a data block.
// The direction of sync is determined by current validity flags.
// Data are synced from the place where data is valid to the place
// where data is invalid.
// NOTE: Block syncing has to ensure data consistency: a block being used as
// OUTPUT in a kernel cannot be synced from host to device or vice versa until
// the kernel finishes; a block being used as INPUT cannot be synced from host
// to device until the kernel finishes.
static int block_sync(struct region *r, int block) {
  int dvalid = r->blocks[block].dev_valid;
  int svalid = r->blocks[block].swp_valid;
  unsigned long off, size;
  int ret = 0;

  // Nothing to sync if both copies are valid or invalid
  if (!(dvalid ^ svalid))
    return 0;
  if (!r->dev_addr || !r->swp_addr)
    panic("block_sync: null pointer");
  mqx_print(DEBUG, "block sync begins: r(%p) block(%d) svalid(%d) dvalid(%d)", r, block, svalid, dvalid);

  stats_time_begin();
  while (atomic_read(&r->c_output) > 0)
    ;
  stats_time_end(&local_ctx->stats, time_sync_rw);

  off = block * BLOCKSIZE;
  size = min(off + BLOCKSIZE, r->size) - off;
  if (dvalid && !svalid) {
    stats_time_begin();
    ret = mqx_memcpy_dtoh(r->swp_addr + off, r->dev_addr + off, size);
    if (ret == 0)
      r->blocks[block].swp_valid = 1;
    stats_time_end(&local_ctx->stats, time_d2s);
    stats_inc(&local_ctx->stats, bytes_d2s, size);
  } else { // !dvalid && svalid
    stats_time_begin();
    while (atomic_read(&r->c_input) > 0)
      ;
    stats_time_end(&local_ctx->stats, time_sync_rw);

    stats_time_begin();
    ret = mqx_memcpy_htod(r->dev_addr + off, r->swp_addr + off, size);
    if (ret == 0)
      r->blocks[block].dev_valid = 1;
    stats_time_end(&local_ctx->stats, time_s2d);
    stats_inc(&local_ctx->stats, bytes_s2d, size);
  }

  mqx_print(DEBUG, "block sync ends");
  return ret;
}

// Copy a block of data from user source buffer to device memory region.
// Overwriting a whole block is different from modifying a block partially.
// The former can be handled by invalidating the dev copy of the block
// and setting the swp copy valid; the later requires a sync of the dev
// copy to the swp, if the dev has a newer copy, before the data can be
// written to the swp.
static int mqx_htod_block(struct region *r, unsigned long offset, const void *src, unsigned long size, int block_idx,
                          int skip_if_wait, char *skipped) {
  int partial_modification = (offset % BLOCKSIZE) || (size < BLOCKSIZE && (offset + size) < r->size);
  struct block *b = r->blocks + block_idx;
  int ret = 0;

  // The offset within the region must match the block index number.
  CHECK(BLOCKIDX(offset) == block_idx, "htod_block: offset does not match block index");

  // Acquire the block lock.
  stats_time_begin();
  while (!try_acquire(&b->lock)) {
    if (skip_if_wait) {
      if (skipped)
        *skipped = 1;
      return 0;
    }
  }
  stats_time_end(&local_ctx->stats, time_sync_block);

  // Lock acquired; now do the copy.
  if (b->swp_valid || !b->dev_valid) {
    // In this case, we do not need to update the swap buffer copy
    // of the data block with its device memory copy, because either
    // the swap buffer already contains a valid data copy or the data
    // in both the swap buffer and device memory are invalid. We can
    // simply invalidate the device memory copy and move the data from
    // the source buffer to the swap buffer.
    if (!b->swp_valid)
      b->swp_valid = 1;
    if (b->dev_valid)
      b->dev_valid = 0;
    release(&b->lock);
    // TODO: This is not thread-safe. Do memcpy before releasing the
    // the lock if multiple threads within the same process may do
    // htod on the same block simultaneously.
    stats_time_begin();
    memcpy(r->swp_addr + offset, src, size);
    stats_time_end(&local_ctx->stats, time_u2s);
    stats_inc(&local_ctx->stats, bytes_u2s, size);
  } else { // dev_valid == 1 && swp_valid == 0
    // In this case, we need to update the swap buffer copy of the
    // data block if the modification to the block is partial.
    // Otherwise, we may lose the valid part of data on the device
    // memory that is not overwritten by this memory copy.
    if (partial_modification) {
      // We don't need to pin the block to device memory because we
      // are holding the lock of a swp_valid=0,dev_valid=1 block,
      // which will prevent the evictor, if any, from freeing the
      // device memory under us.
      // NOTE: b->swp_valid is set within block_sync.
      ret = block_sync(r, block_idx);
      if (ret == 0)
        b->dev_valid = 0;
      release(&b->lock);
      if (ret != 0)
        goto finish;
    } else {
      b->swp_valid = 1;
      b->dev_valid = 0;
      release(&b->lock);
    }
    // TODO: Not thread-safe.
    stats_time_begin();
    memcpy(r->swp_addr + offset, src, size);
    stats_time_end(&local_ctx->stats, time_u2s);
    stats_inc(&local_ctx->stats, bytes_u2s, size);
  }

finish:
  if (skipped)
    *skipped = 0;
  return ret;
}

static int mqx_do_htod(struct region *r, void *dst, const void *src, size_t count) {
  unsigned long off, end, size;
  int ifirst, ilast, iblock;
  void *s = (void *)src;
  char *skipped = NULL;
  int ret = 0;

  mqx_print(DEBUG, "do_htod: %p %p %p %ld", r, dst, src, count);

  off = (unsigned long)(dst - r->swp_addr);
  end = off + count;
  ifirst = BLOCKIDX(off);
  ilast = BLOCKIDX(end - 1);
  skipped = (char *)calloc(ilast - ifirst + 1, sizeof(char));
  if (!skipped) {
    mqx_print(FATAL, "HtoD: failed to allocate memory for skipped array: %s", strerror(errno));
    return -1;
  }

  // Copy data block by block, skipping blocks that are not available
  // for immediate operation (very likely due to being evicted).
  // skipped[] records whether each block was skipped.
  for (iblock = ifirst; iblock <= ilast; ++iblock) {
    size = min(BLOCKUP(off), end) - off;
    ret = mqx_htod_block(r, off, s, size, iblock, 1, skipped + (iblock - ifirst));
    if (ret != 0)
      goto finish;
    s += size;
    off += size;
  }
  // Then, copy the rest blocks, no skipping.
  s = (void *)src;
  off = (unsigned long)(dst - r->swp_addr);
  for (iblock = ifirst; iblock <= ilast; iblock++) {
    size = min(BLOCKUP(off), end) - off;
    if (skipped[iblock - ifirst]) {
      ret = mqx_htod_block(r, off, s, size, iblock, 0, NULL);
      if (ret != 0)
        goto finish;
    }
    s += size;
    off += size;
  }

finish:
  if (skipped)
    __libc_free(skipped);
  return ret;
}

static int mqx_htod_cow(struct region *r, void *dst, const void *src, size_t count) {
  void *begin_aligned, *end_aligned;

  CHECK(count == r->size, "Copyt-on-write currently only applies to complete overwriting.");

  mqx_print(DEBUG, "In mqx_htod_cow");

  if (r->flags & FLAG_MEMSET) {
    r->flags &= ~FLAG_MEMSET;
    r->memset_value = 0;
  } else if (r->flags & FLAG_COW) {
    CHECK(r->cow_src, "COW source address cannot be null.");
    // Clear COW protection for previous COW source region.
    begin_aligned = (void *)PAGE_ALIGN_UP(r->cow_src);
    end_aligned = (void *)PAGE_ALIGN_DOWN(r->cow_src + r->size);
    CHECK(begin_aligned < end_aligned, "Incorrect cow source address");
    if (del_cow_range(begin_aligned, (unsigned long)(end_aligned - begin_aligned), r) == -1) {
      mqx_print(FATAL, "Cannot recover protection for cow source region [%p, %p)", begin_aligned, end_aligned);
      return -1;
    }
    r->flags &= ~FLAG_COW;
    r->cow_src = NULL;
  }

  begin_aligned = (void *)PAGE_ALIGN_UP(src);
  end_aligned = (void *)PAGE_ALIGN_DOWN(src + count);
  if (begin_aligned >= end_aligned) {
    // There are no whole pages between src and src+count,
    // so NO need to do cow.
    return mqx_do_htod(r, dst, src, count);
  }

  // Set COW range from begin_aligned to end_aligned.
  r->flags |= FLAG_COW;
  r->cow_src = (void *)src;
  if (add_cow_range(begin_aligned, (unsigned long)(end_aligned - begin_aligned), r) == -1) {
    mqx_print(FATAL, "Cannot set protection for cow source region [%p, %p)", begin_aligned, end_aligned);
    r->flags &= ~FLAG_COW;
    r->cow_src = NULL;
    return -1;
  }

  // Copy data from the head and tail corners to swap buffer.
  mk_region_inval(r, 0 /* dev */);
  mk_region_inval(r, 1 /* swp */);
  if (src < begin_aligned) {
    mqx_do_htod(r, r->swp_addr, src, (unsigned long)(begin_aligned - src));
  }
  if (end_aligned < src + count) {
    unsigned long off = (unsigned long)(end_aligned - src);
    mqx_do_htod(r, r->swp_addr + off, end_aligned, count - off);
  }

  region_update_cost_evict(r);
  return 0;
}

// Device pointer array regions are special. Their host swap buffers and
// device memory buffers are temporary, and are only meaningful right before
// and during a kernel execution. The opaque dptr values are stored in
// pta_addr, and can be modified only by the host program.
static int mqx_htod_pta(struct region *r, void *dst, const void *src, size_t count) {
  unsigned long off = (unsigned long)(dst - r->swp_addr);

  if (off % sizeof(void *)) {
    mqx_print(ERROR, "htod_pta: offset(%lu) not aligned", off);
    return -1;
  }
  if (count % sizeof(void *)) {
    mqx_print(ERROR, "htod_pta: count(%lu) not aligned", count);
    return -1;
  }

  stats_time_begin();
  memcpy(r->pta_addr + off, src, count);
  stats_time_end(&local_ctx->stats, time_u2s);
  stats_inc(&local_ctx->stats, bytes_u2s, count);

  return 0;
}

// Handle a host-to-device memory copy request with lazy transferring.
// TODO: Currently, we only do copy-on-write when a device memory
// region is being overwritten completely. This constraint is
// temporary and could be eliminated in the future.
static int mqx_htod(struct region *r, void *dst, const void *src, size_t count) {
  int ret;

  mqx_print(DEBUG, "htod: r(%p %p %ld %d %d) dst(%p) src(%p) count(%lu)", r, r->swp_addr, r->size, r->flags, r->state,
            dst, src, count);

  if (r->flags & FLAG_PTARRAY)
    return mqx_htod_pta(r, dst, src, count);

  if (count == r->size)
    return mqx_htod_cow(r, dst, src, count);

  if (r->flags & FLAG_MEMSET) {
    // Left part
    unsigned long off = (unsigned long)(dst - r->swp_addr);
    if (off > 0) {
      if (mqx_do_memset(r, r->swp_addr, r->memset_value, off) != 0)
        return -1;
    }
    // Right part
    off += count;
    if (off < r->size) {
      if (mqx_do_memset(r, r->swp_addr + off, r->memset_value, r->size - off) != 0)
        return -1;
    }
    r->flags &= ~FLAG_MEMSET;
    r->memset_value = 0;
  }

  if (r->flags & FLAG_COW) {
    unsigned long cow_off_begin = PAGE_ALIGN_UP(r->cow_src) - (unsigned long)(r->cow_src);
    unsigned long cow_off_end = PAGE_ALIGN_DOWN(r->cow_src + r->size) - (unsigned long)(r->cow_src);
    unsigned long dst_off_begin = (unsigned long)(dst - r->swp_addr);
    unsigned long dst_off_end = dst_off_begin + count;

    // If there is overlap between this htod and the COW region.
    if (dst_off_end > cow_off_begin && dst_off_begin < cow_off_end) {
      unsigned long overlap_off_begin = max(cow_off_begin, dst_off_begin);
      unsigned long overlap_off_end = min(cow_off_end, dst_off_end);
      if (del_cow_range(r->cow_src + cow_off_begin, cow_off_end - cow_off_begin, r) == -1) {
        mqx_print(FATAL, "Cannot recover protection for cow source region [%p, %p)",
                  r->cow_src + cow_off_begin, r->cow_src + cow_off_end);
        return -1;
      }
      // Left non-overlap
      if (cow_off_begin < overlap_off_begin) {
        mqx_do_htod(r, r->swp_addr + cow_off_begin, r->cow_src + cow_off_begin, overlap_off_begin - cow_off_begin);
      }
      // Right non-overlap
      if (overlap_off_end < cow_off_end) {
        mqx_do_htod(r, r->swp_addr + overlap_off_end, r->cow_src + overlap_off_end, cow_off_end - overlap_off_end);
      }
      r->flags &= ~FLAG_COW;
      r->cow_src = NULL;
    }
  }

  ret = mqx_do_htod(r, dst, src, count);
  region_update_cost_evict(r);
  return ret;
}

static int mqx_memset_block(struct region *r, unsigned long offset, const int value, unsigned long size, int block_idx,
                            int skip_if_wait, char *skipped) {
  int partial_modification = (offset % BLOCKSIZE) || (size < BLOCKSIZE && (offset + size) < r->size);
  struct block *b = r->blocks + block_idx;
  int ret = 0;

  CHECK(BLOCKIDX(offset) == block_idx, "htod_block: offset does not match block index");

  stats_time_begin();
  while (!try_acquire(&b->lock)) {
    if (skip_if_wait) {
      if (skipped)
        *skipped = 1;
      return 0;
    }
  }
  stats_time_end(&local_ctx->stats, time_sync_block);

  if (b->swp_valid || !b->dev_valid) {
    if (!b->swp_valid)
      b->swp_valid = 1;
    if (b->dev_valid)
      b->dev_valid = 0;
    release(&b->lock);
    // TODO: This is not thread-safe.
    memset(r->swp_addr + offset, value, size);
  } else { // dev_valid == 1 && swp_valid == 0
    if (partial_modification) {
      ret = block_sync(r, block_idx);
      if (ret == 0)
        b->dev_valid = 0;
      release(&b->lock);
      if (ret != 0)
        goto finish;
    } else {
      b->swp_valid = 1;
      b->dev_valid = 0;
      release(&b->lock);
    }
    // TODO: Not thread-safe.
    memset(r->swp_addr + offset, value, size);
  }

finish:
  if (skipped)
    *skipped = 0;
  return ret;
}

static int mqx_memset_pta(struct region *r, void *dst, int value, size_t count) {
  unsigned long off = (unsigned long)(dst - r->swp_addr);
  // We don't check alignment for memset
  memset(r->pta_addr + off, value, count);
  return 0;
}

static int mqx_memset_lazy(struct region *r, void *dst, int value, size_t count) {
  if (r->flags & FLAG_COW) {
    void *cow_begin = (void *)PAGE_ALIGN_UP(r->cow_src);
    void *cow_end = (void *)PAGE_ALIGN_DOWN(r->cow_src + r->size);
    if (del_cow_range(cow_begin, (unsigned long)(cow_end - cow_begin), r) == -1) {
      mqx_print(FATAL, "Cannot recover protection for cow region");
      return -1;
    }
    r->flags &= ~FLAG_COW;
    r->cow_src = NULL;
  }

  r->flags |= FLAG_MEMSET;
  r->memset_value = value;
  mk_region_inval(r, 0 /* dev */);
  mk_region_inval(r, 1 /* swp */);
  region_update_cost_evict(r);
  return 0;
}

static int mqx_do_memset(struct region *r, void *dst, int value, size_t count) {
  unsigned long off, end, size;
  int ifirst, ilast, iblock;
  char *skipped;
  int ret = 0;

  // Do memset eagerly.
  off = (unsigned long)(dst - r->swp_addr);
  end = off + count;
  ifirst = BLOCKIDX(off);
  ilast = BLOCKIDX(end - 1);
  skipped = (char *)__libc_malloc(ilast - ifirst + 1);
  if (!skipped) {
    mqx_print(FATAL, "malloc failed for skipped[]: %s", strerror(errno));
    return -1;
  }

  // Copy data block by block, skipping blocks that are not available
  // for immediate operation (very likely because it's being evicted).
  // skipped[] records whether each block was skipped.
  for (iblock = ifirst; iblock <= ilast; iblock++) {
    size = min(BLOCKUP(off), end) - off;
    ret = mqx_memset_block(r, off, value, size, iblock, 1, skipped + (iblock - ifirst));
    if (ret != 0)
      goto finish;
    off += size;
  }
  // Then, copy the rest blocks, no skipping.
  off = (unsigned long)(dst - r->swp_addr);
  for (iblock = ifirst; iblock <= ilast; iblock++) {
    size = min(BLOCKUP(off), end) - off;
    if (skipped[iblock - ifirst]) {
      ret = mqx_memset_block(r, off, value, size, iblock, 0, NULL);
      if (ret != 0)
        goto finish;
    }
    off += size;
  }

finish:
  __libc_free(skipped);
  return ret;
}

static int mqx_memset(struct region *r, void *dst, int value, size_t count) {
  int ret;
  mqx_print(DEBUG, "memset: r(%p %p %ld) dst(%p) value(%d) count(%lu)", r, r->swp_addr, r->size, dst, value, count);

  if (r->flags & FLAG_PTARRAY)
    return mqx_memset_pta(r, dst, value, count);

  if (count == r->size)
    return mqx_memset_lazy(r, dst, value, count);

  // If the region already has lazy memset flag, we have to finish
  // the postponed operation first before doing this one.
  if (r->flags & FLAG_MEMSET) {
    // Left part
    unsigned long off = (unsigned long)(dst - r->swp_addr);
    if (off > 0) {
      if (mqx_do_memset(r, r->swp_addr, r->memset_value, off) != 0)
        return -1;
    }
    // Right part
    off += count;
    if (off < r->size) {
      if (mqx_do_memset(r, r->swp_addr + off, r->memset_value, r->size - off) != 0)
        return -1;
    }
    r->flags &= ~FLAG_MEMSET;
    r->memset_value = 0;
  }

  if (r->flags & FLAG_COW) {
    unsigned long cow_off_begin = PAGE_ALIGN_UP(r->cow_src) - (unsigned long)(r->cow_src);
    unsigned long cow_off_end = PAGE_ALIGN_DOWN(r->cow_src + r->size) - (unsigned long)(r->cow_src);
    unsigned long dst_off_begin = (unsigned long)(dst - r->swp_addr);
    unsigned long dst_off_end = dst_off_begin + count;

    // If there is overlap between this memset and the COW region.
    if (dst_off_end > cow_off_begin && dst_off_begin < cow_off_end) {
      unsigned long overlap_off_begin = max(cow_off_begin, dst_off_begin);
      unsigned long overlap_off_end = min(cow_off_end, dst_off_end);
      if (del_cow_range(r->cow_src + cow_off_begin, cow_off_end - cow_off_begin, r) == -1) {
        mqx_print(FATAL,
                  "Cannot recover protection for cow source "
                  "region [%p, %p)",
                  r->cow_src + cow_off_begin, r->cow_src + cow_off_end);
        return -1;
      }
      // Left non-overlap
      if (cow_off_begin < overlap_off_begin) {
        mqx_do_htod(r, r->swp_addr + cow_off_begin, r->cow_src + cow_off_begin, overlap_off_begin - cow_off_begin);
      }
      // Right non-overlap
      if (overlap_off_end < cow_off_end) {
        mqx_do_htod(r, r->swp_addr + overlap_off_end, r->cow_src + overlap_off_end, cow_off_end - overlap_off_end);
      }
      r->flags &= ~FLAG_COW;
      r->cow_src = NULL;
    }
  }

  ret = mqx_do_memset(r, dst, value, count);
  region_update_cost_evict(r);
  return ret;
}

static int mqx_dtoh_block(struct region *r, void *dst, unsigned long off, unsigned long size, int block_idx,
                          int skip_if_wait, char *skipped) {
  struct block *b = r->blocks + block_idx;
  int ret = 0;

  CHECK(BLOCKIDX(off) == block_idx, "dtoh_block: offset does not match block index");

  // If the swap buffer copy of the block is valid, we can directly
  // copy the data from the swap buffer to the destination buffer.
  // We can do this without locking the block because the evictor,
  // if any, would never change the status of a block's swap buffer
  // from valid to invalid.
  // TODO: Not thread-safe if multiple threads are allowed to access
  // device memory regions simultaneously.
  if (b->swp_valid) {
    stats_time_begin();
    memcpy(dst, r->swp_addr + off, size);
    stats_time_end(&local_ctx->stats, time_s2u);
    stats_inc(&local_ctx->stats, bytes_s2u, size);
    if (skipped)
      *skipped = 0;
    return 0;
  }

  // Now we need to hold the lock to do heavier checks, because the
  // block's status may have changed.
  stats_time_begin();
  while (!try_acquire(&b->lock)) {
    if (skip_if_wait) {
      if (skipped)
        *skipped = 1;
      return 0;
    }
  }
  stats_time_end(&local_ctx->stats, time_sync_block);

  if (b->swp_valid) {
    // We can release the lock now because the evictor, if any,
    // would not change the swap buffer copy of this block's data.
    // TODO: Not thread-safe.
    release(&b->lock);
    stats_time_begin();
    memcpy(dst, r->swp_addr + off, size);
    stats_time_end(&local_ctx->stats, time_s2u);
    stats_inc(&local_ctx->stats, bytes_s2u, size);
  } else if (!b->dev_valid) {
    // Nothing to be copied because both the swap buffer and
    // device memory copies of this block are invalid.
    release(&b->lock);
  }
  // Now we know !b->swp_valid && b->dev_valid. This requires
  // first updating the swap buffer of the block with data from
  // device memory before doing the copy.
  else if (skip_if_wait) {
    // The caller does not want to wait for DMA copying.
    release(&b->lock);
    if (skipped)
      *skipped = 1;
    return 0;
  } else {
    // We don't need to pin the device memory because we are holding the
    // lock of a swp_valid=0,dev_valid=1 block, which will prevent the
    // evictor, if any, from freeing the device memory under us.
    // NOTE: The status of the block's swap buffer copy will be updated
    // within block_sync.
    // TODO: Not thread-safe.
    ret = block_sync(r, block_idx);
    release(&b->lock);
    if (ret != 0)
      goto finish;
    stats_time_begin();
    memcpy(dst, r->swp_addr + off, size);
    stats_time_end(&local_ctx->stats, time_s2u);
    stats_inc(&local_ctx->stats, bytes_s2u, size);
  }

finish:
  if (skipped)
    *skipped = 0;
  return ret;
}

static int mqx_dtoh_pta(struct region *r, void *dst, const void *src, size_t count) {
  unsigned long off = (unsigned long)(src - r->swp_addr);

  if (off % sizeof(void *)) {
    mqx_print(ERROR, "dtoh_pta: offset(%lu) not aligned", off);
    return -1;
  }
  if (count % sizeof(void *)) {
    mqx_print(ERROR, "dtoh_pta: count(%lu) not aligned", count);
    return -1;
  }

  stats_time_begin();
  memcpy(dst, r->pta_addr + off, count);
  stats_time_end(&local_ctx->stats, time_s2u);
  stats_inc(&local_ctx->stats, bytes_s2u, count);
  return 0;
}

int mqx_do_dtoh(struct region *r, void *dst, const void *src, size_t count) {
  unsigned long off = (unsigned long)(src - r->swp_addr);
  unsigned long end = off + count, size;
  int ifirst = BLOCKIDX(off), iblock;
  void *d = dst;
  char *skipped;
  int ret = 0;

  mqx_print(DEBUG, "do_dtoh: r(%p) dst(%p) src(%p) count(%lu)", r, dst, src, count);

  skipped = (char *)__libc_malloc(BLOCKIDX(end - 1) - ifirst + 1);
  if (!skipped) {
    mqx_print(ERROR, "Failed to allocate memory for skipped array: %s", strerror(errno));
    return -1;
  }

  // First, copy data from blocks whose swap buffers contain valid data.
  iblock = ifirst;
  while (off < end) {
    size = min(BLOCKUP(off), end) - off;
    ret = mqx_dtoh_block(r, d, off, size, iblock, 1, skipped + iblock - ifirst);
    if (ret != 0)
      goto finish;
    d += size;
    off += size;
    iblock++;
  }
  // Then, copy data from the rest blocks.
  off = (unsigned long)(src - r->swp_addr);
  iblock = ifirst;
  d = dst;
  while (off < end) {
    size = min(BLOCKUP(off), end) - off;
    if (skipped[iblock - ifirst]) {
      ret = mqx_dtoh_block(r, d, off, size, iblock, 0, NULL);
      if (ret != 0)
        goto finish;
    }
    d += size;
    off += size;
    iblock++;
  }

  region_update_cost_evict(r);

finish:
  __libc_free(skipped);
  return ret;
}

// TODO: It is possible to do pipelined copying, i.e., copying a block
// from swap buffer to user buffer while the next block is being fetched
// from device memory.
int mqx_dtoh(struct region *r, void *dst, const void *src, size_t count) {
  mqx_print(DEBUG, "dtoh: r(%p %p %ld %d %d) dst(%p) src(%p) count(%lu)",
            r, r->swp_addr, r->size, r->flags, r->state, dst, src, count);

  if (r->flags & FLAG_PTARRAY)
    return mqx_dtoh_pta(r, dst, src, count);

  if (r->flags & FLAG_MEMSET) {
    memset(dst, r->memset_value, count);
    return 0;
  }

  // TODO: Check whether r->src_cow overlaps with [dst, dst+count)
  if (r->flags & FLAG_COW) {
    const unsigned long cow_off_begin = PAGE_ALIGN_UP(r->cow_src) - (unsigned long)(r->cow_src);
    const unsigned long cow_off_end = PAGE_ALIGN_DOWN(r->cow_src + r->size) - (unsigned long)(r->cow_src);
    const unsigned long src_off_begin = (unsigned long)(src - r->swp_addr);
    const unsigned long src_off_end = src_off_begin + count;

    mqx_print(DEBUG, "dtoh: cow %lu %lu %lu %lu", cow_off_begin, cow_off_end, src_off_begin, src_off_end);

    // Copy data laying to the left of COW region
    unsigned long copy_off_begin = min(src_off_begin, cow_off_begin);
    unsigned long copy_off_end = min(src_off_end, cow_off_begin);
    if (copy_off_begin < copy_off_end) {
      mqx_do_dtoh(r, dst + copy_off_begin - src_off_begin, r->swp_addr + copy_off_begin, copy_off_end - copy_off_begin);
    }
    // Copy data laying within the COW region
    copy_off_begin = max(src_off_begin, cow_off_begin);
    copy_off_end = min(src_off_end, cow_off_end);
    if (copy_off_begin < copy_off_end) {
      memcpy(dst + copy_off_begin - src_off_begin, r->cow_src + copy_off_begin, copy_off_end - copy_off_begin);
    }
    // Copy data laying to the right of COW region
    copy_off_begin = max(src_off_begin, cow_off_end);
    copy_off_end = max(src_off_end, cow_off_end);
    if (copy_off_begin < copy_off_end) {
      mqx_do_dtoh(r, dst + copy_off_begin - src_off_begin, r->swp_addr + copy_off_begin, copy_off_end - copy_off_begin);
    }
    return 0;
  }

  return mqx_do_dtoh(r, dst, src, count);
}

// TODO: This is a toy implementation; data should be copied
// directly from rs to rd if possible.
static int mqx_dtod(struct region *rd, struct region *rs, void *dst, const void *src, size_t count) {
  void *temp;

  mqx_print(DEBUG,
            "DtoD: rd(%p %p %ld %d %d) rs(%p %p %ld %d %d) "
            "dst(%p) src(%p) count(%lu)",
            rd, rd->swp_addr, rd->size, rd->flags, rd->state, rs, rs->swp_addr, rs->size, rs->flags, rs->state, dst,
            src, count);

  // NOTE: This *malloc* must be malloc, not __libc_malloc, to
  // make COW work for temp when it is freed later.
  temp = malloc(count);
  if (!temp) {
    mqx_print(FATAL, "Failed to allocate temporary buffer for DtoD: %s", strerror(errno));
    return -1;
  }

  if (mqx_dtoh(rs, temp, src, count) < 0) {
    mqx_print(ERROR, "Failed to copy data to temporary DtoD buffer");
    __libc_free(temp);
    return -1;
  }
  if (mqx_htod(rd, dst, temp, count) < 0) {
    mqx_print(ERROR, "Failed to copy data from temporary DtoD buffer");
    __libc_free(temp);
    return -1;
  }

  // NOTE: This *free* must be free, not __libc_free, so that
  // temp, if set COW, can have its data transferred to dst
  // before being freed.
  free(temp);
  return 0;
}

// Searches for a live device memory region by an address that falls
// into its address range.
struct region *region_lookup(struct local_context *ctx, const void *addr) {
  unsigned long lookup_addr = (unsigned long)addr;
  unsigned long start_addr, end_addr;
  struct list_head *pos;
  struct region *r;
  int found = 0;

  acquire(&ctx->lock_alloc);
  list_for_each(pos, &ctx->list_alloc) {
    r = list_entry(pos, struct region, entry_alloc);
    // Skip regions that are being freed or zombie.
    if (r->state == STATE_FREEING || r->state == STATE_ZOMBIE)
      continue;
    start_addr = (unsigned long)(r->swp_addr);
    end_addr = start_addr + (unsigned long)(r->size);
    if (lookup_addr >= start_addr && lookup_addr < end_addr) {
      found = 1;
      break;
    }
  }
  release(&ctx->lock_alloc);

  if (!found)
    r = NULL;
  return r;
}

// Select victims for %size_needed bytes of free device memory space.
// %excls[0:%nexcl) are local regions that should not be selected.
// Put selected victims in the list %victims.
int victim_select(long size_needed, struct region **excls, int nexcl, struct list_head *victims) {
  return victim_select_cost(size_needed, excls, nexcl, victims);
}

// NOTE: When a local region is evicted, no other parties are
// supposed to be accessing the region at the same time.
// This is not true if multiple loadings happen simultaneously,
// but this region has been locked in region_load() anyway.
// A dptr array region's data never needs to be transferred back
// from device to host because swp_valid=0,dev_valid=1 will never
// happen.
long region_evict(struct region *r) {
  int nblocks = NRBLOCKS(r->size);
  long size_spared = 0;
  char *skipped;
  int i, ret = 0;

  CHECK(r->dev_addr, "dev_addr is null");
  CHECK(!region_pinned(r), "evicting a pinned region");
  mqx_print(DEBUG, "Evict: evicting region(%p) size(%ld) dv(%d) sv(%d)", r, r->size, r->blocks[0].dev_valid,
            r->blocks[0].swp_valid);

  skipped = (char *)calloc(nblocks, sizeof(char));
  if (!skipped) {
    mqx_print(FATAL, "Evict: failed to allocate memory for skipped[]: %s", strerror(errno));
    return -1;
  }

  // First round
  for (i = 0; i < nblocks; i++) {
    if (r->state == STATE_FREEING)
      goto success;
    if (try_acquire(&r->blocks[i].lock)) {
      if (!r->blocks[i].swp_valid) {
        ret = block_sync(r, i);
      }
      release(&r->blocks[i].lock);
      if (ret != 0)
        goto finish; // This is problematic if r is freeing
      skipped[i] = 0;
    } else
      skipped[i] = 1;
  }

  // Second round
  for (i = 0; i < nblocks; i++) {
    if (r->state == STATE_FREEING)
      goto success;
    if (skipped[i]) {
      acquire(&r->blocks[i].lock);
      if (!r->blocks[i].swp_valid) {
        ret = block_sync(r, i);
      }
      release(&r->blocks[i].lock);
      if (ret != 0)
        goto finish; // This is problematic if r is freeing
    }
  }

success:
  attach_list_del(r);
  if (r->dev_addr) {
    nv_cudaFree(r->dev_addr);
    r->dev_addr = NULL;
    size_spared = r->size;
  }
  atomic_subl(&local_ctx->size_attached, size_cal(r->size));
  inc_attached_size(-size_cal(r->size));
  inc_detachable_size(-size_cal(r->size));
  mk_region_inval(r, 0 /* dev */);
  acquire(&r->lock);
  if (r->state == STATE_FREEING) {
    if (r->swp_addr) {
      __libc_free(r->swp_addr);
      r->swp_addr = NULL;
    }
    r->state = STATE_ZOMBIE;
  } else
    r->state = STATE_DETACHED;
  release(&r->lock);
  mqx_print(DEBUG, "Evict: region evicted");

finish:
  __libc_free(skipped);
  return (ret == 0) ? size_spared : -1;
}

// NOTE: Client %client should have been pinned when this function
// is called.
long remote_victim_evict(int client, void *addr, long size_needed) {
  long size_spared;

  mqx_print(DEBUG, "Evict: remote eviction in client %d", client);
  size_spared = msq_send_req_evict(client, addr, size_needed, 1);
  mqx_print(DEBUG, "Evict: remote eviction returned: %ld", size_spared);
  client_unpin(client);

  return size_spared;
}

// Similar to mqx_evict, but only select at most one victim from local
// region list, even if it is smaller than required, evict it, and return.
long local_victim_evict(void *addr, long size_needed) {
  struct region *r, *rv = (struct region *)addr;
  struct list_head *pos;
  int found = 0;
  long ret = 0;

  // Check and evict the designated region
  acquire(&local_ctx->lock_attach);
  list_for_each(pos, &local_ctx->list_attach) {
    r = list_entry(pos, struct region, entry_attach);
    if (r == rv) {
      if (try_acquire(&r->lock)) {
        if (r->state == STATE_ATTACHED && !region_pinned(r)) {
          r->state = STATE_EVICTING;
          release(&r->lock);
          found = 1;
        } else
          release(&r->lock);
      }
      break;
    }
  }
  release(&local_ctx->lock_attach);

  // NOTE: It is possible that the selected victim is not exactly
  // what the evictor wants.
  if (found)
    ret = region_evict(rv);

  return ret;
}

// Evict a victim.
// %victim may point to a local region or a remote client that
// may own some evictable region.
// The return value is the size of free space spared during
// this eviction. -1 means error.
long victim_evict(struct victim *victim, long size_needed) {
  long ret;
  mqx_print(DEBUG, "Evict: evicting %s (%p)", victim->r ? "local" : "remote", victim->r ? (victim->r) : (victim->addr));

  if (victim->r)
    ret = region_evict(victim->r);
  else if (victim->client != -1)
    ret = remote_victim_evict(victim->client, victim->addr, size_needed);
  else {
    panic("Victim is neither local nor remote");
    ret = -1;
  }

  return ret;
}

// Evict some device memory so that the size of free space can
// satisfy %size_needed. Regions in %excls[0:%nexcl) should not
// be selected for eviction.
static int mqx_evict(long size_needed, struct region **excls, int nexcl) {
  struct list_head victims, *e;
  struct victim *v;
  long size_spared;
  int ret = 0;

  mqx_print(DEBUG, "Evict: evicting for %ld bytes", size_needed);
  INIT_LIST_HEAD(&victims);
  stats_inc(&local_ctx->stats, bytes_evict_needed, max(size_needed - dev_mem_free(), 0));

  do {
    ret = victim_select(size_needed, excls, nexcl, &victims);
    if (ret != 0)
      return ret;
    for (e = victims.next; e != (&victims);) {
      v = list_entry(e, struct victim, entry);
      if ((size_spared = victim_evict(v, size_needed)) < 0) {
        ret = -1;
        goto fail_evict;
      }
      stats_inc(&local_ctx->stats, bytes_evict_space, size_spared);
      stats_inc(&local_ctx->stats, count_evict_victims, 1);
      list_del(e);
      e = e->next;
      __libc_free(v);
    }
  } while (dev_mem_free() < size_needed);

  mqx_print(DEBUG, "Evict: eviction finished");
  return 0;

fail_evict:
  stats_inc(&local_ctx->stats, count_evict_fail, 1);
  for (e = victims.next; e != (&victims);) {
    v = list_entry(e, struct victim, entry);
    if (v->r) {
      acquire(&v->r->lock);
      if (v->r->state != STATE_FREEING)
        v->r->state = STATE_ATTACHED;
      release(&v->r->lock);
    } else if (v->client != -1)
      client_unpin(v->client);
    list_del(e);
    e = e->next;
    __libc_free(v);
  }

  mqx_print(DEBUG, "Evict: eviction failed");
  return ret;
}

// Allocate device memory to a region (i.e., attach).
static int region_attach(struct region *r, int pin, struct region **excls, int nexcl) {
  cudaError_t cret;
  int ret;

  mqx_print(DEBUG, "Attach: attaching%s region %p", (r->flags & FLAG_COW) ? " cow" : "", r);

  if (r->state == STATE_EVICTING) {
    mqx_print(ERROR, "Attach: should not see an evicting region");
    return -1;
  }
  if (r->state == STATE_ATTACHED) {
    mqx_print(DEBUG, "Attach: region already attached");
    if (pin)
      region_pin(r);
    // Update the region's position in the LRU list.
    attach_list_mov(r);
    return 0;
  }
  if (r->state != STATE_DETACHED) {
    mqx_print(ERROR, "Attach: attaching a non-detached region");
    return -1;
  }

  // Attach if current free memory space is larger than region size.
  if (size_cal(r->size) <= dev_mem_free()) {
    if ((cret = nv_cudaMalloc(&r->dev_addr, r->size)) == cudaSuccess)
      goto attach_success;
    else {
      mqx_print(DEBUG, "Attach: nv_cudaMalloc failed: %s (%d)", cudaGetErrorString(cret), cret);
      if (cret == cudaErrorLaunchFailure)
        return -1;
    }
  }

  // Evict some device memory.
  stats_time_begin();
  ret = mqx_evict(size_cal(r->size), excls, nexcl);
  stats_time_end(&local_ctx->stats, time_evict);
  if (ret < 0 || (ret > 0 && dev_mem_free() < size_cal(r->size)))
    return ret;

  // Try to attach again.
  if ((cret = nv_cudaMalloc(&r->dev_addr, r->size)) != cudaSuccess) {
    r->dev_addr = NULL;
    mqx_print(DEBUG, "nv_cudaMalloc failed: %s (%d)", cudaGetErrorString(cret), cret);
    if (cret == cudaErrorLaunchFailure)
      return -1;
    else
      return 1;
  }

attach_success:
  atomic_addl(&local_ctx->size_attached, size_cal(r->size));
  inc_attached_size(size_cal(r->size));
  inc_detachable_size(size_cal(r->size));
  if (pin)
    region_pin(r);
  // Reassure that the dev copies of all blocks are set to invalid.
  mk_region_inval(r, 0 /* dev */);
  r->state = STATE_ATTACHED;
  attach_list_add(r);

  mqx_print(DEBUG, "region attached");
  return 0;
}

static int region_load_cow(struct region *r) {
  unsigned long cow_off_begin = PAGE_ALIGN_UP(r->cow_src) - (unsigned long)(r->cow_src);
  unsigned long cow_off_end = PAGE_ALIGN_DOWN(r->cow_src + r->size) - (unsigned long)(r->cow_src);

  if (cow_off_begin > 0) {
    if (mqx_memcpy_htod(r->dev_addr, r->swp_addr, cow_off_begin) < 0) {
      mqx_print(ERROR, "Load failed");
      return -1;
    }
  }

  if (mqx_memcpy_htod(r->dev_addr + cow_off_begin, r->cow_src + cow_off_begin, cow_off_end - cow_off_begin) < 0) {
    mqx_print(ERROR, "Load failed");
    return -1;
  }

  if (cow_off_end < r->size) {
    if (mqx_memcpy_htod(r->dev_addr + cow_off_end, r->swp_addr + cow_off_end, r->size - cow_off_end) < 0) {
      mqx_print(ERROR, "Load failed");
      return -1;
    }
  }

  if (del_cow_range(r->cow_src + cow_off_begin, cow_off_end - cow_off_begin, r) == -1) {
    mqx_print(FATAL, "Cannot recover protection after cow region loading");
  }
  r->flags &= ~FLAG_COW;
  r->cow_src = NULL;
  mk_region_valid(r, 0 /* dev */);
  mk_region_inval(r, 1 /* swp */);
  return 0;
}

static int region_load_memset(struct region *r) {
  // Memset in kernel stream ensures correct ordering with the kernel
  // referencing the region. So no explicit sync required.
  if (nv_cudaMemsetAsync(r->dev_addr, r->memset_value, r->size, local_ctx->stream_kernel) != cudaSuccess) {
    mqx_print(ERROR, "Load failed");
    return -1;
  }
  r->flags &= ~FLAG_MEMSET;
  r->memset_value = 0;
  mk_region_valid(r, 0 /* dev */);
  return 0;
}

static int region_load_pta(struct region *r) {
  void **pdptr = (void **)(r->pta_addr);
  void **pend = (void **)(r->pta_addr + r->size);
  unsigned long off = 0;
  int i, ret;

  mqx_print(DEBUG, "Loading pta region %p", r);

  while (pdptr < pend) {
    struct region *tmp = region_lookup(local_ctx, *pdptr);
    if (!tmp) {
      mqx_print(WARN, "Cannot find region for dptr %p", *pdptr);
      off += sizeof(void *);
      pdptr++;
      continue;
    }
    *(void **)(r->swp_addr + off) = tmp->dev_addr + (unsigned long)(*pdptr - tmp->swp_addr);
    off += sizeof(void *);
    pdptr++;
  }

  mk_region_valid(r, 1 /* swp */);
  mk_region_inval(r, 0 /* dev */);
  for (i = 0; i < NRBLOCKS(r->size); i++) {
    ret = block_sync(r, i);
    if (ret != 0) {
      mqx_print(ERROR, "Load failed");
      return -1;
    }
  }

  mqx_print(DEBUG, "Region loaded");
  return 0;
}

// Load the data of a region to device memory.
static int region_load(struct region *r) {
  int i, ret = 0;

  if (r->flags & FLAG_COW)
    ret = region_load_cow(r);
  else if (r->flags & FLAG_MEMSET)
    ret = region_load_memset(r);
  else if (r->flags & FLAG_PTARRAY)
    ret = region_load_pta(r);
  else {
    mqx_print(DEBUG, "Loading region %p", r);
    for (i = 0; i < NRBLOCKS(r->size); i++) {
      acquire(&r->blocks[i].lock);
      if (!r->blocks[i].dev_valid)
        ret = block_sync(r, i);
      release(&r->blocks[i].lock);
      if (ret != 0) {
        mqx_print(ERROR, "Load failed");
        goto finish;
      }
    }
    mqx_print(DEBUG, "Region loaded");
  }

finish:
  return ret;
}

// Attach all %n regions specified by %rgns to device.
// Every successfully attached region is also pinned to device.
// If all regions cannot be attached successfully, successfully
// attached regions will be unpinned so that they can be replaced
// by other kernel launches.
// Return value: 0 - success; < 0 - fatal failure; > 0 - retry later.
static int mqx_attach(struct region **rgns, int n) {
  char *pinned;
  int i, ret;

  if (n == 0)
    return 0;
  if (n < 0 || (n > 0 && !rgns))
    return -1;
  mqx_print(DEBUG, "mqx_attach begins: %d regions to attach", n);

  pinned = (char *)calloc(n, sizeof(char));
  if (!pinned) {
    mqx_print(FATAL, "Attach: malloc failed for pinned array: %s", strerror(errno));
    return -1;
  }

  for (i = 0; i < n; i++) {
    if (rgns[i]->state == STATE_FREEING || rgns[i]->state == STATE_ZOMBIE) {
      mqx_print(ERROR, "Attach: cannot attach a released region r(%p %p %ld %d %d)",
                rgns[i], rgns[i]->swp_addr, rgns[i]->size, rgns[i]->flags, rgns[i]->state);
      ret = -1;
      goto fail;
    }
    // NOTE: In current design, this locking is redundant
    acquire(&rgns[i]->lock);
    ret = region_attach(rgns[i], 1, rgns, n);
    release(&rgns[i]->lock);
    if (ret != 0)
      goto fail;
    pinned[i] = 1;
  }

  mqx_print(DEBUG, "mqx_attach succeeded");
  __libc_free(pinned);
  return 0;

fail:
  stats_inc(&local_ctx->stats, count_attach_fail, 1);
  for (i = 0; i < n; i++)
    if (pinned[i])
      region_unpin(rgns[i]);
  __libc_free(pinned);
  mqx_print(DEBUG, "mqx_attach failed");
  return ret;
}

// Load region data to device memory. All regions should have been
// allocated with device memory and pinned when this function is called.
static int mqx_load(struct region **rgns, int nrgns) {
  int i, ret;

  for (i = 0; i < nrgns; i++) {
    struct region *r = rgns[i];
    if (r->accadv.advice & CADV_INPUT) {
      ret = region_load(r);
      if (ret != 0)
        return -1;
    }
    if (r->accadv.advice & CADV_OUTPUT) {
      if (r->flags & FLAG_MEMSET) {
        ret = region_load_memset(r);
        if (ret != 0)
          return -1;
      }
      mk_region_inval(r, 1);
      mk_region_valid(r, 0);
      if (r->flags & FLAG_MEMSET) {
        r->flags &= ~FLAG_MEMSET;
        r->memset_value = 0;
      } else if (r->flags & FLAG_COW) {
        void *cow_begin = (void *)PAGE_ALIGN_UP(r->cow_src);
        unsigned long cow_bytes = PAGE_ALIGN_DOWN(r->cow_src + r->size) - (unsigned long)cow_begin;
        if (del_cow_range(cow_begin, cow_bytes, r) == -1) {
          mqx_print(ERROR, "Failed to recover protection");
          return -1;
        }
        r->flags &= ~FLAG_COW;
        r->cow_src = NULL;
      }
    }
    region_update_cost_evict(r);
    r->freq++;
    region_inc_freq(r);
  }

  return 0;
}

// The callback function invoked by CUDA after each kernel finishes
// execution. Keep it as short as possible because it blocks GPU
// hardware command queue.
void CUDART_CB mqx_kernel_callback(cudaStream_t stream, cudaError_t status, void *data) {
  struct kcb *pcb = (struct kcb *)data;
  int i;

  if (status != cudaSuccess)
    mqx_print(ERROR, "kernel failed: %d", status);
  else
    mqx_print(DEBUG, "kernel succeeded");

  for (i = 0; i < pcb->nrgns; i++) {
    if (pcb->flags[i] & CADV_OUTPUT)
      atomic_dec(&pcb->rgns[i]->c_output);
    if (pcb->flags[i] & CADV_INPUT)
      atomic_dec(&pcb->rgns[i]->c_input);
    region_unpin(pcb->rgns[i]);
  }
  __libc_free(pcb);
}

// Here we utilize CUDA 5+'s stream callback feature to capture kernel
// finish event and unpin related regions accordingly. This is different
// from our initial design because we want to minimize changes to user programs.
static int mqx_launch(const char *entry, struct region **rgns, int nrgns) {
  cudaError_t cret;
  struct kcb *pcb;
  int i;

  if (nrgns > MAX_ARGS) {
    mqx_print(ERROR, "too many regions");
    return -1;
  }

  pcb = (struct kcb *)__libc_malloc(sizeof(*pcb));
  if (!pcb) {
    mqx_print(FATAL, "cudaLaunch: failed to allocate memory for kcb: %s", strerror(errno));
    return -1;
  }
  if (nrgns > 0)
    memcpy(pcb->rgns, rgns, sizeof(void *) * nrgns);
  for (i = 0; i < nrgns; i++) {
    pcb->flags[i] = rgns[i]->accadv.advice;
    if (pcb->flags[i] & CADV_OUTPUT)
      atomic_inc(&rgns[i]->c_output);
    if (pcb->flags[i] & CADV_INPUT)
      atomic_inc(&rgns[i]->c_input);
  }
  pcb->nrgns = nrgns;

#ifdef MQX_CONFIG_STAT_KERNEL_TIME
  stats_time_begin();
#endif
  if ((cret = nv_cudaLaunch(entry)) != cudaSuccess) {
    for (i = 0; i < nrgns; i++) {
      if (pcb->flags[i] & CADV_OUTPUT)
        atomic_dec(&pcb->rgns[i]->c_output);
      if (pcb->flags[i] & CADV_INPUT)
        atomic_dec(&pcb->rgns[i]->c_input);
    }
    __libc_free(pcb);
    mqx_print(ERROR, "cudaLaunch: nv_cudaLaunch failed: %s (%d)", cudaGetErrorString(cret), cret);
    return -1;
  }
  nv_cudaStreamAddCallback(stream_issue, mqx_kernel_callback, (void *)pcb, 0);
#ifdef MQX_CONFIG_STAT_KERNEL_TIME
  if (nv_cudaStreamSynchronize(local_ctx->stream_kernel) != cudaSuccess) {
    mqx_print(ERROR, "stream synchronization failed in mqx_launch");
  }
  stats_time_end(&local_ctx->stats, time_kernel);
#endif

  // Update this client's position in global LRU client list
  client_mov();
  mqx_print(DEBUG, "kernel launched");
  return 0;
}

void mqx_handle_cow(void *cow_begin, void *cow_end, struct region *r) {
  CHECK(r->flags & FLAG_COW, "Handling lazy copy on a non-cow region");
  CHECK((unsigned long)cow_begin == PAGE_ALIGN_UP(r->cow_src), "");
  CHECK((unsigned long)cow_end == PAGE_ALIGN_DOWN(r->cow_src + r->size), "");
  mqx_do_htod(r, r->swp_addr + (unsigned long)(cow_begin - r->cow_src), cow_begin,
              (unsigned long)(cow_end - cow_begin));
  r->flags &= ~FLAG_COW;
  r->cow_src = NULL;
}
