/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.  This source code is a "commercial item" as
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */

#include "common.h"
#include "gpuCudaLib.h"
#include <assert.h>
#ifdef HAS_GMM
#include "gmm.h"
#endif

// Define this to more rigorously avoid bank conflicts,
// even at the lower (root) levels of the tree
// Note that due to the higher addressing overhead, performance
// is lower with ZERO_BANK_CONFLICTS enabled.  It is provided
// as an example.
//#define ZERO_BANK_CONFLICTS

// 16 banks on G80
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

#define BLOCK_SIZE 256

/*#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif*/

inline __device__ int CONFLICT_FREE_OFFSET(int index) {
  // return ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS));
  return ((index) >> LOG_NUM_BANKS);
}

template <bool isNP2>
__device__ static void loadSharedChunkFromMem(int *s_data, const int *g_idata, int n, int baseIndex, int &ai, int &bi,
                                              int &mem_ai, int &mem_bi, int &bankOffsetA, int &bankOffsetB) {
  int thid = threadIdx.x;
  mem_ai = baseIndex + threadIdx.x;
  mem_bi = mem_ai + blockDim.x;

  ai = thid;
  bi = thid + blockDim.x;

  // compute spacing to avoid bank conflicts
  bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  bankOffsetB = CONFLICT_FREE_OFFSET(bi);

  s_data[ai + bankOffsetA] = g_idata[mem_ai];

  if (isNP2) {
    s_data[bi + bankOffsetB] = (bi < n) ? g_idata[mem_bi] : 0;
  } else {
    s_data[bi + bankOffsetB] = g_idata[mem_bi];
  }
}

template <bool isNP2>
static __device__ void storeSharedChunkToMem(int *g_odata, const int *s_data, int n, int ai, int bi, int mem_ai,
                                             int mem_bi, int bankOffsetA, int bankOffsetB) {
  __syncthreads();

  g_odata[mem_ai] = s_data[ai + bankOffsetA];
  if (isNP2) {
    if (bi < n)
      g_odata[mem_bi] = s_data[bi + bankOffsetB];
  } else {
    g_odata[mem_bi] = s_data[bi + bankOffsetB];
  }
}

template <bool storeSum> static __device__ void clearLastElement(int *s_data, int *g_blockSums, int blockIndex) {
  if (threadIdx.x == 0) {
    int index = (blockDim.x << 1) - 1;
    index += CONFLICT_FREE_OFFSET(index);

    if (storeSum) // compile-time decision
    {
      // write this block's total sum to the corresponding index in the blockSums array
      g_blockSums[blockIndex] = s_data[index];
    }

    // zero the last element in the scan so it will propagate back to the front
    s_data[index] = 0;
  }
}

__device__ static unsigned int buildSum(int *s_data) {
  unsigned int thid = threadIdx.x;
  unsigned int stride = 1;

  // build the sum in place up the tree
  for (int d = blockDim.x; d > 0; d >>= 1) {
    __syncthreads();

    if (thid < d) {
      int i = __mul24(__mul24(2, stride), thid);
      int ai = i + stride - 1;
      int bi = ai + stride;

      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      s_data[bi] += s_data[ai];
    }

    stride *= 2;
  }

  return stride;
}

__device__ static void scanRootToLeaves(int *s_data, unsigned int stride) {
  unsigned int thid = threadIdx.x;

  // traverse down the tree building the scan in place
  for (int d = 1; d <= blockDim.x; d *= 2) {
    stride >>= 1;

    __syncthreads();

    if (thid < d) {
      int i = __mul24(__mul24(2, stride), thid);
      int ai = i + stride - 1;
      int bi = ai + stride;

      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      int t = s_data[ai];
      s_data[ai] = s_data[bi];
      s_data[bi] += t;
    }
  }
}

template <bool storeSum> static __device__ void prescanBlock(int *data, int blockIndex, int *blockSums) {
  int stride = buildSum(data); // build the sum in place up the tree
  clearLastElement<storeSum>(data, blockSums, (blockIndex == 0) ? blockIdx.x : blockIndex);
  scanRootToLeaves(data, stride); // traverse down tree to build the scan
}

// no shared memory
template <bool storeSum, bool isNP2>
__global__ void prescan(int *d_data, int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex,
                               int baseIndex, unsigned int sharedMemSize) {
  int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
  // extern __shared__ int s_data[];

  int bx = blockIdx.x;
  int *s_data = d_data + (sharedMemSize / sizeof(int)) * bx;

  // load data into shared memory
  loadSharedChunkFromMem<isNP2>(s_data, g_idata, n,
                                (baseIndex == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)) : baseIndex, ai, bi, mem_ai,
                                mem_bi, bankOffsetA, bankOffsetB);
  // scan the data in each block
  prescanBlock<storeSum>(s_data, blockIndex, g_blockSums);
  // write results to device memory
  storeSharedChunkToMem<isNP2>(g_odata, s_data, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);
}

// with shared memory
template <bool storeSum, bool isNP2>
__global__ void prescan(int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex,
                               int baseIndex) {
  int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
  extern __shared__ int s_data[];

  // load data into shared memory
  loadSharedChunkFromMem<isNP2>(s_data, g_idata, n,
                                (baseIndex == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)) : baseIndex, ai, bi, mem_ai,
                                mem_bi, bankOffsetA, bankOffsetB);
  // scan the data in each block
  prescanBlock<storeSum>(s_data, blockIndex, g_blockSums);
  // write results to device memory
  storeSharedChunkToMem<isNP2>(g_odata, s_data, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);
}

extern "C" __global__ void uniformAdd(int *g_data, int *uniforms, int n, int blockOffset, int baseIndex, int total) {
  __shared__ int uni;
  if (threadIdx.x == 0)
    uni = uniforms[blockIdx.x + blockOffset];

  unsigned int address = __mul24(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x;

  __syncthreads();

  // note two adds per thread
  g_data[address] += uni;
  if (address + blockDim.x < total)
    g_data[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * uni;
}

static inline bool isPowerOfTwo(int n) { return ((n & (n - 1)) == 0); }

static inline int floorPow2(int n) {
  int exp;
  frexp((int)n, &exp);
  return 1 << (exp - 1);
}

static int **g_scanBlockSums;
static unsigned int g_numEltsAllocated = 0;
static unsigned int g_numLevelsAllocated = 0;

static void preallocBlockSums(unsigned int maxNumElements) {
  assert(g_numEltsAllocated == 0);

  g_numEltsAllocated = maxNumElements;

  unsigned int blockSize = BLOCK_SIZE;
  unsigned int numElts = maxNumElements;

  int level = 0;

  do {
    unsigned int numBlocks = max(1, (int)ceil((int)numElts / (2.f * blockSize)));
    if (numBlocks > 1) {
      level++;
    }
    numElts = numBlocks;
  } while (numElts > 1);

  // printf("level = %d\n", level);

  g_scanBlockSums = (int **)malloc(level * sizeof(int *));
  g_numLevelsAllocated = level;

  numElts = maxNumElements;
  level = 0;

  do {
    unsigned int numBlocks = max(1, (int)ceil((int)numElts / (2.f * blockSize)));
    if (numBlocks > 1) {
      cudaMalloc((void **)&g_scanBlockSums[level++], numBlocks * sizeof(int));
    }
    numElts = numBlocks;
  } while (numElts > 1);
}

static void deallocBlockSums() {
  for (int i = 0; i < g_numLevelsAllocated; i++) {
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(g_scanBlockSums[i]));
  }

  free((void **)g_scanBlockSums);

  g_scanBlockSums = 0;
  g_numEltsAllocated = 0;
  g_numLevelsAllocated = 0;
}

static void prescanArrayRecursive(int *outArray, const int *inArray, int numElements, int level, struct statistic *pp) {
  unsigned int blockSize = BLOCK_SIZE;
  unsigned int numBlocks = max(1, (int)ceil((int)numElements / (2.f * blockSize)));
  unsigned int numThreads;

  if (numBlocks > 1)
    numThreads = blockSize;
  else if (isPowerOfTwo(numElements))
    numThreads = numElements / 2;
  else
    numThreads = floorPow2(numElements);

  // printf("blocks(%u) threads(%u) elements(%d)\n", numBlocks, numThreads, numElements);

  unsigned int numEltsPerBlock = numThreads * 2;

  unsigned int numEltsLastBlock = numElements - (numBlocks - 1) * numEltsPerBlock;
  unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
  unsigned int np2LastBlock = 0;
  unsigned int sharedMemLastBlock = 0;

  if (numEltsLastBlock != numEltsPerBlock) {
    np2LastBlock = 1;

    if (!isPowerOfTwo(numEltsLastBlock))
      numThreadsLastBlock = floorPow2(numEltsLastBlock);

    unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
    sharedMemLastBlock = sizeof(int) * (2 * numThreadsLastBlock + extraSpace);
  }

  unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
  unsigned int sharedMemSize = sizeof(int) * (numEltsPerBlock + extraSpace);

  dim3 grid(max(1, numBlocks - np2LastBlock), 1, 1);
  dim3 threads(numThreads, 1, 1);

  if (numBlocks > 1) {
    GMM_CALL(cudaAdvise(0, CADV_OUTPUT));
    GMM_CALL(cudaAdvise(1, CADV_INPUT));
    GMM_CALL(cudaAdvise(2, CADV_OUTPUT));
    prescan<true, false>
        <<<grid, threads, sharedMemSize>>>(outArray, inArray, g_scanBlockSums[level], numThreads * 2, 0, 0);

    if (np2LastBlock) {
      GMM_CALL(cudaAdvise(0, CADV_OUTPUT));
      GMM_CALL(cudaAdvise(1, CADV_INPUT));
      GMM_CALL(cudaAdvise(2, CADV_OUTPUT));
      prescan<true, true><<<1, numThreadsLastBlock, sharedMemLastBlock>>>(
          outArray, inArray, g_scanBlockSums[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
    }

    prescanArrayRecursive(g_scanBlockSums[level], g_scanBlockSums[level], numBlocks, level + 1, pp);

    GMM_CALL(cudaAdvise(0, CADV_OUTPUT));
    GMM_CALL(cudaAdvise(1, CADV_INPUT));
    uniformAdd<<<grid, threads>>>(outArray, g_scanBlockSums[level], numElements - numEltsLastBlock, 0, 0, numElements);

    if (np2LastBlock) {
      GMM_CALL(cudaAdvise(0, CADV_OUTPUT));
      GMM_CALL(cudaAdvise(1, CADV_INPUT));
      uniformAdd<<<1, numThreadsLastBlock>>>(outArray, g_scanBlockSums[level], numEltsLastBlock, numBlocks - 1,
                                             numElements - numEltsLastBlock, numElements);
    }
  } else if (isPowerOfTwo(numElements)) {
    GMM_CALL(cudaAdvise(0, CADV_OUTPUT));
    GMM_CALL(cudaAdvise(1, CADV_INPUT));
    prescan<false, false><<<grid, threads, sharedMemSize>>>(outArray, inArray, 0, numThreads * 2, 0, 0);
  } else {
    GMM_CALL(cudaAdvise(0, CADV_OUTPUT));
    GMM_CALL(cudaAdvise(1, CADV_INPUT));
    prescan<false, true><<<grid, threads, sharedMemSize>>>(outArray, inArray, 0, numElements, 0, 0);
  }
}

static void prescanArray(int *outArray, int *inArray, int numElements, struct statistic *pp) {
  prescanArrayRecursive(outArray, inArray, numElements, 0, pp);
}

void scanImpl(int *d_input, int rLen, int *d_output, struct statistic *pp) {
  int len = 2;
  if (rLen < len) {
    int *input, *output;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&input, len * sizeof(int)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&output, len * sizeof(int)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemset(input, 0, len * sizeof(int)));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(input, d_input, rLen * sizeof(int), cudaMemcpyDeviceToDevice));
    preallocBlockSums(len);
    prescanArray(output, input, len, pp);
    deallocBlockSums();
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_output, output, rLen * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(input));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(output));
    return;
  } else {
    preallocBlockSums(rLen);
    prescanArray(d_output, d_input, rLen, pp);
    deallocBlockSums();
  }
  //	preallocBlockSums(rLen);
  //	prescanArray(d_output, d_input, rLen, pp);
  //	deallocBlockSums();
}

