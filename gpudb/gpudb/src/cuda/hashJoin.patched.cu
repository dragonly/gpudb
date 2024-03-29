/*
   Copyright (c) 2012-2013 The Ohio State University.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "common.h"
#include "gpuCudaLib.h"
#include "hashJoin.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#ifdef HAS_GMM
#include "gmm.h"
#endif

#define CHECK_POINTER(p)                                                                                               \
  do {                                                                                                                 \
    if (p == NULL) {                                                                                                   \
      perror("Failed to allocate host memory");                                                                        \
      exit(-1);                                                                                                        \
    }                                                                                                                  \
  } while (0)

#define NP2(n)                                                                                                         \
  do {                                                                                                                 \
    n--;                                                                                                               \
    n |= n >> 1;                                                                                                       \
    n |= n >> 2;                                                                                                       \
    n |= n >> 4;                                                                                                       \
    n |= n >> 8;                                                                                                       \
    n |= n >> 16;                                                                                                      \
    n++;                                                                                                               \
  } while (0)

/*
 * Count the number of dimension keys for each bucket.
 */

extern "C" __global__ void count_hash_num_hj(char *dim, long inNum, int *num, int hsize) {
  int stride = blockDim.x * gridDim.x;
  int offset = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = offset; i < inNum; i += stride) {
    int joinKey = ((int *)dim)[i];
    int hKey = joinKey & (hsize - 1);
    atomicAdd(&(num[hKey]), 1);
  }
}

/*
 * All the buckets are stored in a continues memory region.
 * The starting position of each bucket is stored in the psum array.
 * For star schema quereis, the size of fact table is usually much
 * larger than the size of the dimension table. In this case, hash probing is much more
 * time consuming than building hash table. By avoiding pointer, the efficiency of hash probing
 * can be improved.
 */

extern "C" __global__ void build_hash_table_hj(char *dim, long inNum, int *psum, char *bucket, int hsize) {

  int stride = blockDim.x * gridDim.x;
  int offset = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = offset; i < inNum; i += stride) {
    int joinKey = ((int *)dim)[i];
    int hKey = joinKey & (hsize - 1);
    int pos = atomicAdd(&psum[hKey], 1) * 2;
    ((int *)bucket)[pos] = joinKey;
    pos += 1;
    int dimId = i + 1;
    ((int *)bucket)[pos] = dimId;
  }
}

/*
 * Count join result for each thread for dictionary encoded column.
 */

extern "C" __global__ void count_join_result_dict_hj(int *num, int *psum, char *bucket, struct dictHeader *dheader, int dNum,
                                              int *dictFilter, int hsize) {

  int stride = blockDim.x * gridDim.x;
  int offset = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = offset; i < dNum; i += stride) {
    int fkey = dheader->hash[i];
    int hkey = fkey & (hsize - 1);
    int keyNum = num[hkey];
    int fvalue = 0;

    for (int j = 0; j < keyNum; j++) {
      int pSum = psum[hkey];
      int dimKey = ((int *)(bucket))[2 * j + 2 * pSum];
      if (dimKey == fkey) {
        int dimId = ((int *)(bucket))[2 * j + 2 * pSum + 1];
        fvalue = dimId;
        break;
      }
    }

    dictFilter[i] = fvalue;
  }
}

/*
 * Transform the dictionary filter to the final filter than can be used to generate the result
 */

extern "C" __global__ void transform_dict_filter_hj(int *dictFilter, char *fact, long tupleNum, int dNum, int *filter) {

  int stride = blockDim.x * gridDim.x;
  int offset = blockIdx.x * blockDim.x + threadIdx.x;

  struct dictHeader *dheader;
  dheader = (struct dictHeader *)fact;

  int byteNum = dheader->bitNum / 8;
  int numInt = (tupleNum * byteNum + sizeof(int) - 1) / sizeof(int);

  for (long i = offset; i < numInt; i += stride) {
    int tmp = ((int *)(fact + sizeof(struct dictHeader)))[i];

    for (int j = 0; j < sizeof(int) / byteNum; j++) {
      int fkey = 0;
      memcpy(&fkey, ((char *)&tmp) + j * byteNum, byteNum);

      filter[i * sizeof(int) / byteNum + j] = dictFilter[fkey];
    }
  }
}

/*
 * count the number that is not zero in the filter
 */
extern "C" __global__ void filter_count_hj(long tupleNum, int *count, int *factFilter) {

  int lcount = 0;
  int stride = blockDim.x * gridDim.x;
  long offset = blockIdx.x * blockDim.x + threadIdx.x;

  for (long i = offset; i < tupleNum; i += stride) {
    if (factFilter[i] != 0)
      lcount++;
  }
  count[offset] = lcount;
}

/*
 * count join result for rle-compressed key.
 */

extern "C" __global__ void count_join_result_rle_hj(int *num, int *psum, char *bucket, char *fact, long tupleNum,
                                             int *factFilter, int hsize) {

  int stride = blockDim.x * gridDim.x;
  long offset = blockIdx.x * blockDim.x + threadIdx.x;

  struct rleHeader *rheader = (struct rleHeader *)fact;
  int dNum = rheader->dictNum;

  for (int i = offset; i < dNum; i += stride) {
    int fkey = ((int *)(fact + sizeof(struct rleHeader)))[i];
    int fcount = ((int *)(fact + sizeof(struct rleHeader)))[i + dNum];
    int fpos = ((int *)(fact + sizeof(struct rleHeader)))[i + 2 * dNum];

    int hkey = fkey & (hsize - 1);
    int keyNum = num[hkey];
    int pSum = psum[hkey];

    for (int j = 0; j < keyNum; j++) {

      int dimKey = ((int *)(bucket))[2 * j + 2 * pSum];

      if (dimKey == fkey) {

        int dimId = ((int *)(bucket))[2 * j + 2 * pSum + 1];
        for (int k = 0; k < fcount; k++)
          factFilter[fpos + k] = dimId;

        break;
      }
    }
  }
}

/*
 * Count join result for uncompressed column
 */

extern "C" __global__ void count_join_result_hj(int *num, int *psum, char *bucket, char *fact, long inNum, int *count,
                                         int *factFilter, int hsize) {
  int lcount = 0;
  int stride = blockDim.x * gridDim.x;
  long offset = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = offset; i < inNum; i += stride) {
    int fkey = ((int *)(fact))[i];
    int hkey = fkey & (hsize - 1);
    int keyNum = num[hkey];
    int fvalue = 0;

    for (int j = 0; j < keyNum; j++) {
      int pSum = psum[hkey];
      int dimKey = ((int *)(bucket))[2 * j + 2 * pSum];
      if (dimKey == fkey) {
        int dimId = ((int *)(bucket))[2 * j + 2 * pSum + 1];
        lcount++;
        fvalue = dimId;
        break;
      }
    }
    factFilter[i] = fvalue;
  }

  count[offset] = lcount;
}

/*
 * Unpact the rle-compressed data
 */

extern "C" __global__ void unpack_rle_hj(char *fact, char *rle, long tupleNum, int dNum) {

  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = offset; i < dNum; i += stride) {

    int fvalue = ((int *)(fact + sizeof(struct rleHeader)))[i];
    int fcount = ((int *)(fact + sizeof(struct rleHeader)))[i + dNum];
    int fpos = ((int *)(fact + sizeof(struct rleHeader)))[i + 2 * dNum];

    for (int k = 0; k < fcount; k++) {
      ((int *)rle)[fpos + k] = fvalue;
    }
  }
}

/*
 * generate psum for RLE compressed column based on filter
 * current implementaton: scan through rle element and find the correponsding element in the filter
 */

extern "C" __global__ static void rle_psum(int *count, char *fact, long tupleNum, int *filter) {

  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  struct rleHeader *rheader = (struct rleHeader *)fact;
  int dNum = rheader->dictNum;

  for (int i = offset; i < dNum; i += stride) {

    int fcount = ((int *)(fact + sizeof(struct rleHeader)))[i + dNum];
    int fpos = ((int *)(fact + sizeof(struct rleHeader)))[i + 2 * dNum];
    int lcount = 0;

    for (int k = 0; k < fcount; k++) {
      if (filter[fpos + k] != 0)
        lcount++;
    }
    count[i] = lcount;
  }
}

/*
 * filter the column that is compressed using Run Length Encoding
 */

extern "C" __global__ void joinFact_rle_hj(int *resPsum, char *fact, int attrSize, long tupleNum, int *filter, char *result) {

  int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  struct rleHeader *rheader = (struct rleHeader *)fact;
  int dNum = rheader->dictNum;

  for (int i = startIndex; i < dNum; i += stride) {
    int fkey = ((int *)(fact + sizeof(struct rleHeader)))[i];
    int fcount = ((int *)(fact + sizeof(struct rleHeader)))[i + dNum];
    int fpos = ((int *)(fact + sizeof(struct rleHeader)))[i + 2 * dNum];

    int toffset = resPsum[i];
    for (int j = 0; j < fcount; j++) {
      if (filter[fpos - j] != 0) {
        ((int *)result)[toffset] = fkey;
        toffset++;
      }
    }
  }
}

/*
 * filter the column in the fact table that is compressed using dictionary encoding
 */
extern "C" __global__ void joinFact_dict_other_hj(int *resPsum, char *fact, struct dictHeader *dheader, int byteNum,
                                           int attrSize, long num, int *filter, char *result) {

  int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  long localOffset = resPsum[startIndex] * attrSize;

  for (long i = startIndex; i < num; i += stride) {
    if (filter[i] != 0) {
      int key = 0;
      memcpy(&key, fact + sizeof(struct dictHeader) + i * byteNum, byteNum);
      memcpy(result + localOffset, &dheader->hash[key], attrSize);
      localOffset += attrSize;
    }
  }
}

extern "C" __global__ void joinFact_dict_int_hj(int *resPsum, char *fact, struct dictHeader *dheader, int byteNum,
                                         int attrSize, long num, int *filter, char *result) {

  int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  long localCount = resPsum[startIndex];

  for (long i = startIndex; i < num; i += stride) {
    if (filter[i] != 0) {
      int key = 0;
      memcpy(&key, fact + sizeof(struct dictHeader) + i * byteNum, byteNum);
      ((int *)result)[localCount] = dheader->hash[key];
      localCount++;
    }
  }
}

extern "C" __global__ void joinFact_other_hj(int *resPsum, char *fact, int attrSize, long num, int *filter, char *result) {

  int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  long localOffset = resPsum[startIndex] * attrSize;

  for (long i = startIndex; i < num; i += stride) {
    if (filter[i] != 0) {
      memcpy(result + localOffset, fact + i * attrSize, attrSize);
      localOffset += attrSize;
    }
  }
}

extern "C" __global__ void joinFact_other_soa_hj(int *resPsum, char *fact, int attrSize, long tupleNum, long resultNum,
                                          int *filter, char *result) {

  int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  long tNum = resPsum[startIndex];

  for (long i = startIndex; i < tupleNum; i += stride) {
    if (filter[i] != 0) {
      for (int j = 0; j < attrSize; j++) {
        long inPos = j * tupleNum + i;
        long outPos = j * resultNum + tNum;
        result[outPos] = fact[inPos];
      }
    }
  }
}

extern "C" __global__ void joinFact_int_hj(int *resPsum, char *fact, int attrSize, long num, int *filter, char *result) {

  int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  long localCount = resPsum[startIndex];

  for (long i = startIndex; i < num; i += stride) {
    if (filter[i] != 0) {
      ((int *)result)[localCount] = ((int *)fact)[i];
      localCount++;
    }
  }
}

extern "C" __global__ void joinDim_rle_hj(int *resPsum, char *dim, int attrSize, long tupleNum, int *filter, char *result) {

  int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  long localCount = resPsum[startIndex];

  struct rleHeader *rheader = (struct rleHeader *)dim;
  int dNum = rheader->dictNum;

  for (int i = startIndex; i < tupleNum; i += stride) {
    int dimId = filter[i];
    if (dimId != 0) {
      for (int j = 0; j < dNum; j++) {
        int dkey = ((int *)(dim + sizeof(struct rleHeader)))[j];
        int dcount = ((int *)(dim + sizeof(struct rleHeader)))[j + dNum];
        int dpos = ((int *)(dim + sizeof(struct rleHeader)))[j + 2 * dNum];

        if (dpos == dimId || ((dpos < dimId) && (dpos + dcount) > dimId)) {
          ((int *)result)[localCount] = dkey;
          localCount++;
          break;
        }
      }
    }
  }
}

extern "C" __global__ void joinDim_dict_other_hj(int *resPsum, char *dim, struct dictHeader *dheader, int byteNum,
                                          int attrSize, long num, int *filter, char *result) {

  int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  long localOffset = resPsum[startIndex] * attrSize;

  for (long i = startIndex; i < num; i += stride) {
    int dimId = filter[i];
    if (dimId != 0) {
      int key = 0;
      memcpy(&key, dim + sizeof(struct dictHeader) + (dimId - 1) * byteNum, byteNum);
      memcpy(result + localOffset, &dheader->hash[key], attrSize);
      localOffset += attrSize;
    }
  }
}

extern "C" __global__ void joinDim_dict_int_hj(int *resPsum, char *dim, struct dictHeader *dheader, int byteNum, int attrSize,
                                        long num, int *filter, char *result) {

  int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  long localCount = resPsum[startIndex];

  for (long i = startIndex; i < num; i += stride) {
    int dimId = filter[i];
    if (dimId != 0) {
      int key = 0;
      memcpy(&key, dim + sizeof(struct dictHeader) + (dimId - 1) * byteNum, byteNum);
      ((int *)result)[localCount] = dheader->hash[key];
      localCount++;
    }
  }
}

extern "C" __global__ void joinDim_int_hj(int *resPsum, char *dim, int attrSize, long num, int *filter, char *result) {

  int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  long localCount = resPsum[startIndex];

  for (long i = startIndex; i < num; i += stride) {
    int dimId = filter[i];
    if (dimId != 0) {
      ((int *)result)[localCount] = ((int *)dim)[dimId - 1];
      localCount++;
    }
  }
}

extern "C" __global__ void joinDim_other_hj(int *resPsum, char *dim, int attrSize, long num, int *filter, char *result) {

  int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  long localOffset = resPsum[startIndex] * attrSize;

  for (long i = startIndex; i < num; i += stride) {
    int dimId = filter[i];
    if (dimId != 0) {
      memcpy(result + localOffset, dim + (dimId - 1) * attrSize, attrSize);
      localOffset += attrSize;
    }
  }
}

extern "C" __global__ void joinDim_other_soa_hj(int *resPsum, char *dim, int attrSize, long tupleNum, long resultNum,
                                         int *filter, char *result) {

  int startIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  long tNum = resPsum[startIndex];

  for (long i = startIndex; i < tupleNum; i += stride) {
    int dimId = filter[i];
    if (dimId != 0) {
      for (int j = 0; j < attrSize; j++) {
        long inPos = j * tupleNum + (dimId - 1);
        long outPos = j * resultNum + tNum;
        result[outPos] = dim[inPos];
      }
    }
  }
}

/*
 * hashJoin implements the foreign key join between a fact table and dimension table.
 *
 * Prerequisites:
 *  1. the data to be joined can be fit into GPU device memory.
 *  2. dimension table is not compressed
 *
 * Input:
 *  jNode: contains information about the two joined tables.
 *  pp: records statistics such as kernel execution time
 *
 * Output:
 *  A new table node
 */

struct tableNode *hashJoin(struct joinNode *jNode, struct statistic *pp) {

  extern char *col_buf;
  struct timeval t;
  struct timespec start, end;
  clock_gettime(CLOCK_REALTIME, &start);

  struct tableNode *res = NULL;

  char *gpu_result, *gpu_bucket, *gpu_fact, *gpu_dim;
  int *gpu_count, *gpu_psum, *gpu_resPsum, *gpu_hashNum;

  int defaultBlock = 4096;

  dim3 grid(defaultBlock);
  dim3 block(256);

  int blockNum;
  int threadNum;

  blockNum = jNode->leftTable->tupleNum / block.x + 1;
  if (blockNum < defaultBlock)
    grid = blockNum;
  else
    grid = defaultBlock;

  res = (struct tableNode *)malloc(sizeof(struct tableNode));
  CHECK_POINTER(res);
  res->totalAttr = jNode->totalAttr;
  res->tupleSize = jNode->tupleSize;
  res->attrType = (int *)malloc(res->totalAttr * sizeof(int));
  CHECK_POINTER(res->attrType);
  res->attrSize = (int *)malloc(res->totalAttr * sizeof(int));
  CHECK_POINTER(res->attrSize);
  res->attrIndex = (int *)malloc(res->totalAttr * sizeof(int));
  CHECK_POINTER(res->attrIndex);
  res->attrTotalSize = (int *)malloc(res->totalAttr * sizeof(int));
  CHECK_POINTER(res->attrTotalSize);
  res->dataPos = (int *)malloc(res->totalAttr * sizeof(int));
  CHECK_POINTER(res->dataPos);
  res->dataFormat = (int *)malloc(res->totalAttr * sizeof(int));
  CHECK_POINTER(res->dataFormat);
  res->content = (char **)malloc(res->totalAttr * sizeof(char *));
  CHECK_POINTER(res->content);

  for (int i = 0; i < jNode->leftOutputAttrNum; i++) {
    int pos = jNode->leftPos[i];
    res->attrType[pos] = jNode->leftOutputAttrType[i];
    int index = jNode->leftOutputIndex[i];
    res->attrSize[pos] = jNode->leftTable->attrSize[index];
    res->dataFormat[pos] = UNCOMPRESSED;
  }

  for (int i = 0; i < jNode->rightOutputAttrNum; i++) {
    int pos = jNode->rightPos[i];
    res->attrType[pos] = jNode->rightOutputAttrType[i];
    int index = jNode->rightOutputIndex[i];
    res->attrSize[pos] = jNode->rightTable->attrSize[index];
    res->dataFormat[pos] = UNCOMPRESSED;
  }

  long primaryKeySize = sizeof(int) * jNode->rightTable->tupleNum;

  /*
   *  build hash table on GPU
   */

  int *gpu_psum1;

  int hsize = jNode->rightTable->tupleNum;
  NP2(hsize);

  CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_hashNum, sizeof(int) * hsize));
  CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpu_hashNum, 0, sizeof(int) * hsize));

  CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_psum, hsize * sizeof(int)));
  CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_bucket, 2 * primaryKeySize));
  CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_psum1, hsize * sizeof(int)));

  int dataPos = jNode->rightTable->dataPos[jNode->rightKeyIndex];

  if (dataPos == MEM || dataPos == MMAP || dataPos == PINNED) {
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_dim, primaryKeySize));
    gettimeofday(&t, NULL);
    // printf("[gvm] %lf intercepting diskIO\n", t.tv_sec + t.tv_usec / 1000000.0);
    memcpy(col_buf, jNode->rightTable->content[jNode->rightKeyIndex], primaryKeySize);
    gettimeofday(&t, NULL);
    // printf("[gvm] %lf intercepted diskIO\n", t.tv_sec + t.tv_usec / 1000000.0);
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_dim, jNode->rightTable->content[jNode->rightKeyIndex], primaryKeySize,
                                      (enum cudaMemcpyKind)(cudaMemcpyHostToDevice)));

  } else if (dataPos == GPU || dataPos == UVA) {
    gpu_dim = jNode->rightTable->content[jNode->rightKeyIndex];
  }

  GMM_CALL(cudaAdvise(0, CADV_INPUT));
  GMM_CALL(cudaAdvise(2, CADV_DEFAULT));
  GMM_CALL(cudaSetFunction(7));
  count_hash_num_hj<<<grid, block>>>(gpu_dim, jNode->rightTable->tupleNum, gpu_hashNum, hsize);
  scanImpl(gpu_hashNum, hsize, gpu_psum, pp);

  CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_psum1, gpu_psum, sizeof(int) * hsize, cudaMemcpyDeviceToDevice));

  GMM_CALL(cudaAdvise(0, CADV_INPUT));
  GMM_CALL(cudaAdvise(2, CADV_DEFAULT));
  GMM_CALL(cudaAdvise(3, CADV_OUTPUT));
  GMM_CALL(cudaSetFunction(4));
  build_hash_table_hj<<<grid, block>>>(gpu_dim, jNode->rightTable->tupleNum, gpu_psum1, gpu_bucket, hsize);

  if (dataPos == MEM || dataPos == MMAP || dataPos == PINNED)
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_dim));

  CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_psum1));

  /*
   *  join on GPU
   */

  threadNum = grid.x * block.x;

  CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_count, sizeof(int) * threadNum));
  CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_resPsum, sizeof(int) * threadNum));

  int *gpuFactFilter;

  dataPos = jNode->leftTable->dataPos[jNode->leftKeyIndex];
  int format = jNode->leftTable->dataFormat[jNode->leftKeyIndex];

  long foreignKeySize = jNode->leftTable->attrTotalSize[jNode->leftKeyIndex];
  long filterSize = jNode->leftTable->attrSize[jNode->leftKeyIndex] * jNode->leftTable->tupleNum;

  if (dataPos == MEM || dataPos == MMAP || dataPos == PINNED) {
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, foreignKeySize));
    gettimeofday(&t, NULL);
    // printf("[gvm] %lf intercepting diskIO\n", t.tv_sec + t.tv_usec / 1000000.0);
    memcpy(col_buf, jNode->leftTable->content[jNode->leftKeyIndex], foreignKeySize);
    gettimeofday(&t, NULL);
    // printf("[gvm] %lf intercepted diskIO\n", t.tv_sec + t.tv_usec / 1000000.0);
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, jNode->leftTable->content[jNode->leftKeyIndex], foreignKeySize,
                                      (enum cudaMemcpyKind)(cudaMemcpyHostToDevice)));
    GMM_CALL(cudaAdvise(1, CADV_INPUT));

  } else if (dataPos == GPU || dataPos == UVA) {
    gpu_fact = jNode->leftTable->content[jNode->leftKeyIndex];
  }

  CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuFactFilter, filterSize));
  CUDA_SAFE_CALL_NO_SYNC(cudaMemset(gpuFactFilter, 0, filterSize));

  if (format == UNCOMPRESSED) {
    GMM_CALL(cudaAdvise(0, CADV_INPUT));
    GMM_CALL(cudaAdvise(1, CADV_INPUT));
    GMM_CALL(cudaAdvise(2, CADV_INPUT));
    GMM_CALL(cudaAdvise(3, CADV_INPUT));
    GMM_CALL(cudaAdvise(5, CADV_OUTPUT));
    GMM_CALL(cudaAdvise(6, CADV_OUTPUT));
    GMM_CALL(cudaSetFunction(12));
    count_join_result_hj<<<grid, block>>>(gpu_hashNum, gpu_psum, gpu_bucket, gpu_fact, jNode->leftTable->tupleNum,
                                       gpu_count, gpuFactFilter, hsize);
  } else if (format == DICT) {
    int dNum;
    struct dictHeader *dheader;
    struct dictHeader *gpuDictHeader;

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictHeader, sizeof(struct dictHeader)));

    if (dataPos == MEM || dataPos == MMAP || dataPos == UVA || dataPos == PINNED) {
      dheader = (struct dictHeader *)jNode->leftTable->content[jNode->leftKeyIndex];
      dNum = dheader->dictNum;
      CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader, dheader, sizeof(struct dictHeader), cudaMemcpyHostToDevice));

    } else if (dataPos == GPU) {
      dheader = (struct dictHeader *)malloc(sizeof(struct dictHeader));
      memset(dheader, 0, sizeof(struct dictHeader));
      CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(dheader, gpu_fact, sizeof(struct dictHeader), cudaMemcpyDeviceToHost));
      dNum = dheader->dictNum;
      CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader, dheader, sizeof(struct dictHeader), cudaMemcpyHostToDevice));
      free(dheader);
    }

    int *gpuDictFilter;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictFilter, dNum * sizeof(int)));

    GMM_CALL(cudaAdvise(0, CADV_INPUT));
    GMM_CALL(cudaAdvise(1, CADV_INPUT));
    GMM_CALL(cudaAdvise(2, CADV_INPUT));
    GMM_CALL(cudaAdvise(3, CADV_INPUT));
    GMM_CALL(cudaAdvise(5, CADV_OUTPUT));
    GMM_CALL(cudaSetFunction(11));
    count_join_result_dict_hj<<<grid, block>>>(gpu_hashNum, gpu_psum, gpu_bucket, gpuDictHeader, dNum, gpuDictFilter,
                                            hsize);

    GMM_CALL(cudaAdvise(0, CADV_INPUT));
    GMM_CALL(cudaAdvise(1, CADV_INPUT));
    GMM_CALL(cudaAdvise(4, CADV_OUTPUT));
    GMM_CALL(cudaSetFunction(127));
    transform_dict_filter_hj<<<grid, block>>>(gpuDictFilter, gpu_fact, jNode->leftTable->tupleNum, dNum, gpuFactFilter);

    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictFilter));
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

    GMM_CALL(cudaAdvise(1, CADV_OUTPUT));
    GMM_CALL(cudaAdvise(2, CADV_INPUT));
    GMM_CALL(cudaSetFunction(25));
    filter_count_hj<<<grid, block>>>(jNode->leftTable->tupleNum, gpu_count, gpuFactFilter);
  } else if (format == RLE) {
    GMM_CALL(cudaAdvise(0, CADV_INPUT));
    GMM_CALL(cudaAdvise(1, CADV_INPUT));
    GMM_CALL(cudaAdvise(2, CADV_INPUT));
    GMM_CALL(cudaAdvise(3, CADV_INPUT));
    GMM_CALL(cudaAdvise(5, CADV_OUTPUT));
    GMM_CALL(cudaSetFunction(15));
    count_join_result_rle_hj<<<512, 64>>>(gpu_hashNum, gpu_psum, gpu_bucket, gpu_fact, jNode->leftTable->tupleNum,
                                       gpuFactFilter, hsize);

    GMM_CALL(cudaAdvise(1, CADV_OUTPUT));
    GMM_CALL(cudaAdvise(2, CADV_INPUT));
    GMM_CALL(cudaSetFunction(25));
    filter_count_hj<<<grid, block>>>(jNode->leftTable->tupleNum, gpu_count, gpuFactFilter);
  }

  int tmp1, tmp2;

  CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&tmp1, &gpu_count[threadNum - 1], sizeof(int), cudaMemcpyDeviceToHost));
  scanImpl(gpu_count, threadNum, gpu_resPsum, pp);
  CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(&tmp2, &gpu_resPsum[threadNum - 1], sizeof(int), cudaMemcpyDeviceToHost));

  res->tupleNum = tmp1 + tmp2;
  printf("(INFO) Number of join results: %d\n", res->tupleNum);

  if (dataPos == MEM || dataPos == MMAP || dataPos == PINNED) {
    CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_fact));
  }

  CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_bucket));

  for (int i = 0; i < res->totalAttr; i++) {
    int index, pos;
    long colSize = 0, resSize = 0;
    int leftRight = 0;

    int attrSize, attrType;
    char *table;
    int found = 0, dataPos, format;

    if (jNode->keepInGpu[i] == 1)
      res->dataPos[i] = GPU;
    else
      res->dataPos[i] = MEM;

    for (int k = 0; k < jNode->leftOutputAttrNum; k++) {
      if (jNode->leftPos[k] == i) {
        found = 1;
        leftRight = 0;
        pos = k;
        break;
      }
    }
    if (!found) {
      for (int k = 0; k < jNode->rightOutputAttrNum; k++) {
        if (jNode->rightPos[k] == i) {
          found = 1;
          leftRight = 1;
          pos = k;
          break;
        }
      }
    }

    if (leftRight == 0) {
      index = jNode->leftOutputIndex[pos];
      dataPos = jNode->leftTable->dataPos[index];
      format = jNode->leftTable->dataFormat[index];

      table = jNode->leftTable->content[index];
      attrSize = jNode->leftTable->attrSize[index];
      attrType = jNode->leftTable->attrType[index];
      colSize = jNode->leftTable->attrTotalSize[index];

      resSize = res->tupleNum * attrSize;
    } else {
      index = jNode->rightOutputIndex[pos];
      dataPos = jNode->rightTable->dataPos[index];
      format = jNode->rightTable->dataFormat[index];

      table = jNode->rightTable->content[index];
      attrSize = jNode->rightTable->attrSize[index];
      attrType = jNode->rightTable->attrType[index];
      colSize = jNode->rightTable->attrTotalSize[index];

      resSize = attrSize * res->tupleNum;
      leftRight = 1;
    }

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_result, resSize));

    if (leftRight == 0) {
      if (format == UNCOMPRESSED) {

        if (dataPos == MEM || dataPos == MMAP || dataPos == PINNED) {
          CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
          gettimeofday(&t, NULL);
          // printf("[gvm] %lf intercepting diskIO\n", t.tv_sec + t.tv_usec / 1000000.0);
          memcpy(col_buf, table, colSize);
          gettimeofday(&t, NULL);
          // printf("[gvm] %lf intercepted diskIO\n", t.tv_sec + t.tv_usec / 1000000.0);
          CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize, (enum cudaMemcpyKind)(cudaMemcpyHostToDevice)));
        } else {
          gpu_fact = table;
        }

        if (attrSize == sizeof(int)) {
          GMM_CALL(cudaAdvise(0, CADV_INPUT));
          GMM_CALL(cudaAdvise(1, CADV_INPUT));
          GMM_CALL(cudaAdvise(4, CADV_INPUT));
          GMM_CALL(cudaAdvise(5, CADV_OUTPUT));
          GMM_CALL(cudaSetFunction(100));
          joinFact_int_hj<<<grid, block>>>(gpu_resPsum, gpu_fact, attrSize, jNode->leftTable->tupleNum, gpuFactFilter,
                                        gpu_result);
        } else {
          GMM_CALL(cudaAdvise(0, CADV_INPUT));
          GMM_CALL(cudaAdvise(1, CADV_INPUT));
          GMM_CALL(cudaAdvise(4, CADV_INPUT));
          GMM_CALL(cudaAdvise(5, CADV_OUTPUT));
          GMM_CALL(cudaSetFunction(103));
          joinFact_other_hj<<<grid, block>>>(gpu_resPsum, gpu_fact, attrSize, jNode->leftTable->tupleNum, gpuFactFilter,
                                          gpu_result);
        }

      } else if (format == DICT) {
        struct dictHeader *dheader;
        int byteNum;
        struct dictHeader *gpuDictHeader;

        dheader = (struct dictHeader *)table;
        byteNum = dheader->bitNum / 8;
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictHeader, sizeof(struct dictHeader)));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader, dheader, sizeof(struct dictHeader), cudaMemcpyHostToDevice));

        if (dataPos == MEM || dataPos == MMAP || dataPos == PINNED) {
          CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
          gettimeofday(&t, NULL);
          // printf("[gvm] %lf intercepting diskIO\n", t.tv_sec + t.tv_usec / 1000000.0);
          memcpy(col_buf, table, colSize);
          gettimeofday(&t, NULL);
          // printf("[gvm] %lf intercepted diskIO\n", t.tv_sec + t.tv_usec / 1000000.0);
          CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize, (enum cudaMemcpyKind)(cudaMemcpyHostToDevice)));
        } else {
          gpu_fact = table;
        }

        if (attrSize == sizeof(int)) {
          GMM_CALL(cudaAdvise(0, CADV_INPUT));
          GMM_CALL(cudaAdvise(1, CADV_INPUT));
          GMM_CALL(cudaAdvise(2, CADV_INPUT));
          GMM_CALL(cudaAdvise(6, CADV_INPUT));
          GMM_CALL(cudaAdvise(7, CADV_OUTPUT));
          GMM_CALL(cudaSetFunction(95));
          joinFact_dict_int_hj<<<grid, block>>>(gpu_resPsum, gpu_fact, gpuDictHeader, byteNum, attrSize,
                                             jNode->leftTable->tupleNum, gpuFactFilter, gpu_result);
        } else {
          GMM_CALL(cudaAdvise(0, CADV_INPUT));
          GMM_CALL(cudaAdvise(1, CADV_INPUT));
          GMM_CALL(cudaAdvise(2, CADV_INPUT));
          GMM_CALL(cudaAdvise(6, CADV_INPUT));
          GMM_CALL(cudaAdvise(7, CADV_OUTPUT));
          GMM_CALL(cudaSetFunction(97));
          joinFact_dict_other_hj<<<grid, block>>>(gpu_resPsum, gpu_fact, gpuDictHeader, byteNum, attrSize,
                                               jNode->leftTable->tupleNum, gpuFactFilter, gpu_result);
        }

        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

      } else if (format == RLE) {

        struct rleHeader *rheader;

        if (dataPos == MEM || dataPos == MMAP || dataPos == PINNED) {
          CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
          gettimeofday(&t, NULL);
          // printf("[gvm] %lf intercepting diskIO\n", t.tv_sec + t.tv_usec / 1000000.0);
          memcpy(col_buf, table, colSize);
          gettimeofday(&t, NULL);
          // printf("[gvm] %lf intercepted diskIO\n", t.tv_sec + t.tv_usec / 1000000.0);
          CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize, (enum cudaMemcpyKind)(cudaMemcpyHostToDevice)));
        } else {
          gpu_fact = table;
        }

        rheader = (struct rleHeader *)table;

        int dNum = rheader->dictNum;

        char *gpuRle;
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuRle, jNode->leftTable->tupleNum * sizeof(int)));

        GMM_CALL(cudaAdvise(0, CADV_INPUT));
        GMM_CALL(cudaAdvise(1, CADV_OUTPUT));
        GMM_CALL(cudaSetFunction(132));
        unpack_rle_hj<<<grid, block>>>(gpu_fact, gpuRle, jNode->leftTable->tupleNum, dNum);

        GMM_CALL(cudaAdvise(0, CADV_INPUT));
        GMM_CALL(cudaAdvise(1, CADV_INPUT));
        GMM_CALL(cudaAdvise(4, CADV_INPUT));
        GMM_CALL(cudaAdvise(5, CADV_OUTPUT));
        GMM_CALL(cudaSetFunction(100));
        joinFact_int_hj<<<grid, block>>>(gpu_resPsum, gpuRle, attrSize, jNode->leftTable->tupleNum, gpuFactFilter,
                                      gpu_result);

        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuRle));
      }

    } else {
      if (format == UNCOMPRESSED) {

        if (dataPos == MEM || dataPos == MMAP || dataPos == PINNED) {
          CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
          gettimeofday(&t, NULL);
          // printf("[gvm] %lf intercepting diskIO\n", t.tv_sec + t.tv_usec / 1000000.0);
          memcpy(col_buf, table, colSize);
          gettimeofday(&t, NULL);
          // printf("[gvm] %lf intercepted diskIO\n", t.tv_sec + t.tv_usec / 1000000.0);
          CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize, (enum cudaMemcpyKind)(cudaMemcpyHostToDevice)));
          GMM_CALL(cudaAdvise(1, CADV_INPUT));
        } else {
          gpu_fact = table;
        }

        if (attrType == sizeof(int)) {
          GMM_CALL(cudaAdvise(0, CADV_INPUT));
          GMM_CALL(cudaAdvise(1, CADV_INPUT));
          GMM_CALL(cudaAdvise(4, CADV_INPUT));
          GMM_CALL(cudaAdvise(5, CADV_OUTPUT));
          GMM_CALL(cudaSetFunction(87));
          joinDim_int_hj<<<grid, block>>>(gpu_resPsum, gpu_fact, attrSize, jNode->leftTable->tupleNum, gpuFactFilter,
                                       gpu_result);
        } else {
          GMM_CALL(cudaAdvise(0, CADV_INPUT));
          GMM_CALL(cudaAdvise(1, CADV_INPUT));
          GMM_CALL(cudaAdvise(4, CADV_INPUT));
          GMM_CALL(cudaAdvise(5, CADV_OUTPUT));
          GMM_CALL(cudaSetFunction(90));
          joinDim_other_hj<<<grid, block>>>(gpu_resPsum, gpu_fact, attrSize, jNode->leftTable->tupleNum, gpuFactFilter,
                                         gpu_result);
        }

      } else if (format == DICT) {
        struct dictHeader *dheader;
        int byteNum;
        struct dictHeader *gpuDictHeader;

        dheader = (struct dictHeader *)table;
        byteNum = dheader->bitNum / 8;
        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpuDictHeader, sizeof(struct dictHeader)));
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpuDictHeader, dheader, sizeof(struct dictHeader), cudaMemcpyHostToDevice));
        if (dataPos == MEM || dataPos == MMAP || dataPos == PINNED) {
          CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
          gettimeofday(&t, NULL);
          // printf("[gvm] %lf intercepting diskIO\n", t.tv_sec + t.tv_usec / 1000000.0);
          memcpy(col_buf, table, colSize);
          gettimeofday(&t, NULL);
          // printf("[gvm] %lf intercepted diskIO\n", t.tv_sec + t.tv_usec / 1000000.0);
          CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize, (enum cudaMemcpyKind)(cudaMemcpyHostToDevice)));
        } else {
          gpu_fact = table;
        }

        if (attrSize == sizeof(int)) {
          GMM_CALL(cudaAdvise(0, CADV_INPUT));
          GMM_CALL(cudaAdvise(1, CADV_INPUT));
          GMM_CALL(cudaAdvise(2, CADV_INPUT));
          GMM_CALL(cudaAdvise(6, CADV_INPUT));
          GMM_CALL(cudaAdvise(7, CADV_OUTPUT));
          GMM_CALL(cudaSetFunction(82));
          joinDim_dict_int_hj<<<grid, block>>>(gpu_resPsum, gpu_fact, gpuDictHeader, byteNum, attrSize,
                                            jNode->leftTable->tupleNum, gpuFactFilter, gpu_result);
        } else {
          GMM_CALL(cudaAdvise(0, CADV_INPUT));
          GMM_CALL(cudaAdvise(1, CADV_INPUT));
          GMM_CALL(cudaAdvise(2, CADV_INPUT));
          GMM_CALL(cudaAdvise(6, CADV_INPUT));
          GMM_CALL(cudaAdvise(7, CADV_OUTPUT));
          GMM_CALL(cudaSetFunction(84));
          joinDim_dict_other_hj<<<grid, block>>>(gpu_resPsum, gpu_fact, gpuDictHeader, byteNum, attrSize,
                                              jNode->leftTable->tupleNum, gpuFactFilter, gpu_result);
        }
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpuDictHeader));

      } else if (format == RLE) {

        if (dataPos == MEM || dataPos == MMAP || dataPos == PINNED) {
          CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&gpu_fact, colSize));
          gettimeofday(&t, NULL);
          // printf("[gvm] %lf intercepting diskIO\n", t.tv_sec + t.tv_usec / 1000000.0);
          memcpy(col_buf, table, colSize);
          gettimeofday(&t, NULL);
          // printf("[gvm] %lf intercepted diskIO\n", t.tv_sec + t.tv_usec / 1000000.0);
          CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(gpu_fact, table, colSize, (enum cudaMemcpyKind)(cudaMemcpyHostToDevice)));
        } else {
          gpu_fact = table;
        }

        GMM_CALL(cudaAdvise(0, CADV_INPUT));
        GMM_CALL(cudaAdvise(1, CADV_INPUT));
        GMM_CALL(cudaAdvise(4, CADV_INPUT));
        GMM_CALL(cudaAdvise(5, CADV_OUTPUT));
        GMM_CALL(cudaSetFunction(93));
        joinDim_rle_hj<<<grid, block>>>(gpu_resPsum, gpu_fact, attrSize, jNode->leftTable->tupleNum, gpuFactFilter,
                                     gpu_result);
      }
    }

    res->attrTotalSize[i] = resSize;
    res->dataFormat[i] = UNCOMPRESSED;
    if (res->dataPos[i] == MEM) {
      res->content[i] = (char *)malloc(resSize);
      memset(res->content[i], 0, resSize);
      CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(res->content[i], gpu_result, resSize, cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_result));

    } else if (res->dataPos[i] == GPU) {
      res->content[i] = gpu_result;
    }
    if (dataPos == MEM || dataPos == MMAP || dataPos == PINNED)
      CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_fact));
  }

  CUDA_SAFE_CALL(cudaFree(gpuFactFilter));

  CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_count));
  CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_hashNum));
  CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_psum));
  CUDA_SAFE_CALL_NO_SYNC(cudaFree(gpu_resPsum));

  clock_gettime(CLOCK_REALTIME, &end);
  double timeE = (end.tv_sec - start.tv_sec) * BILLION + end.tv_nsec - start.tv_nsec;
  printf("HashJoin Time: %lf\n", timeE / (1000 * 1000));

  return res;
}
