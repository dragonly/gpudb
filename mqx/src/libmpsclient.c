#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>
#include <errno.h>
#include "common.h"
#include "protocol.h"
#include "serialize.h"
#include "libmps.h"
#include "kernel_symbols.h"

// TODO: check all kinds of pointers
// TODO: statistics

static uint8_t buf[1024];
#define CLIENT_REQ_HEAD(LEN, REQ_TYPE, ROUND, LAST_LEN) \
  uint32_t payload_size = (LEN);                        \
  struct mps_req req;                                   \
  req.type = REQ_TYPE;                                  \
  req.len = (LEN);                                      \
  req.round = (ROUND);                                  \
  req.last_len = (LAST_LEN);                            \
  serialize_mps_req(buf, req);                          \
  send(client_socket, buf, MPS_REQ_SIZE, 0);

#define CLIENT_REQ_TAIL(EXTRA_STMT)         \
  cudaError_t ret;                          \
  recv(client_socket, buf, sizeof(ret), 0); \
  deserialize_uint32(buf, &ret);            \
  EXTRA_STMT;                               \
  return ret;


static int client_socket;

int mpsclient_init() {
  struct sockaddr_un server;
  client_socket = socket(AF_UNIX, SOCK_STREAM, 0);
  if (client_socket < 0) {
    mqx_print(FATAL, "opening client socket: %s\n", strerror(errno));
    return -1;
  }
  server.sun_family = AF_UNIX;
  strcpy(server.sun_path, SERVER_SOCKET_FILE);
  if (connect(client_socket, (struct sockaddr *) &server, sizeof(struct sockaddr_un)) < 0) {
    close(client_socket);
    perror("conencting server socket");
    return -1;
  }
  mqx_print(DEBUG, "connected");
  return 0;
}
cudaError_t mpsclient_destroy() {
  CLIENT_REQ_HEAD(0, REQ_QUIT, 0, 0);
  send(client_socket, buf, payload_size, 0);
  close(client_socket);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "quit"));
}

cudaError_t mpsclient_cudaMalloc(void **devPtr, size_t size, uint32_t flags) {
  // devPtr is not sent in socket
  CLIENT_REQ_HEAD(sizeof(size) + sizeof(flags), REQ_CUDA_MALLOC, 0, 0);
  uint8_t *pbuf;
  pbuf = buf;
  pbuf = serialize_uint64(pbuf, size);
  pbuf = serialize_uint32(pbuf, flags);
  send(client_socket, buf, payload_size, 0);
  recv(client_socket, buf, sizeof(devPtr), 0);
  deserialize_uint64(buf, (uint64_t *)devPtr);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "devPtr(%p) size(%lu) ret(%d)", *(void **)devPtr, size, ret))
}
cudaError_t mpsclient_cudaFree(void *devPtr) {
  CLIENT_REQ_HEAD(sizeof(devPtr), REQ_CUDA_MEMFREE, 0, 0);
  serialize_uint64(buf, (uint64_t)devPtr);
  send(client_socket, buf, payload_size, 0);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "%p", devPtr));
}
cudaError_t mpsclient_cudaMemcpyHostToDevice(void *dst, void *src, size_t size) {
  CLIENT_REQ_HEAD(sizeof(dst), REQ_CUDA_MEMCPY_HTOD, size / MAX_BUFFER_SIZE + 1, size % MAX_BUFFER_SIZE);
  serialize_uint64(buf, (uint64_t)dst);
  send(client_socket, buf, payload_size, 0);
  mqx_print(DEBUG, "starting a (%lu bytes/%d round) Host->Device memcpy", size, req.round);
  int round_size;
  for (int i = 0; i < req.round; i++) {
    round_size = i == req.round - 1 ? req.last_len : MAX_BUFFER_SIZE;
    if (send_large_buf(client_socket, src + i * MAX_BUFFER_SIZE, round_size) != 0) {
      return cudaErrorUnknown;
    }
  }
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "dst(%p) size(%zu) ret(%d)", dst, size, ret))
}
cudaError_t mpsclient_cudaMemcpyDeviceToHost(void *dst, void *src, size_t size) {
  CLIENT_REQ_HEAD(sizeof(src), REQ_CUDA_MEMCPY_DTOH, size / MAX_BUFFER_SIZE + 1, size % MAX_BUFFER_SIZE);
  serialize_uint64(buf, (uint64_t)src);
  send(client_socket, buf, payload_size, 0);
  int round_size;
  for (int i = 0; i < req.round; i++) {
    round_size = i == req.round - 1 ? req.last_len : MAX_BUFFER_SIZE;
    if (recv_large_buf(client_socket, dst + i * MAX_BUFFER_SIZE, round_size) != 0) {
      return cudaErrorUnknown;
    }
  }
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "src(%p) size(%zu) ret(%d)", src, size, ret))
}
cudaError_t mpsclient_cudaMemcpyDeviceToDevice(void *dst, void *src, size_t size) {
  CLIENT_REQ_HEAD(sizeof(dst) + sizeof(src) + sizeof(size), REQ_CUDA_MEMCPY_DTOD, 0, 0);
  uint8_t *pbuf;
  pbuf = buf;
  pbuf = serialize_uint64(pbuf, (uint64_t)dst);
  pbuf = serialize_uint64(pbuf, (uint64_t)src);
  pbuf = serialize_uint64(pbuf, size);
  send(client_socket, buf, payload_size, 0);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "dst(%p) src(%p) size(%lu)", dst, src, size));
}
cudaError_t mpsclient_cudaMemcpyHostToHost(void *dst, void *src, size_t size) {
  CLIENT_REQ_HEAD(0, REQ_CUDA_MEMCPY_HTOH, 0, 0);
  send(client_socket, buf, payload_size, 0);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "dst(%p) src(%p) size(%lu)", dst, src, size));
}
cudaError_t mpsclient_cudaMemcpyDefault(void *dst, void *src, size_t size) {
  CLIENT_REQ_HEAD(0, REQ_CUDA_MEMCPY_DEFAULT, 0, 0);
  send(client_socket, buf, payload_size, 0);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "dst(%p) src(%p) size(%lu)", dst, src, size));
}
cudaError_t mpsclient_cudaMemcpy(void *dst, void *src, size_t size, enum cudaMemcpyKind kind) {
  switch (kind) {
    case cudaMemcpyHostToDevice:
      return mpsclient_cudaMemcpyHostToDevice(dst, src, size);
    case cudaMemcpyDeviceToHost:
      return mpsclient_cudaMemcpyDeviceToHost(dst, src, size);
    case cudaMemcpyDeviceToDevice:
      return mpsclient_cudaMemcpyDeviceToDevice(dst, src, size);
    case cudaMemcpyDefault:
      return mpsclient_cudaMemcpyDefault(dst, src, size);
    case cudaMemcpyHostToHost:
      return cudaErrorNotYetImplemented;
  }
  return cudaErrorInvalidMemcpyDirection;
}
cudaError_t mpsclient_cudaMemset(void *devPtr, int32_t value, size_t count) {
  CLIENT_REQ_HEAD(sizeof(devPtr) + sizeof(int32_t) + sizeof(count), REQ_CUDA_MEMSET, 0, 0);
  uint8_t *pbuf;
  pbuf = buf;
  pbuf = serialize_uint64(pbuf, (uint64_t)devPtr);
  pbuf = serialize_uint32(pbuf, value);
  pbuf = serialize_uint64(pbuf, count);
  send(client_socket, buf, payload_size, 0);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "devPtr(%p) value(%d) count(%lu)", devPtr, value, count));
}
cudaError_t mpsclient_cudaAdvise(uint8_t iarg, uint8_t advice) {
  CLIENT_REQ_HEAD(sizeof(iarg) + sizeof(advice), REQ_CUDA_ADVISE, 0, 0);
  uint8_t *pbuf;
  pbuf = buf;
  pbuf = serialize_str(pbuf, &iarg, 1);
  pbuf = serialize_str(pbuf, &advice, 1);
  send(client_socket, buf, payload_size, 0);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "iarg(%d) advice(%d)", iarg, advice));
}
cudaError_t mpsclient_cudaSetFunction(uint32_t index) {
  CLIENT_REQ_HEAD(sizeof(index), REQ_SET_KERNEL_FUNCTION, 0, 0);
  serialize_uint32(buf, index);
  send(client_socket, buf, payload_size, 0);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "<%s>[%d]", fname_table[index], index));
}
cudaError_t mpsclient_cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream) {
  CLIENT_REQ_HEAD(6*sizeof(uint32_t) + sizeof(sharedMem) + sizeof(stream), REQ_CUDA_CONFIGURE_CALL, 0, 0);
  uint8_t *pbuf;
  pbuf = buf;
  pbuf = serialize_uint32(pbuf, gridDim.x);
  pbuf = serialize_uint32(pbuf, gridDim.y);
  pbuf = serialize_uint32(pbuf, gridDim.z);
  pbuf = serialize_uint32(pbuf, blockDim.x);
  pbuf = serialize_uint32(pbuf, blockDim.y);
  pbuf = serialize_uint32(pbuf, blockDim.z);
  pbuf = serialize_uint64(pbuf, sharedMem);
  pbuf = serialize_uint64(pbuf, (uint64_t)stream);
  send(client_socket, buf, payload_size, 0);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "<<<(%d %d %d), (%d %d %d), %lu, %p>>>", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, sharedMem, stream));
}
// if `arg` is a device ptr, it is passed as is or it points to a host area and `size` bytes should be copied as the argument, i.e. primitive types such as int
// anyway it is a ptr, and we should send the underneath value
cudaError_t mpsclient_cudaSetupArgument(const void *arg, size_t size, size_t offset) {
  CLIENT_REQ_HEAD(sizeof(arg) + sizeof(size) + sizeof(offset), REQ_CUDA_SETUP_ARGUMENT, 0, 0);
  uint8_t *pbuf;
  pbuf = buf;
  //pbuf = serialize_uint64(pbuf, (uint64_t)arg);
  pbuf = serialize_uint64(pbuf, size);
  pbuf = serialize_str(pbuf, (uint8_t *)arg, size);
  pbuf = serialize_uint64(pbuf, offset);
  send(client_socket, buf, payload_size, 0);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "arg(%p) size(%zu) offset(%zu)", arg, size, offset));
}
cudaError_t mpsclient_cudaLaunch(const void *func) {
  CLIENT_REQ_HEAD(0, REQ_CUDA_LAUNCH_KERNEL, 0, 0);
  send(client_socket, buf, payload_size, 0);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "kernel launched"));
}
cudaError_t mpsclient_cudaMemGetInfo(size_t *mem_free, size_t *mem_total) {
  CLIENT_REQ_HEAD(0, REQ_CUDA_MEM_GET_INFO, 0, 0);
  send(client_socket, buf, payload_size, 0);
  recv(client_socket, buf, sizeof(size_t) * 2, 0);
  uint8_t *pbuf;
  pbuf = buf;
  pbuf = deserialize_uint64(pbuf, mem_free);
  pbuf = deserialize_uint64(pbuf, mem_total);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "got free and total memory size"));
}
cudaError_t mpsclient_cudaDeviceSynchronize() {
  CLIENT_REQ_HEAD(0, REQ_CUDA_DEVICE_SYNCHRONIZE, 0, 0);
  send(client_socket, buf, payload_size, 0);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "synchronized"));
}
cudaError_t mpsclient_cudaEventCreate(cudaEvent_t *event) {
  CLIENT_REQ_HEAD(0, REQ_CUDA_EVENT_CREATE, 0, 0);
  send(client_socket, buf, payload_size, 0);
  recv(client_socket, buf, sizeof(cudaEvent_t), 0);
  deserialize_uint64(buf, (uint64_t *)event);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "event created(%p)", *event));
}
cudaError_t mpsclient_cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
  CLIENT_REQ_HEAD(sizeof(start) + sizeof(end), REQ_CUDA_EVENT_ELAPSED_TIME, 0, 0);
  uint8_t *pbuf;
  pbuf = buf;
  pbuf = serialize_uint64(pbuf, (uint64_t)start);
  pbuf = serialize_uint64(pbuf, (uint64_t)end);
  send(client_socket, buf, payload_size, 0);
  recv(client_socket, buf, sizeof(float), 0);
  deserialize_uint32(buf, (uint32_t *)ms);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "%.2f ms", *ms));
}
cudaError_t mpsclient_cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  CLIENT_REQ_HEAD(sizeof(event) + sizeof(stream), REQ_CUDA_EVENT_RECORD, 0, 0);
  uint8_t *pbuf;
  pbuf = buf;
  pbuf = serialize_uint64(pbuf, (uint64_t)event);
  pbuf = serialize_uint64(pbuf, (uint64_t)stream);
  send(client_socket, buf, payload_size, 0);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "event(%p) recorded", event));
}
cudaError_t mpsclient_cudaEventSynchronize(cudaEvent_t event) {
  CLIENT_REQ_HEAD(sizeof(event), REQ_CUDA_EVENT_SYNCHRONIZE, 0, 0);
  serialize_uint64(buf, (uint64_t)event);
  send(client_socket, buf, payload_size, 0);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "event(%p) synchronized", event));
}
cudaError_t mpsclient_cudaGetDevice(int *device) {
  CLIENT_REQ_HEAD(0, REQ_CUDA_GET_DEVICE, 0, 0);
  send(client_socket, buf, payload_size, 0);
  recv(client_socket, buf, sizeof(int), 0);
  deserialize_uint32(buf, (uint32_t *)device);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "current device is %d", *device));
}
cudaError_t mpsclient_cudaGetColumnBlockAddress(void **devPtr, const char *colname, uint32_t iblock) {
  CLIENT_REQ_HEAD(strlen(colname) + sizeof(size_t) + sizeof(iblock), REQ_CUDA_GET_COLUMN_BLOCK_ADDRESS, 0, 0);
  uint8_t *pbuf;
  pbuf = buf;
  pbuf = serialize_uint64(pbuf, strlen(colname));
  pbuf = serialize_str(pbuf, (uint8_t *)colname, strlen(colname));
  pbuf = serialize_uint32(pbuf, iblock);
  send(client_socket, buf, payload_size, 0);
  recv(client_socket, buf, sizeof(void *), 0);
  deserialize_uint64(buf, (uint64_t *)devPtr);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "%p", *devPtr));
}
// the received payload is larger than the sent one, which breaks this buggy design. oops...
cudaError_t mpsclient_cudaGetColumnBlockHeader(struct columnHeader *pheader, const char *colname, uint32_t iblock) {
  CLIENT_REQ_HEAD(sizeof(size_t) + strlen(colname) + sizeof(iblock), REQ_CUDA_GET_COLUMN_BLOCK_HEADER, 0, 0);
  uint8_t *pbuf;
  pbuf = buf;
  pbuf = serialize_uint64(pbuf, strlen(colname));
  pbuf = serialize_str(pbuf, (uint8_t *)colname, strlen(colname));
  pbuf = serialize_uint32(pbuf, iblock);
  send(client_socket, buf, payload_size, 0);
  recv(client_socket, buf, sizeof(uint64_t) * 3 + sizeof(uint32_t) * 3, 0);
  pbuf = buf;
  pbuf = deserialize_uint64(pbuf, (uint64_t *)&pheader->totalTupleNum);
  pbuf = deserialize_uint64(pbuf, (uint64_t *)&pheader->tupleNum);
  pbuf = deserialize_uint64(pbuf, (uint64_t *)&pheader->blockSize);
  pbuf = deserialize_uint32(pbuf, (uint32_t *)&pheader->blockTotal);
  pbuf = deserialize_uint32(pbuf, (uint32_t *)&pheader->blockId);
  pbuf = deserialize_uint32(pbuf, (uint32_t *)&pheader->format);
  CLIENT_REQ_TAIL(\
    mqx_print(DEBUG, "%s totalTupleNum(%lu) tupleNum(%lu) blockSize(%lu) blockTotal(%d) blockId(%d) format(%d)", colname, pheader->totalTupleNum, pheader->tupleNum, pheader->blockSize, pheader->blockTotal, pheader->blockId, pheader->format));
}
// test function
cudaError_t mpsclient_cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
  struct mps_req req;
  req.type = REQ_CUDA_LAUNCH_KERNEL;
  struct kernel_args kargs;
  kargs.arg_info[0] = (void *)3;
  kargs.arg_info[1] = (void *)0;
  kargs.arg_info[2] = (void *)2;
  kargs.arg_info[3] = (void *)10;
  *(uint16_t *)kargs.args = 63;
  *(uint64_t *)((uint8_t *)kargs.args+2) = 6;
  *(uint16_t *)((uint8_t *)kargs.args+10) = 7;
  kargs.last_arg_len = 2;
  kargs.blocks_per_grid = 2;
  kargs.threads_per_block = 32;
  kargs.function_index = 233;
  int nbytes = kernel_args_bytes(kargs);
  req.len = nbytes;

  unsigned char *buf = (unsigned char *)malloc(nbytes);
  serialize_mps_req(buf, req);
  if (send(client_socket, buf, 4, 0) == -1) {
    perror("writing to client socket");
  }
  sleep(1);
  serialize_kernel_args(buf, kargs);
  if (send(client_socket, buf, nbytes, 0) == -1) {
    perror("writing to client socket");
  }
  recv(client_socket, buf, 2, 0);
  cudaError_t res;
  deserialize_uint32(buf, &res);
  return cudaSuccess;
}

