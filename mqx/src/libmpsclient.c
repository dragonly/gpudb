#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>
#include <errno.h>
#include <cuda_runtime_api.h>
#include "common.h"
#include "protocol.h"
#include "serialize.h"

// TODO: check all kinds of pointers
// TODO: statistics

static int client_socket;

int mpsclient_init() {
  struct sockaddr_un server;
  client_socket = socket(AF_UNIX, SOCK_STREAM, 0);
  if (client_socket < 0) {
    printf("opening client socket: %s\n", strerror(errno));
    return -1;
  }
  server.sun_family = AF_UNIX;
  strcpy(server.sun_path, SERVER_SOCKET_FILE);
  if (connect(client_socket, (struct sockaddr *) &server, sizeof(struct sockaddr_un)) < 0) {
    close(client_socket);
    perror("conencting server socket");
    return -1;
  }
  return 0;
}
void mpsclient_destroy() {
  close(client_socket);
}
cudaError_t mpsclient_cudaMalloc(void **devPtr, size_t size, uint32_t flags) {
  // devPtr is not sent in socket
  int buf_size = sizeof(size_t) + sizeof(uint32_t);
  unsigned char *buf = malloc(buf_size);
  struct mps_req req;
  req.type = REQ_GPU_MALLOC;
  req.len = buf_size;
  serialize_mps_req(buf, req);
  send(client_socket, buf, MPS_REQ_SIZE, 0);
  unsigned char *pbuf;
  pbuf = buf;
  pbuf = serialize_uint64(pbuf, size);
  pbuf = serialize_uint32(pbuf, flags);
  send(client_socket, buf, buf_size, 0);
  recv(client_socket, buf, 4+8, 0);
  cudaError_t ret;
  pbuf = buf;
  pbuf = deserialize_uint32(pbuf, &ret);
  pbuf = deserialize_uint64(pbuf, (uint64_t *)devPtr);
  mqx_print(DEBUG, "client cudaMalloc: devPtr(%p), ret(%d)", *(void **)devPtr, ret);
  return ret;
}
cudaError_t mpsclient_cudaFree(void *devPtr) {
  struct mps_req req;
  req.type = REQ_GPU_MEMFREE;
  int buf_size = sizeof(void *);
  unsigned char *buf = malloc(buf_size);
  serialize_mps_req(buf, req);
  send(client_socket, buf, MPS_REQ_SIZE, 0);
  serialize_uint64(buf, (uint64_t)devPtr);
  send(client_socket, buf, buf_size, 0);
  recv(client_socket, buf, 4, 0);
  cudaError_t ret;
  deserialize_uint32(buf, &ret);
  return ret;
}
cudaError_t mpsclient_cudaMemcpyHtoD(void *dst, const void *src, size_t size) {
  return cudaSuccess;
}
cudaError_t mpsclient_cudaMemcpyDtoH(void *dst, const void *src, size_t size) {
  return cudaSuccess;
}
cudaError_t mpsclient_cudaMemcpyDtoD(void *dst, const void *src, size_t size) {
  return cudaSuccess;
}
cudaError_t mpsclient_cudaMemcpyDefault(void *dst, const void *src, size_t size) {
  return cudaSuccess;
}
cudaError_t mpsclient_cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args, size_t sharedMem, cudaStream_t stream) {
  struct mps_req req;
  req.type = REQ_GPU_LAUNCH_KERNEL;
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
}

