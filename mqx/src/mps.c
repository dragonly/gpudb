#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>
#include <errno.h>
#include <cuda_runtime_api.h>
#include "common.h"
#include "protocol.h"
#include "serialize.h"
#include "kernel_symbols.h"

static int client_socket;

int mps_client_init() {
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
}
cudaError_t mpsclient_cudaMalloc(void **devPtr, size_t size, int flags) {
  struct mps_req req;
  req.type = REQ_GPU_MALLOC;
  int buf_size = max(sizeof(void **) + sizeof(size_t) + sizeof(int), MPS_REQ_SIZE);
  char *buf = malloc(buf_size);
  serialize_mps_req(buf, req);
  send(client_socket, buf, MPS_REQ_SIZE, 0);
  char *pbuf = buf;
  pbuf = serialize_uint64(pbuf, (uint64_t)devPtr);
  pbuf = serialize_uint64(pbuf, size);
  pbuf = serialize_uint32(pbuf, flags);
  send(client_socket, buf, buf_size, 0);
  recv(client_socket, buf, MPS_RES_SIZE, 0);
  struct mps_res res;
  deserialize_mps_res(buf, &res);
  return res.type;
}

