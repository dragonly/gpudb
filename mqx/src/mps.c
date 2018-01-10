#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>
#include <errno.h>
#include <pthread.h>
#include <cuda_runtime_api.h>
#include "common.h"
#include "protocol.h"
#include "serialize.h"
#include "mps.h"

// TODO: check all kinds of pointers
// TODO: statistics

static int client_socket;
extern struct global_context *pglobal;

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
  return 0;
}
cudaError_t mpsclient_cudaMalloc(void **devPtr, size_t size, uint32_t flags) {
  struct mps_req req;
  req.type = REQ_GPU_MALLOC;
  int buf_size = max(sizeof(void **) + sizeof(size_t) + sizeof(uint32_t), MPS_REQ_SIZE);
  unsigned char *buf = malloc(buf_size);
  serialize_mps_req(buf, req);
  send(client_socket, buf, MPS_REQ_SIZE, 0);
  unsigned char *pbuf = buf;
  pbuf = serialize_uint64(pbuf, (uint64_t)devPtr);
  pbuf = serialize_uint64(pbuf, size);
  pbuf = serialize_uint32(pbuf, flags);
  send(client_socket, buf, buf_size, 0);
  recv(client_socket, buf, 4, 0);
  cudaError_t ret;
  deserialize_uint32(buf, &ret);
  return ret;
}
cudaError_t mps_cudaMalloc(void **devPtr, size_t size, uint32_t flags) {
  if (size == 0) {
    mqx_print(WARN, "allocating 0 bytes");
  } else if (size > pglobal->mem_total) {
    return cudaErrorMemoryAllocation;
  }
  mqx_print(DEBUG, "allocate %zu bytes, %zu bytes free", size, pglobal->mem_total);

  struct mps_region *rgn;
  rgn = (struct mps_region *)calloc(1, sizeof(struct mps_region));
  rgn->swap_addr = malloc(size);
  // TODO: FLAG_PTARRAY
  rgn->nblocks = NBLOCKS(size);
  rgn->blocks = (struct mps_block *)calloc(rgn->nblocks, sizeof(struct mps_block));
  rgn->size = size;
  rgn->state = DETACHED;
  rgn->flags = flags;
  add_allocated_region(rgn);
  *devPtr = rgn->swap_addr;
  return cudaSuccess;
}

/**
 * helper functions
 */
void add_allocated_region(struct mps_region *rgn) {
  pthread_mutex_lock(&pglobal->alloc_mutex);
  list_add(&rgn->entry_alloc, &pglobal->allocated_regions);
  pthread_mutex_unlock(&pglobal->alloc_mutex);
}
