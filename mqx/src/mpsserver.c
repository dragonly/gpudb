/*
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

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <pthread.h>
#include "common.h"
#include "protocol.h"
#include "kernel_symbols.h"
#include "serialize.h"
#include "libmpsserver.h"
#include "libmps.h"
#include "list.h"

const char *mod_file_name = "ops.cubin";
static CUcontext cudaContext;
static CUmodule mod_ops;

static CUmodule test_mod_ops;
CUfunction F_vectorAdd;

int initCUDA() {
  int device_count = 0;
  CUdevice device;
  checkCudaErrors(cuInit(0));
  checkCudaErrors(cuDeviceGetCount(&device_count));
  if (device_count == 0) {
    mqx_print(FATAL, "no devices supporting CUDA");
    return -1;
  }
  mqx_print(DEBUG, "device count: %d", device_count);
  checkCudaErrors(cuDeviceGet(&device, 0));
  char name[100];
  checkCudaErrors(cuDeviceGetName(name, 100, device));
  mqx_print(DEBUG, "Using device 0: %s", name);

  checkCudaErrors(cuCtxCreate(&cudaContext, 0, device));
  mqx_print(DEBUG, "CUDA context created");
  
  mqx_print(DEBUG, "loading functions from cubin module");

  // load test module: vectorAdd.cubin
  // NOTE: CUDA_ERROR_INVALID_SOURCE indicates that no -arch=sm_xx provided
  checkCudaErrors(cuModuleLoad(&test_mod_ops, "vectorAdd.cubin"));
  checkCudaErrors(cuModuleGetFunction(&F_vectorAdd, test_mod_ops, "vectorAdd"));
  mqx_print(DEBUG, "vectorAdd: %p", F_vectorAdd);

  // load the real shit
  checkCudaErrors(cuModuleLoad(&mod_ops, mod_file_name));
  for (int i = 0; i < NUMFUNC; i++) {
    checkCudaErrors(cuModuleGetFunction(&fsym_table[i], mod_ops, fname_table[i]));
    //mqx_print(DEBUG, "loaded module: %s(%p), function: %s(%p)", mod_file_name, mod_ops, fname_table[i], fsym_table[i]);
  }
  mqx_print(DEBUG, "CUDA module and functions loaded");
  return 0;
}

void *worker_thread(void *socket);

int shmfd;
struct global_context *pglobal;

void sigint_handler(int signum) {
  mqx_print(DEBUG, "closing server...");
  pthread_mutex_destroy(&pglobal->client_mutex);
  unlink(SERVER_SOCKET_FILE);
  close(shmfd);
  checkCudaErrors(cuCtxDestroy(cudaContext));
  exit(0);
}

int main(int argc, char **argv) {
  if (initCUDA() == -1) goto fail_cuda;
  //mqx_print(DEBUG, "early exit for debugging");
  //exit(0);

  mqx_print(DEBUG, "opening shared memory");
  shmfd = shm_open(MQX_SHM_GLOBAL, O_RDWR, 0);
  if (shmfd == -1) {
    mqx_print(FATAL, "Failed to open shared memory: %s.", strerror(errno));
      goto fail_shm;
  }
  mqx_print(DEBUG, "mmaping global context");
  pglobal = (struct global_context *)mmap(NULL /* address */, sizeof(struct global_context), PROT_READ | PROT_WRITE, MAP_SHARED, shmfd, 0 /* offset */);
  if (pglobal == MAP_FAILED) {
    mqx_print(FATAL, "Failed to mmap shared memory: %s.", strerror(errno));
    goto fail_mmap;
  }
  mqx_print(DEBUG, "initializing global context");
  memset(pglobal->mps_clients, 0, sizeof(struct mps_client) * MAX_CLIENTS);
  for (int i = 0; i < MAX_CLIENTS; i++) {
    pglobal->mps_clients[i].id = -1;
  }
  pglobal->mps_clients_bitmap = 0;
  pglobal->mps_nclients = 0;
  pthread_mutex_init(&pglobal->client_mutex, NULL);
  pthread_mutex_init(&pglobal->alloc_mutex, NULL);
  pthread_mutex_init(&pglobal->kernel_launch_mutex, NULL);
  INIT_LIST_HEAD(&pglobal->allocated_regions);
  INIT_LIST_HEAD(&pglobal->attached_regions);

  int server_socket;
  struct sockaddr_un server;

  signal(SIGINT, sigint_handler);
  signal(SIGKILL, sigint_handler);

  mqx_print(DEBUG, "initializing server socket");
  server_socket = socket(AF_UNIX, SOCK_STREAM, 0);
  if (socket < 0) {
    mqx_print(DEBUG, "opening server socket: %s", strerror(errno));
    goto fail_create_socket;
  }
  server.sun_family = AF_UNIX;
  strcpy(server.sun_path, SERVER_SOCKET_FILE);
  if (bind(server_socket, (struct sockaddr *) &server, sizeof(struct sockaddr_un))) {
    mqx_print(FATAL, "binding server socket: %s", strerror(errno));
    goto fail_bind;
  }
  listen(server_socket, 128);

  mqx_print(DEBUG, "accepting client sockets");
  int client_socket, *new_socket;
  while (1) {
    client_socket = accept(server_socket, NULL, NULL);
    if (client_socket == -1) {
      mqx_print(ERROR, "accept: %s", strerror(errno));
      continue;
    }
    if (pglobal->mps_nclients >= MAX_CLIENTS) {
      mqx_print(ERROR, "max clients served (%d), connection rejected", MAX_CLIENTS);
      close(client_socket);
      continue;
    }
    mqx_print(INFO, "starting up worker thread to serve new connection");
    new_socket = malloc(sizeof(int));
    *new_socket = client_socket;
    pthread_t thread;
    if(pthread_create(&thread, NULL, worker_thread, (void *)new_socket)) {
      mqx_print(ERROR, "pthread_create: %s", strerror(errno));
      continue;
    }
  }

  memset(pglobal->mps_clients, 0, sizeof(struct mps_client) * MAX_CLIENTS);
  for (int i = 0; i < MAX_CLIENTS; i++) {
    pglobal->mps_clients[i].id = -1;
  }
  pglobal->mps_clients_bitmap = 0;
  pglobal->mps_nclients = 0;
  pthread_mutex_destroy(&pglobal->client_mutex);
  pthread_mutex_destroy(&pglobal->alloc_mutex);
  pthread_mutex_destroy(&pglobal->kernel_launch_mutex);
  struct list_head *pos;
  struct mps_region *rgn;
  list_for_each(pos, &pglobal->allocated_regions) {
    rgn = list_entry(pos, struct mps_region, entry_alloc);
    free(rgn->blocks);
    free(rgn);
  }
  list_for_each(pos, &pglobal->attached_regions) {
    rgn = list_entry(pos, struct mps_region, entry_attach);
    free(rgn->blocks);
    free(rgn);
  }
  unlink(SERVER_SOCKET_FILE);
fail_bind:
  close(server_socket);
fail_create_socket:
fail_mmap:
  close(shmfd);
fail_shm:
fail_cuda:
  checkCudaErrors(cuCtxDestroy(cudaContext));
}

// TODO: support cuda program using multiple streams
void *worker_thread(void *client_socket) {
  checkCudaErrors(cuCtxSetCurrent(cudaContext));

  pthread_mutex_lock(&pglobal->client_mutex);
  pglobal->mps_nclients += 1;
  int client_id = -1;
  int nclients = pglobal->mps_nclients;
  for (int i = 0; i < nclients; i++) {
    if ((pglobal->mps_clients_bitmap & (1 << i)) == 0) {
      client_id = i;
      pglobal->mps_clients_bitmap |= (1 << i);
      break;
    }
  }
  pthread_mutex_unlock(&pglobal->client_mutex);
  struct mps_client *client = &pglobal->mps_clients[client_id];
  client->id = client_id;
  checkCudaErrors(cuStreamCreate(&client->stream, CU_STREAM_DEFAULT));
  dma_channel_init(&client->dma_htod, 1);
  dma_channel_init(&client->dma_dtoh, 0);
  pthread_mutex_init(&client->dma_mutex, NULL);
  client->kconf.ktop = client->kconf.kstack;
  client->kconf.nargs = 0;
  client->kconf.nadvices = 0;
  client->kconf.func_index = -1;
  mqx_print(DEBUG, "++++++++++++++++ worker thread created (%d/%d) ++++++++++++++++", client_id + 1, pglobal->mps_nclients);
  
  int rret;
  int socket = *(int *)client_socket;
  unsigned char buf[MAX_BUFFER_SIZE];
  memset(buf, 0, MAX_BUFFER_SIZE);
  struct mps_req req;
  uint16_t len;
  unsigned char *pbuf;

  while (1) {
    // read request header
    rret = recv(socket, buf, MPS_REQ_SIZE, 0);
    deserialize_mps_req(buf, &req);
    if (req.type == REQ_QUIT) {
      mqx_print(DEBUG, "client quit");
      goto finish;
    }
    memset(buf, 0, MPS_REQ_SIZE);
    len = req.len;
    pbuf = buf;
    // read the real shit
    if (len > 0) {
      do {
        rret = recv(socket, pbuf, len, 0);
        if (rret < 0) {
          mqx_print(ERROR, "reading from client socket: %s", strerror(errno));
          goto finish;
        } else if (rret > 0) {
          pbuf += rret;
          len -= rret;
        };
      } while (len > 0);
    }
    // at your service
    switch (req.type) {
      case REQ_CUDA_MALLOC: {
        void *devPtr;
        size_t size;
        uint32_t flags;
        pbuf = buf;
        pbuf = deserialize_uint64(pbuf, (uint64_t *)&size);
        pbuf = deserialize_uint32(pbuf, &flags);
        cudaError_t ret = mpsserver_cudaMalloc(&devPtr, size, flags);
        pbuf = buf;
        pbuf = serialize_uint64(pbuf, (uint64_t)devPtr);
        pbuf = serialize_uint32(pbuf, ret);
        mqx_print(DEBUG, "cudaMalloc: devPtr(%p) size(%zu) ret(%d)", devPtr, size, ret);
        send(socket, buf, sizeof(devPtr) + sizeof(ret), 0);
      } break;
      case REQ_CUDA_MEMFREE: {
        void *devPtr;
        deserialize_uint64(buf, (uint64_t *)&devPtr);
        cudaError_t ret = mpsserver_cudaFree(devPtr);
        serialize_uint32(buf, ret);
        send(socket, buf, sizeof(ret), 0);
      } break;
      case REQ_CUDA_MEMCPY_HTOD: {
        void *dst;  // swap addr
        deserialize_uint64(buf, (uint64_t *)&dst);
        int round_size;
        cudaError_t ret;
        // FIXME: bug resides in multi-round situations
        mqx_print(DEBUG, "starting a (%lu bytes/%d round) Host->Swap memcpy", (req.round-1)*MAX_BUFFER_SIZE+req.last_len, req.round);
        for (int i = 0; i < req.round; i++) {
          round_size = i == req.round - 1 ? req.last_len : i * MAX_BUFFER_SIZE;
          if (recv_large_buf(socket, buf, round_size) != 0) {
            pthread_exit(NULL);
          }
          ret = mpsserver_cudaMemcpy(client, dst + i * MAX_BUFFER_SIZE, buf, round_size, cudaMemcpyHostToDevice);
          if (ret != cudaSuccess) {
            mqx_print(ERROR, "mpsserver_cudaMemcpy: HtoD failed");
            break;
          }
        }
        //uint64_t size = (req.round - 1) * MAX_BUFFER_SIZE + req.last_len;
        serialize_uint32(buf, ret);
        send(socket, buf, sizeof(ret), 0);
      } break;
      // not implemented yet
      case REQ_CUDA_MEMCPY_DTOH: {
        void *src;  // swap addr
        deserialize_uint64(buf, (uint64_t *)&src);
        int round_size;
        cudaError_t ret;
        mqx_print(DEBUG, "starting a (%lu bytes/%d round) Device->Host memcpy", (req.round-1)*MAX_BUFFER_SIZE+req.last_len, req.round);
        for (int i = 0; i < req.round; i++) {
          round_size = i == req.round - 1 ? req.last_len : i * MAX_BUFFER_SIZE;
          ret = mpsserver_cudaMemcpy(client, buf, src + i * MAX_BUFFER_SIZE, round_size, cudaMemcpyDeviceToHost);
          if (ret != cudaSuccess) {
            mqx_print(ERROR, "mpsserver_cudaMemcpy: DtoH failed");
            break;
          }
          if (send_large_buf(socket, buf, round_size) != 0) {
            pthread_exit(NULL);
          }
        }
        serialize_uint32(buf, ret);
        send(socket, buf, sizeof(ret), 0);
      } break;
      case REQ_CUDA_MEMCPY_DTOD:
      case REQ_CUDA_MEMCPY_HTOH:
      case REQ_CUDA_MEMCPY_DEFAULT: {
        serialize_uint32(buf, cudaErrorNotYetImplemented);
        send(socket, buf, sizeof(cudaError_t), 0);
      } break;
      case REQ_CUDA_ADVISE: {
        uint8_t iarg;
        uint8_t advice;
        uint8_t *pbuf = buf;
        pbuf = deserialize_str(pbuf, &iarg, 1);
        pbuf = deserialize_str(pbuf, &advice, 1);
        cudaError_t ret = mpsserver_cudaAdvise(client, iarg, advice);
        serialize_uint32(buf, ret);
        send(socket, buf, sizeof(ret), 0);
      } break;
      case REQ_SET_KERNEL_FUNCTION: {
        uint32_t index;
        uint8_t *pbuf = buf;
        pbuf = deserialize_uint32(pbuf, &index);
        cudaError_t ret = mpsserver_cudaSetFunction(client, index);
        serialize_uint32(buf, ret);
        send(socket, buf, sizeof(ret), 0);
      } break;
      case REQ_CUDA_CONFIGURE_CALL: {
        dim3 gridDim;
        dim3 blockDim;
        size_t sharedMem;
        // TODO: maintain a client-side stream ptr -> id map
        uint8_t *pbuf = buf;
        pbuf = deserialize_uint32(pbuf, &gridDim.x);
        pbuf = deserialize_uint32(pbuf, &gridDim.y);
        pbuf = deserialize_uint32(pbuf, &gridDim.z);
        pbuf = deserialize_uint32(pbuf, &blockDim.x);
        pbuf = deserialize_uint32(pbuf, &blockDim.y);
        pbuf = deserialize_uint32(pbuf, &blockDim.z);
        pbuf = deserialize_uint64(pbuf, &sharedMem);
        // TODO: using the only client CUstream for now
        cudaError_t ret = mpsserver_cudaConfigureCall(client, gridDim, blockDim, sharedMem, client->stream);
        serialize_uint32(buf, ret);
        send(socket, buf, sizeof(ret), 0);
      } break;
      case REQ_CUDA_SETUP_ARGUMENT: {
        uint8_t *parg;
        size_t size;
        size_t offset;
        // TODO: maintain a client-side stream ptr -> id map
        uint8_t *pbuf = buf;
        pbuf = deserialize_uint64(pbuf, &size);
        // parg is freed in `mpsserver_cudaSetupArgument`
        parg = malloc(size);
        pbuf = deserialize_str(pbuf, parg, size);
        pbuf = deserialize_uint64(pbuf, &offset);
        cudaError_t ret = mpsserver_cudaSetupArgument(client, parg, size, offset);
        serialize_uint32(buf, ret);
        send(socket, buf, sizeof(ret), 0);
      } break;
      case REQ_CUDA_LAUNCH_KERNEL: {
        cudaError_t ret = mpsserver_cudaLaunchKernel(client);
        serialize_uint32(buf, ret);
        send(socket, buf, 4, 0);
      } break;
      case REQ_TEST_CUDA_LAUNCH_KERNEL: {
        struct kernel_args kargs;
        uint64_t nargs, offset; 
        deserialize_kernel_args(buf, &kargs);
        nargs = (uint64_t)kargs.arg_info[0];
        for (int i = 0; i < nargs; i++) {
          offset = (uint64_t)kargs.arg_info[i+1];
          kargs.arg_info[i+1] = (void *)((uint8_t *)kargs.args + offset);
        }
        // test kernel
        cudaError_t ret;
        if (kargs.function_index == 233) {
          mqx_print(DEBUG, "F_%d<<<%d, %d>>>(%zu args)", kargs.function_index, kargs.blocks_per_grid, kargs.threads_per_block, nargs);
          ret = cuLaunchKernel(F_vectorAdd,
                kargs.blocks_per_grid, 1, 1,
                kargs.threads_per_block, 1, 1,
                0, client->stream, (void **)&(kargs.arg_info[1]), 0);
        } else {
          mqx_print(DEBUG, "%s[%d]<<<%d, %d>>>(%zu args)", fname_table[kargs.function_index], kargs.function_index, kargs.blocks_per_grid, kargs.threads_per_block, nargs);
          ret = cuLaunchKernel(fsym_table[kargs.function_index],
                kargs.blocks_per_grid, 1, 1,
                kargs.threads_per_block, 1, 1,
                0, client->stream, (void **)&(kargs.arg_info[1]), 0);
        }
        serialize_uint32(buf, ret);
        send(socket, buf, 4, 0);
      } break;
      default:
        mqx_print(FATAL, "type not implemented: %d", req.type);
        goto finish;
    }
  }

finish:
  pthread_mutex_lock(&pglobal->client_mutex);
  pglobal->mps_nclients -= 1;
  pglobal->mps_clients_bitmap &= ~(1 << client_id);
  client->id = -1;
  checkCudaErrors(cuStreamDestroy(client->stream));
  dma_channel_destroy(&client->dma_htod);
  dma_channel_destroy(&client->dma_dtoh);
  pthread_mutex_destroy(&client->dma_mutex);
  pthread_mutex_unlock(&pglobal->client_mutex);

  free(client_socket);
  close(socket);
  checkCudaErrors(cuCtxPopCurrent(&cudaContext));
  mqx_print(DEBUG, "++++++++++++++++ worker thread exit (%d/%d) ++++++++++++++++", client_id, pglobal->mps_nclients);
  return NULL;
}
