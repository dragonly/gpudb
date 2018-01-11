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
#include "list.h"

const char *mod_file_name = "ops.cubin";
static CUcontext cudaContext;
static CUmodule mod_ops;

static CUmodule test_mod_ops;
static CUfunction F_simpleAssert;

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

  // load test module: simpleAssert.cubin
  checkCudaErrors(cuModuleLoad(&test_mod_ops, "simpleAssert.cubin"));
  checkCudaErrors(cuModuleGetFunction(&F_simpleAssert, test_mod_ops, "testKernel"));

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
  pthread_mutex_destroy(&pglobal->mps_lock);
  unlink(SERVER_SOCKET_FILE);
  close(shmfd);
  checkCudaErrors(cuCtxDestroy(cudaContext));
  exit(0);
}

int main(int argc, char **argv) {
  if (initCUDA() == -1) goto fail_cuda;
  //mqx_print(DEBUG, "early exit for debugging");
  //exit(0);

  shmfd = shm_open(MQX_SHM_GLOBAL, O_RDWR, 0);
  if (shmfd == -1) {
    mqx_print(FATAL, "Failed to open shared memory: %s.", strerror(errno));
      goto fail_shm;
  }
  pglobal = (struct global_context *)mmap(NULL /* address */, sizeof(struct global_context), PROT_READ | PROT_WRITE, MAP_SHARED, shmfd, 0 /* offset */);
  if (pglobal == MAP_FAILED) {
    mqx_print(FATAL, "Failed to mmap shared memory: %s.", strerror(errno));
    goto fail_mmap;
  }

  pglobal->mps_clients_bitmap = 0;
  pglobal->mps_nclients = 0;
  int server_socket;
  struct sockaddr_un server;

  signal(SIGINT, sigint_handler);
  signal(SIGKILL, sigint_handler);

  server_socket = socket(AF_UNIX, SOCK_STREAM, 0);
  if (socket < 0) {
    mqx_print(DEBUG, "opening server socket: %s", strerror(errno));
    goto fail_create_socket;
  }
  server.sun_family = AF_UNIX;
  strcpy(server.sun_path, SERVER_SOCKET_FILE);
  if (bind(server_socket, (struct sockaddr *) &server, sizeof(struct sockaddr_un))) {
    mqx_print(DEBUG, "binding server socket: %s", strerror(errno));
    goto fail_bind;
  }
  listen(server_socket, 128);

  pthread_mutex_init(&pglobal->mps_lock, NULL);
  pthread_mutex_init(&pglobal->alloc_mutex, NULL);
  INIT_LIST_HEAD(&pglobal->allocated_regions);
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

  pthread_mutex_destroy(&pglobal->mps_lock);
  pthread_mutex_destroy(&pglobal->alloc_mutex);
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

void *worker_thread(void *client_socket) {
  checkCudaErrors(cuCtxSetCurrent(cudaContext));

  pthread_mutex_lock(&pglobal->mps_lock);
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
  struct mps_client *cl = &pglobal->mps_clients[client_id];
  cl->id = client_id;
  checkCudaErrors(cuStreamCreate(&cl->stream, CU_STREAM_DEFAULT));
  pthread_mutex_unlock(&pglobal->mps_lock);
  mqx_print(DEBUG, "worker thread created (%d/%d)", client_id, pglobal->mps_nclients);
  
  int rret;
  int socket = *(int *)client_socket;
  unsigned char buf[MAX_BUFFER_SIZE];
  memset(buf, 0, MAX_BUFFER_SIZE);
  struct mps_req req;
  uint16_t len;
  unsigned char *pbuf;

  while (1) {
    // read request header
    rret = recv(socket, buf, 4 /*sizeof serialized struct mps_req*/, 0);
    deserialize_mps_req(buf, &req);
    if (req.type == REQ_QUIT) {
      mqx_print(DEBUG, "client quit");
      goto finish;
    }
    memset(buf, 0, 4);
    len = req.len;
    pbuf = buf;
    // read the real shit
    do {
      rret = recv(socket, pbuf, len, 0);
      if (rret < 0) {
        mqx_print(ERROR, "reading client socket: %s", strerror(errno));
      } else if (rret > 0) {
        pbuf += rret;
        len -= rret;
      };
    } while (len > 0);
    // at your service
    switch (req.type) {
      case REQ_GPU_MALLOC: {
        pbuf = buf;
        void *devPtr;
        size_t size;
        uint32_t flags;
        pbuf = deserialize_uint64(pbuf, (uint64_t *)&size);
        pbuf = deserialize_uint32(pbuf, &flags);
        cudaError_t ret = mpsserver_cudaMalloc(&devPtr, size, flags);
        pbuf = buf;
        pbuf = serialize_uint32(pbuf, ret);
        pbuf = serialize_uint64(pbuf, (uint64_t)devPtr);
        mqx_print(DEBUG, "server cudaMalloc: devPtr(%p), ret(%d)", devPtr, ret);
        send(socket, buf, 4+8, 0);
      } break;
      case REQ_GPU_MEMFREE: {
        void *devPtr;
        deserialize_uint64(buf, (uint64_t *)&devPtr);
        cudaError_t ret = mpsserver_cudaFree(devPtr);
        serialize_uint32(buf, ret);
        send(socket, buf, 4, 0);
      } break;
      case REQ_GPU_LAUNCH_KERNEL: {
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
          ret = cuLaunchKernel(F_simpleAssert,
                kargs.blocks_per_grid, 1, 1,
                kargs.threads_per_block, 1, 1,
                0, cl->stream, (void **)&(kargs.arg_info[1]), 0);
        } else {
          mqx_print(DEBUG, "%s[%d]<<<%d, %d>>>(%zu args)", fname_table[kargs.function_index], kargs.function_index, kargs.blocks_per_grid, kargs.threads_per_block, nargs);
          ret = cuLaunchKernel(fsym_table[kargs.function_index],
                kargs.blocks_per_grid, 1, 1,
                kargs.threads_per_block, 1, 1,
                0, cl->stream, (void **)&(kargs.arg_info[1]), 0);
        }
        serialize_uint32(buf, ret);
        send(socket, buf, 4, 0);
      } break;
      default:
        mqx_print(FATAL, "no such type: %d", req.type);
        goto finish;
    }
  }

finish:
  pthread_mutex_lock(&pglobal->mps_lock);
  pglobal->mps_nclients -= 1;
  pglobal->mps_clients_bitmap &= ~(1 << client_id);
  cl->id = -1;
  checkCudaErrors(cuStreamDestroy(cl->stream));
  pthread_mutex_unlock(&pglobal->mps_lock);

  free(client_socket);
  close(socket);
  checkCudaErrors(cuCtxPopCurrent(&cudaContext));
  mqx_print(DEBUG, "living threads: %d", pglobal->mps_nclients);
  return NULL;
}
