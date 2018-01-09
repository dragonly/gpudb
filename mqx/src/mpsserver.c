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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <cuda.h>
#include <pthread.h>
#include "common.h"
#include "protocol.h"
#include "mps.h"
#include "drvapi_error_string.h"
#include "kernel_symbols.h"
#include "serialize.h"

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
static inline void __checkCudaErrors( CUresult err, const char *file, const int line )
{
  if (CUDA_SUCCESS != err) {
    mqx_print(FATAL, "CUDA Driver API error = %04d  \"%s\" from file <%s>, line %i.",
            err, getCudaDrvErrorString(err), file, line );
    exit(-1);
  }
}

void sigint_handler(int signum) {
  mqx_print(DEBUG, "closing server...");
  unlink(SERVER_SOCKET_FILE);
  exit(0);
}

const char *mod_file_name = "ops.cubin";
static CUcontext cudaContext;
static CUmodule mod_ops;

static CUmodule test_mod_ops;
static CUfunction F_simpleAssert;

void initCUDA() {
  int device_count = 0;
  CUdevice device;
  checkCudaErrors(cuInit(0));
  checkCudaErrors(cuDeviceGetCount(&device_count));
  if (device_count == 0) {
    mqx_print(FATAL, "no devices supporting CUDA");
    exit(-1);
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

  // load the real SHIT
  checkCudaErrors(cuModuleLoad(&mod_ops, mod_file_name));
  for (int i = 0; i < NUMFUNC; i++) {
    checkCudaErrors(cuModuleGetFunction(&fsym_table[i], mod_ops, fname_table[i]));
    //mqx_print(DEBUG, "loaded module: %s(%p), function: %s(%p)", mod_file_name, mod_ops, fname_table[i], fsym_table[i]);
  }
  mqx_print(DEBUG, "CUDA module and functions loaded");
}

void *worker_thread(void *socket);

pthread_mutex_t stats_lock;
struct server_stats server_stats;

#ifdef STANDALONE
int main(int argc, char **argv) {
#else
void server_main() {
#endif
  initCUDA();
  //mqx_print(DEBUG, "early exit for debugging");
  //exit(0);

  server_stats.clients_bitmap = 0;
  server_stats.nclients = 0;
  int server_socket;
  struct sockaddr_un server;

  signal(SIGINT, sigint_handler);
  signal(SIGKILL, sigint_handler);

  server_socket = socket(AF_UNIX, SOCK_STREAM, 0);
  if (socket < 0) {
    mqx_print(DEBUG, "opening server socket: %s", strerror(errno));
    exit(-1);
  }
  server.sun_family = AF_UNIX;
  strcpy(server.sun_path, SERVER_SOCKET_FILE);
  if (bind(server_socket, (struct sockaddr *) &server, sizeof(struct sockaddr_un))) {
    mqx_print(DEBUG, "binding server socket: %s", strerror(errno));
    exit(-1);
  }
  listen(server_socket, 128);

  pthread_mutex_init(&stats_lock, NULL);
  int client_socket, *new_socket;
  while (1) {
    client_socket = accept(server_socket, NULL, NULL);
    if (client_socket == -1) {
      mqx_print(ERROR, "accept: %s", strerror(errno));
      continue;
    }
    if (server_stats.nclients >= MAX_CLIENTS) {
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
  pthread_mutex_destroy(&stats_lock);
  close(server_socket);
  unlink(SERVER_SOCKET_FILE);
  checkCudaErrors(cuCtxDestroy(cudaContext));
}

void *worker_thread(void *client_socket) {
  checkCudaErrors(cuCtxSetCurrent(cudaContext));

  pthread_mutex_lock(&stats_lock);
  server_stats.nclients += 1;
  int client_id = -1;
  for (int i = 0; i < server_stats.nclients; i++) {
    if ((server_stats.clients_bitmap & (1 << i)) == 0) {
      client_id = i;
      server_stats.clients_bitmap |= (1 << i);
      break;
    }
  }
  struct mps_client *cl = &server_stats.clients[client_id];
  cl->id = client_id;
  checkCudaErrors(cuStreamCreate(&cl->stream, CU_STREAM_DEFAULT));
  pthread_mutex_unlock(&stats_lock);
  mqx_print(DEBUG, "worker thread created (%d/%d)", client_id, server_stats.nclients);
  
  int rret;
  int socket = *(int *)client_socket;
  unsigned char buf[MAX_BUFFER_SIZE];
  memset(buf, 0, MAX_BUFFER_SIZE);
  struct mps_req req;
  struct mps_res res;
  uint16_t len;
  unsigned char *pbuf;
  struct kernel_args kargs;
  uint64_t nargs, offset; 

  while (1) {
    rret = recv(socket, buf, 4 /*sizeof serialized struct mps_req*/, 0);
    deserialize_mps_req(buf, &req);
    memset(buf, 0, 4);
    len = req.len;
    pbuf = buf;
    do {
      rret = recv(socket, pbuf, len, 0);
      if (rret < 0) {
        mqx_print(ERROR, "reading client socket: %s", strerror(errno));
      } else if (rret > 0) {
        pbuf += rret;
        len -= rret;
      } else {
        mqx_print(DEBUG, "End of connection");
      }
    } while (len > 0);
    switch (req.type) {
      case REQ_GPU_LAUNCH_KERNEL:;
        deserialize_kernel_args(buf, &kargs);
        nargs = (uint64_t)kargs.arg_info[0];

        for (int i = 0; i < nargs; i++) {
          offset = (uint64_t)kargs.arg_info[i+1];
          kargs.arg_info[i+1] = (void *)((uint8_t *)kargs.args + offset);
        }
        // test kernel
        if (kargs.function_index == 233) {
          mqx_print(DEBUG, "F_%d<<<%d, %d>>>(%zu args)", kargs.function_index, kargs.blocks_per_grid, kargs.threads_per_block, nargs);
          checkCudaErrors(cuLaunchKernel(F_simpleAssert,
                kargs.blocks_per_grid, 1, 1,
                kargs.threads_per_block, 1, 1,
                0, cl->stream, (void **)&(kargs.arg_info[1]), 0));
        } else {
          mqx_print(DEBUG, "%s[%d]<<<%d, %d>>>(%zu args)", fname_table[kargs.function_index], kargs.function_index, kargs.blocks_per_grid, kargs.threads_per_block, nargs);
          checkCudaErrors(cuLaunchKernel(fsym_table[kargs.function_index],
                kargs.blocks_per_grid, 1, 1,
                kargs.threads_per_block, 1, 1,
                0, cl->stream, (void **)&(kargs.arg_info[1]), 0));
        }
        res.type = RES_OK;
        serialize_mps_res(buf, res);
        send(socket, buf, 2, 0);
        break;
      default:
        mqx_print(FATAL, "no such type: %d", req.type);
        goto finish;
    }
  }

finish:
  pthread_mutex_lock(&stats_lock);
  server_stats.nclients -= 1;
  server_stats.clients_bitmap &= ~(1 << client_id);
  cl->id = -1;
  checkCudaErrors(cuStreamDestroy(cl->stream));
  pthread_mutex_unlock(&stats_lock);

  free(client_socket);
  close(socket);
  checkCudaErrors(cuCtxPopCurrent(&cudaContext));
  mqx_print(DEBUG, "living threads: %d", server_stats.nclients);
  return NULL;
}
