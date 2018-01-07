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
#include "common.h"
#include "protocol.h"
#include "mps.h"
#include "drvapi_error_string.h"
#include "kernel_symbols.h"

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
  unlink(SERVER_PATH);
  exit(0);
}


static CUcontext context;
static CUmodule mod_ops;
static CUfunction f_groupBy;
const char *mod_file_name = "ops.cubin";

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

  checkCudaErrors(cuCtxCreate(&context, 0, device));
  mqx_print(DEBUG, "CUDA context created");
  
  mqx_print(DEBUG, "testing load functions from cubin module");
  checkCudaErrors(cuModuleLoad(&mod_ops, mod_file_name));
  checkCudaErrors(cuModuleGetFunction(&f_groupBy, mod_ops, f_table[128]));
  mqx_print(DEBUG, "module: %s(%p), function: %s(%p)", mod_file_name, mod_ops, f_table[128], f_groupBy);
}

#ifdef STANDALONE
int main(int argc, char **argv) {
#else
void server_main() {
#endif
  initCUDA();
  mqx_print(DEBUG, "early exit for debugging");
  exit(0);

  int server_socket;
  struct sockaddr_un server;
  char buf[MAX_BUFFER_SIZE];

  signal(SIGINT, sigint_handler);
  signal(SIGKILL, sigint_handler);

  server_socket = socket(AF_UNIX, SOCK_STREAM, 0);
  if (socket < 0) {
    mqx_print(DEBUG, "opening server socket: %s", strerror(errno));
    exit(-1);
  }
  server.sun_family = AF_UNIX;
  strcpy(server.sun_path, SERVER_PATH);
  if (bind(server_socket, (struct sockaddr *) &server, sizeof(struct sockaddr_un))) {
    mqx_print(DEBUG, "binding server socket: %s", strerror(errno));
    exit(-1);
  }
  listen(server_socket, 128);

  int client_socket, rret;
  while (1) {
    client_socket = accept(server_socket, NULL, NULL);
    if (client_socket == -1) {
      mqx_print(DEBUG, "accept: %s", strerror(errno));
    } else {
      do {
        memset(buf, 0, MAX_BUFFER_SIZE);
        rret = read(client_socket, buf, MAX_BUFFER_SIZE);
        if (rret < 0) {
          mqx_print(DEBUG, "reading client socket: %s", strerror(errno));
        } else if (rret > 0) {
          mqx_print(DEBUG, "-> %s", buf);
        } else {
          mqx_print(DEBUG, "End of connection");
        }
      } while (rret > 0);
    }
  }
  close(server_socket);
  unlink(SERVER_PATH);
}
