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
#include "mps.h"
#include "serialize.h"

int main(int argc, char **argv) {
  int client_socket;
  struct sockaddr_un server;

  client_socket = socket(AF_UNIX, SOCK_STREAM, 0);
  if (client_socket < 0) {
    printf("opening client socket: %s\n", strerror(errno));
    exit(-1);
  }
  server.sun_family = AF_UNIX;
  strcpy(server.sun_path, SERVER_SOCKET_FILE);
  if (connect(client_socket, (struct sockaddr *) &server, sizeof(struct sockaddr_un)) < 0) {
    close(client_socket);
    perror("conencting server socket");
    exit(-1);
  }
  struct mps_req req;
  req.type = REQ_GPU_LAUNCH_KERNEL;
  struct kernel_args kargs;
  kargs.arg_info[0] = (void *)3;
  kargs.arg_info[1] = (void *)0;
  kargs.arg_info[2] = (void *)2;
  kargs.arg_info[3] = (void *)10;
  *(uint16_t *)kargs.args = 59;
  *(uint64_t *)((uint8_t *)kargs.args+2) = 6;
  *(uint16_t *)((uint8_t *)kargs.args+10) = 7;
  kargs.last_arg_len = 2;
  kargs.blocks_per_grid = 2;
  kargs.threads_per_block = 32;
  kargs.function_index = 233;
  int nbytes = kernel_args_bytes(kargs);
  printf("kargs nbytes = %d\n", nbytes);
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
  struct mps_res res;
  deserialize_mps_res(buf, &res);
  if (res.type == RES_OK) {
    printf("kernel launch success\n");
  }
  free(buf);
  close(client_socket);
}
