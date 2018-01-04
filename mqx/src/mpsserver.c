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
// #include <cuda_runtime_api.h>
#include "protocol.h"
#include "mps.h"

void sigint_handler(int signum) {
  printf("closing server...\n");
  unlink(SERVER_PATH);
  exit(0);
}

#ifdef STANDALONE
int main(int argc, char **argv) {
#else
void server_main() {
#endif
  int server_socket;
  struct sockaddr_un server;
  char buf[MAX_BUFFER_SIZE];

  signal(SIGINT, sigint_handler);
  signal(SIGKILL, sigint_handler);

  server_socket = socket(AF_UNIX, SOCK_STREAM, 0);
  if (socket < 0) {
    printf("opening server socket: %s\n", strerror(errno));
    exit(-1);
  }
  server.sun_family = AF_UNIX;
  strcpy(server.sun_path, SERVER_PATH);
  if (bind(server_socket, (struct sockaddr *) &server, sizeof(struct sockaddr_un))) {
    printf("binding server socket: %s\n", strerror(errno));
    exit(-1);
  }
  listen(server_socket, 128);

  int client_socket, rret;
  while (1) {
    client_socket = accept(server_socket, NULL, NULL);
    if (client_socket == -1) {
      printf("accept: %s\n", strerror(errno));
    } else {
      do {
        memset(buf, 0, MAX_BUFFER_SIZE);
        rret = read(client_socket, buf, MAX_BUFFER_SIZE);
        if (rret < 0) {
          printf("reading client socket: %s\n", strerror(errno));
        } else if (rret > 0) {
          printf("-> %s", buf);
        } else {
          printf("End of connection\n");
        }
      } while (rret > 0);
    }
  }
  close(server_socket);
  unlink(SERVER_PATH);
}