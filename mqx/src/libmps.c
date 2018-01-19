#include <stdint.h>
#include <sys/socket.h>
#include <errno.h>
#include <string.h>
#include "common.h"

int send_large_buf(int socket, unsigned char *buf, uint32_t size) {
  unsigned char *pbuf = buf;
  uint32_t sent, rest_size;
  rest_size = size;
  do {
    sent = send(socket, pbuf, rest_size, 0);
    if (sent < 0) {
      mqx_print(ERROR, "sending to server socket: %s", strerror(errno));
      return -1;
    } else if (sent > 0) {
      pbuf += sent;
      rest_size -= sent;
    };
  } while (rest_size > 0);
  return 0;
}
int recv_large_buf(int socket, unsigned char *buf, uint32_t size) {
  unsigned char *pbuf = buf;
  uint32_t recved, rest_size;
  rest_size = size;
  do {
    recved = recv(socket, pbuf, rest_size, 0);
    if (recved < 0) {
      mqx_print(ERROR, "receiving from client socket: %s", strerror(errno));
      return -1;
    } else if (recved > 0) {
      pbuf += recved;
      rest_size -= recved;
    };
  } while (rest_size > 0);
  return 0;
}
