/**
 * This is just a toy implementation, no any kinds of protection against exploitations.
 */
#include "serialize.h"

unsigned char* serialize_int(unsigned char *buffer, int value) {
  char *p = (char *)&value;
  buffer[0] = p[0];
  buffer[1] = p[1];
  buffer[2] = p[2];
  buffer[3] = p[3];
  return buffer + 4;
}
unsigned char* serialize_str(unsigned char *buffer, char *str, int len) {
  for (int i = 0; i < len; i++) {
    buffer[i] = str[i];
  }
  return buffer + len;
}
unsigned char* deserialize_int(unsigned char *buffer, int *value) {
  char *p = (char *)value;
  p[0] = buffer[0];
  p[1] = buffer[1];
  p[2] = buffer[2];
  p[3] = buffer[3];
  return buffer + 4;
}
unsigned char* deserialize_str(unsigned char *buffer, char *str, int len) {
  for (int i = 0; i < len; i++) {
    str[i] = buffer[i];
  }
  return buffer + len;
}
