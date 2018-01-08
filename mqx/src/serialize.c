/**
 * This is just a toy implementation, no any kinds of protection against exploitations.
 */
#include "serialize.h"
#include "common.h"

unsigned char* serialize_uint32(unsigned char *buffer, const uint32_t value) {
  buffer[0] = value >> 24;
  buffer[1] = value >> 16;
  buffer[2] = value >> 8;
  buffer[3] = value;
  return buffer + 4;
}
unsigned char* deserialize_uint32(unsigned char* const buffer, uint32_t *value) {
  *value = (buffer[0] << 24) + (buffer[1] << 16) + (buffer[2] << 8) + buffer[3];
  return buffer + 4;
}
unsigned char* serialize_uint64(unsigned char *buffer, const uint64_t value) {
  buffer[0] = value >> 56;
  buffer[1] = value >> 48;
  buffer[2] = value >> 40;
  buffer[3] = value >> 32;
  buffer[4] = value >> 24;
  buffer[5] = value >> 16;
  buffer[6] = value >> 8;
  buffer[7] = value;
  return buffer + 8;
}
unsigned char* deserialize_uint64(unsigned char* const buffer, uint64_t *value) {
  uint64_t tmp = (uint64_t)buffer[0] << 56;
  tmp += (uint64_t)buffer[1] << 48;
  tmp += (uint64_t)buffer[2] << 40;
  tmp += (uint64_t)buffer[3] << 32;
  tmp += (uint64_t)buffer[4] << 24;
  tmp += (uint64_t)buffer[5] << 16;
  tmp += (uint64_t)buffer[6] << 8;
  tmp += (uint64_t)buffer[7];
  *value = tmp;
  return buffer + 8;
}
unsigned char* serialize_str(unsigned char *buffer, const char* const str, int len) {
  for (int i = 0; i < len; i++) {
    buffer[i] = str[i];
  }
  return buffer + len;
}
unsigned char* deserialize_str(unsigned char* const buffer, char *str, int len) {
  for (int i = 0; i < len; i++) {
    str[i] = buffer[i];
  }
  return buffer + len;
}
unsigned char* serialize_kernel_args(unsigned char *buffer, const struct kernel_args kargs) {
  unsigned char *pbuf = buffer;
  int len, tmp;
  uint64_t nargs = (uint64_t)(kargs.arg_info[0]);
  pbuf = serialize_uint64(pbuf, nargs);
  len = 0;
  for (int i = 1; i < nargs + 1; i++) {
    tmp = (uint64_t)(kargs.arg_info[i]);
    pbuf = serialize_uint64(pbuf, tmp);
    len += tmp;
  }
  mqx_print(DEBUG, "len = %d", len);
  pbuf = serialize_str(pbuf, kargs.args, len);
  return pbuf;
}

unsigned char* deserialize_kernel_args(unsigned char* const buffer, struct kernel_args *kargs) {
  int len = 0;
  uint64_t *tmp;
  unsigned char *pbuf = buffer;
  pbuf = deserialize_uint64(pbuf, (uint64_t *)(&kargs->arg_info[0]));
  uint64_t nargs = (uint64_t)(kargs->arg_info[0]);
  for (int i = 1; i < nargs + 1; i++) {
    tmp = (uint64_t *)(&kargs->arg_info[i]);
    pbuf = deserialize_uint64(pbuf, tmp);
    len += *tmp;
  }
  mqx_print(DEBUG, "len = %d", len);
  pbuf = deserialize_str(pbuf, kargs->args, len);
  return pbuf;
}

int kernel_args_bytes(const struct kernel_args kargs) {
  uint64_t nargs = (uint64_t)(kargs.arg_info[0]);
  int len = 0;
  for (int i = 0; i < nargs; i++) {
    len += (uint64_t)(kargs.arg_info[i+1]);
  }
  return (nargs + 1) * sizeof(void *) + len;
}
