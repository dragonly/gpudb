/**
 * This is just a toy implementation, no any kinds of protection against exploitations.
 */
#include <stdio.h>
#include "serialize.h"

unsigned char* serialize_uint16(unsigned char *buffer, const uint16_t value) {
  buffer[0] = value >> 8;
  buffer[1] = value;
  return buffer + 2;
}
unsigned char* deserialize_uint16(unsigned char* const buffer, uint16_t *value) {
  uint16_t tmp = (uint16_t)buffer[0] << 8;
  tmp += (uint16_t)buffer[1];
  *value = tmp;
  return buffer + 2;
}
unsigned char* serialize_uint32(unsigned char *buffer, const uint32_t value) {
  buffer[0] = value >> 24;
  buffer[1] = value >> 16;
  buffer[2] = value >> 8;
  buffer[3] = value;
  return buffer + 4;
}
unsigned char* deserialize_uint32(unsigned char* const buffer, uint32_t *value) {
  uint32_t tmp = (uint32_t)buffer[0] << 24;
  tmp += (uint32_t)buffer[1] << 16;
  tmp += (uint32_t)buffer[2] << 8;
  tmp += (uint32_t)buffer[3];
  *value = tmp;
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
  uint64_t nargs = (uint64_t)(kargs.arg_info[0]);
  pbuf = serialize_uint64(pbuf, nargs);
  for (int i = 1; i < nargs + 1; i++) {
    pbuf = serialize_uint64(pbuf, (uint64_t)(kargs.arg_info[i]));
  }
  pbuf = serialize_uint16(pbuf, kargs.last_arg_len);
  pbuf = serialize_str(pbuf, kargs.args, (uint64_t)kargs.arg_info[nargs] + kargs.last_arg_len);
  pbuf = serialize_uint16(pbuf, kargs.threads_per_block);
  pbuf = serialize_uint16(pbuf, kargs.blocks_per_grid);
  pbuf = serialize_uint16(pbuf, kargs.function_index);
  return pbuf;
}
unsigned char* deserialize_kernel_args(unsigned char* const buffer, struct kernel_args *kargs) {
  unsigned char *pbuf = buffer;
  pbuf = deserialize_uint64(pbuf, (uint64_t *)(&kargs->arg_info[0]));
  uint64_t nargs = (uint64_t)(kargs->arg_info[0]);
  for (int i = 1; i < nargs + 1; i++) {
    pbuf = deserialize_uint64(pbuf, (uint64_t *)(&kargs->arg_info[i]));
  }
  pbuf = deserialize_uint16(pbuf, &kargs->last_arg_len);
  pbuf = deserialize_str(pbuf, kargs->args, (uint64_t)kargs->arg_info[nargs] + kargs->last_arg_len);
  pbuf = deserialize_uint16(pbuf, &kargs->threads_per_block);
  pbuf = deserialize_uint16(pbuf, &kargs->blocks_per_grid);
  pbuf = deserialize_uint16(pbuf, &kargs->function_index);
  return pbuf;
}
int kernel_args_bytes(const struct kernel_args kargs) {
  uint64_t nargs = (uint64_t)(kargs.arg_info[0]);
  int offset = 0;
  for (int i = 0; i < nargs; i++) {
    offset = (uint64_t)(kargs.arg_info[i+1]);
  }
  return (nargs + 1) * sizeof(void *) + offset + kargs.last_arg_len + 8;
}
unsigned char* serialize_mps_req(unsigned char *buffer, const struct mps_req req) {
  unsigned char *pbuf = buffer;
  pbuf = serialize_uint16(pbuf, req.type);
  pbuf = serialize_uint16(pbuf, req.len);
  return pbuf;
}
unsigned char* deserialize_mps_req(unsigned char* const buffer, struct mps_req *req) {
  unsigned char *pbuf = buffer;
  pbuf = deserialize_uint16(pbuf, &req->type);
  pbuf = deserialize_uint16(pbuf, &req->len);
  return pbuf;
}
