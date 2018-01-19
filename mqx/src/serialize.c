/**
 * This is just a toy implementation, no any kinds of protection against exploitations.
 */
#include <stdio.h>
#include "protocol.h"

uint8_t* serialize_uint16(uint8_t *buffer, const uint16_t value) {
  buffer[0] = value >> 8;
  buffer[1] = value;
  return buffer + 2;
}
uint8_t* deserialize_uint16(uint8_t *const buffer, uint16_t *value) {
  uint16_t tmp = (uint16_t)buffer[0] << 8;
  tmp += (uint16_t)buffer[1];
  *value = tmp;
  return buffer + 2;
}
uint8_t* serialize_uint32(uint8_t *buffer, const uint32_t value) {
  buffer[0] = value >> 24;
  buffer[1] = value >> 16;
  buffer[2] = value >> 8;
  buffer[3] = value;
  return buffer + 4;
}
uint8_t* deserialize_uint32(uint8_t *const buffer, uint32_t *value) {
  uint32_t tmp = (uint32_t)buffer[0] << 24;
  tmp += (uint32_t)buffer[1] << 16;
  tmp += (uint32_t)buffer[2] << 8;
  tmp += (uint32_t)buffer[3];
  *value = tmp;
  return buffer + 4;
}
uint8_t* serialize_uint64(uint8_t *buffer, const uint64_t value) {
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
uint8_t* deserialize_uint64(uint8_t *const buffer, uint64_t *value) {
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
uint8_t* serialize_str(uint8_t *buffer, const uint8_t *const str, int len) {
  for (int i = 0; i < len; i++) {
    buffer[i] = str[i];
  }
  return buffer + len;
}
uint8_t* deserialize_str(uint8_t *const buffer, uint8_t *str, int len) {
  for (int i = 0; i < len; i++) {
    str[i] = buffer[i];
  }
  return buffer + len;
}
uint8_t* serialize_kernel_args(uint8_t *buffer, const struct kernel_args kargs) {
  uint8_t *pbuf = buffer;
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
uint8_t* deserialize_kernel_args(uint8_t *const buffer, struct kernel_args *kargs) {
  uint8_t *pbuf = buffer;
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
uint8_t* serialize_mps_req(uint8_t *buffer, const struct mps_req req) {
  uint8_t *pbuf = buffer;
  pbuf = serialize_uint16(pbuf, req.type);
  pbuf = serialize_uint32(pbuf, req.len);
  pbuf = serialize_uint16(pbuf, req.round);
  pbuf = serialize_uint64(pbuf, req.last_len);
  return pbuf;
}
uint8_t* deserialize_mps_req(uint8_t *const buffer, struct mps_req *req) {
  uint8_t *pbuf = buffer;
  pbuf = deserialize_uint16(pbuf, &req->type);
  pbuf = deserialize_uint32(pbuf, &req->len);
  pbuf = deserialize_uint16(pbuf, &req->round);
  pbuf = deserialize_uint64(pbuf, &req->last_len);
  return pbuf;
}

