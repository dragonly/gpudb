#include <stdint.h>
#include <stdio.h>
#include <cuda.h>
#include "interfaces.h"
#include "mps.h"
#include "common.h"

CUresult (*nv_cuMemFree)(CUdeviceptr) = NULL;

volatile uint8_t mps_initialized = 0;

__attribute__((constructor)) void mps_init(void) {
  DEFAULT_API_POINTER("cuMemFree", nv_cuMemFree);

  if (mps_client_init()) {
    mqx_print(FATAL, "fail to connect to mps server");
    return;
  }
  mps_initialized = 1;
}
