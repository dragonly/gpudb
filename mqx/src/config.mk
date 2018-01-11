# Printing flags
USE_PRINT_BUFFER = 0
PRINT_LEVEL = 5

MQX_CONFIGS :=

# Set cudaDeviceMapHost at startup.
# This flag needs to be set when your program uses cudaHostRegister
# to pin host virtual memory areas directly to main memory for UVA.
#MQX_CONFIGS += -DMQX_SET_MAPHOST

# Collect runtime statistics.
#MQX_CONFIGS += -DMQX_CONFIG_STATS

# Collect kernel execution time.
# NOTE: This flag, if set, causes a global barrier inserted after
# each kernel is issued, which reduces the concurrency level of using
# GPU device. Therefore, this flag should be turned on only during
# development.
# DO NOT turn on unless knowing how it works!
#MQX_CONFIGS += -DMQX_CONFIG_STAT_KERNEL_TIME

# Enable profiling.
# This is used to profile basic application DMA and kernel statistics
# when MQX is not enabled (i.e. when global arena is not created, thus
# linking to default CUDA runtime).
# DO NOT turn on unless knowing how it works!
#MQX_CONFIGS += -DMQX_CONFIG_PROFILE
