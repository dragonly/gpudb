include config.mk

ifeq ($(USE_PRINT_BUFFER), 0)
FLAG_PRINT_BUFFER :=
else
FLAG_PRINT_BUFFER := -DMQX_PRINT_BUFFER
endif

# CUDA installation path
CUDAPATH = /usr/local/cuda

# Source files needed to generate libmqx
SRCS = client.c common.c core.c cow.c interfaces.c \
	   msq.c replacement.c stats.c libc_hooks.c
OBJS = $(SRCS:.c=.o)

# Compiler/linker settings
CC := gcc
NVCC := $(CUDAPATH)/bin/nvcc
#CFLAGS := -g -Wall -fPIC -fvisibility=hidden -I$(CUDAPATH)/include \
#          -DCUDAPATH=\"$(CUDAPATH)\" -DMQX_PRINT_LEVEL=$(PRINT_LEVEL) \
#          $(FLAG_PRINT_BUFFER) $(MQX_CONFIGS)
CFLAGS := -g -Wall -fPIC -I$(CUDAPATH)/include \
          -DCUDAPATH=\"$(CUDAPATH)\" -DMQX_PRINT_LEVEL=$(PRINT_LEVEL) \
          $(FLAG_PRINT_BUFFER) $(MQX_CONFIGS)
LDFLAGS := -shared -pthread -ldl

.DEFAULT_GOAL := all
.PHONY : depend all clean install uninstall

all: depend mqxctl libmqx.so libmps.so mpsserver mps_testclient

# Generate dependencies for $(OBJS)
depend : .depend

.depend : $(SRCS)
	$(CC) $(CFLAGS) -MM $(SRCS) > .depend

-include .depend

# No rules for source files
%.c: ;

mpsserver : mpsserver.o common.o serialize.o libmpsserver.o
	$(NVCC) -cudart=static -lcuda -arch=sm_61 $^ -o $@

mpsserver.o: mpsserver.c
	$(NVCC) -c -g -Xcompiler -Wall -DMQX_PRINT_LEVEL=5 -DSTANDALONE -I$(CUDAPATH)/include -arch=sm_61 $< -o $@

libmps.so: libmpsclient.o serialize.o common.o mps_interfaces.o
	$(CC) -g -Wall -shared -pthread -ldl -lcuda -DMQX_PRINT_LEVEL=5  -I$(CUDAPATH)/include $^ -o $@

mps_testclient : mps_testclient.o serialize.o common.o
	$(NVCC) -cudart=shared -L$(CUDAPATH)/lib64 -lcuda -lrt -ldl -lpthread -arch=sm_61 $^ -o $@

mps_testclient.o: mps_testclient.c
	$(NVCC) -c -g -shared -Xcompiler -Wall,-fPIC,-Wall -DMQX_PRINT_LEVEL=5 $< -o $@

mqxctl : mqxctl.o common.o
	$(NVCC) -L$(CUDAPATH)/lib64 -arch=sm_61 $^ -o $@

mqxctl.o : mqxctl.c common.h protocol.h spinlock.h list.h atomic.h
	$(NVCC) -c --compiler-options -Wall -DMQX_PRINT_LEVEL=$(PRINT_LEVEL) -I$(CUDAPATH)/include -arch=sm_61 $< -o $@

libmqx.so: $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

clean:
	-rm -f mpsserver mps_testclient mqxctl libmps.so libmqx.so *.o .depend

# TODO
install: ;

# TODO
uninstall: ;
