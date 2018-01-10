include config.mk

ifeq ($(USE_PRINT_BUFFER), 0)
FLAG_PRINT_BUFFER :=
else
FLAG_PRINT_BUFFER := -DMQX_PRINT_BUFFER
endif

# Name of output shared library
LIBNAME = libmqx.so

# CUDA installation path
CUDAPATH = /usr/local/cuda

# Source files needed to generate $(LIBNAME)
SRCS = client.c common.c core.c cow.c interfaces.c \
	   msq.c replacement.c stats.c libc_hooks.c
OBJS = $(SRCS:.c=.o)

# Compiler/linker settings
CC := gcc
NVCC := $(CUDAPATH)/bin/nvcc
CFLAGS := -g -Wall -fPIC -fvisibility=hidden -I$(CUDAPATH)/include \
          -DCUDAPATH=\"$(CUDAPATH)\" -DMQX_PRINT_LEVEL=$(PRINT_LEVEL) \
          $(FLAG_PRINT_BUFFER) $(MQX_CONFIGS)
LDFLAGS := -shared -pthread -ldl

.DEFAULT_GOAL := all
.PHONY : depend all clean install uninstall

all : depend mqxctl $(LIBNAME) mpsserver mpsclient

# Generate dependencies for $(OBJS)
depend : .depend

.depend : $(SRCS)
	$(CC) $(CFLAGS) -MM $(SRCS) > .depend

-include .depend

# No rules for source files
%.c: ;

mpsserver : mpsserver.o common.o serialize.o mps.o
	$(NVCC) -L$(CUDAPATH)/lib64 -lcuda -arch=sm_61 $^ -o $@

mpsserver.o: mpsserver.c
	$(NVCC) -c -g --compiler-options -Wall -DMQX_PRINT_LEVEL=5 -DSTANDALONE -I$(CUDAPATH)/include -arch=sm_61 $< -o $@

mpsclient : mpsclient.o serialize.o common.o
	$(NVCC) -L$(CUDAPATH)/lib64 -lcuda -arch=sm_61 $^ -o $@

mpsclient.o: mpsclient.c
	$(NVCC) -c -g --compiler-options -Wall -DMQX_PRINT_LEVEL=5 $< -o $@

mqxctl : mqxctl.o common.o
	$(NVCC) -L$(CUDAPATH)/lib64 -arch=sm_61 $^ -o $@

mqxctl.o : mqxctl.c common.h protocol.h spinlock.h list.h atomic.h
	$(NVCC) -c --compiler-options -Wall -DMQX_PRINT_LEVEL=$(PRINT_LEVEL) -I$(CUDAPATH)/include -arch=sm_61 $< -o $@

$(LIBNAME): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

clean:
	-rm -f mpsserver mpsclient mqxctl $(LIBNAME) *.o .depend

# TODO
install: ;

# TODO
uninstall: ;
