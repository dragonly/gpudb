
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")

CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin

ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
endif

NVCC            ?= $(CUDA_BIN_PATH)/nvcc

GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
GENCODE_SM61    := -gencode arch=compute_61,code=sm_61
GENCODE_FLAGS   := $(GENCODE_SM61)

ifeq ($(OS_SIZE),32)
    LDFLAGS   := -lcudart -lrt
else
    LDFLAGS   := -lcudart -lrt
endif

DATAPATH := /home/lyl/gpudb/gpudb/data
GMMPATH := /home/lyl/gpudb/gpudb/gmm
LDFLAGS += -L$(GMMPATH)
CFLAGS := -I$(GMMPATH) -I../include
SRC := tableScan.cu hashJoin.cu cuckoo.cu inviJoin.cu materialize.cu groupBy.cu orderBy.cu scanImpl.cu
OBJ := tableScan.o  hashJoin.o  cuckoo.o  inviJoin.o  materialize.o  groupBy.o  orderBy.o  scanImpl.o

all: gmmdb

modules: $(SRC) driver.cu
	 $(NVCC) $(GENCODE_FLAGS) -cubin -dlink -pg $(SRC) driver.cu $(CFLAGS) -L$(CUDA_LIB_PATH) $(LDFLAGS) -Xlinker -rpath=$(GMMPATH) -lgmm -o ops.cubin

gmmdb: gmm cpu
	 $(NVCC) -pg $(OBJ) driver.o -L$(CUDA_LIB_PATH) $(LDFLAGS) -Xlinker -rpath=$(GMMPATH) -lgmm -o GPUDATABASE

gpudb: gpu cpu
	 $(NVCC) -pg $(OBJ) driver.o -L$(CUDA_LIB_PATH) $(LDFLAGS) -o GPUDATABASE

gmm: $(SRC)
	$(NVCC) -c -pg $(GENCODE_FLAGS) $(CFLAGS) -DHAS_GMM $(SRC)

cpu: driver.cu 
	$(NVCC) -c -pg $(GENCODE_FLAGS) $(CFLAGS) driver.cu

gpu: $(SRC)
	$(NVCC) -c -pg $(GENCODE_FLAGS) $(CFLAGS) $(SRC)

run:
	LD_PRELOAD=../../../lib-intercept/libicept.so ./GPUDATABASE --datadir ../../data

run1:
	./GPUDATABASE --datadir $(DATAPATH)

run2:
	LD_PRELOAD=$(GMMPATH)/libgmm.so ./GPUDATABASE --datadir $(DATAPATH)

corun:
	LD_PRELOAD=$(GMMPATH)/libgmm.so ./GPUDATABASE --datadir $(DATAPATH) & LD_PRELOAD=$(GMMPATH)/libgmm.so ./GPUDATABASE --datadir $(DATAPATH)

clean:
	rm -rf *.o GPUDATABASE
