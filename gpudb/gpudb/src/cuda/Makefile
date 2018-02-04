
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

DATAPATH := ../../../../data
GMMPATH := /home/lyl/gpudb/gpudb/gmm
LDFLAGS += -L$(GMMPATH)
CFLAGS := -I$(GMMPATH) -I../include
SRC := tableScan.cu hashJoin.cu cuckoo.cu inviJoin.cu materialize.cu groupBy.cu orderBy.cu scanImpl_merged.cu
SRC_PATCHED := tableScan.patched.cu hashJoin.patched.cu cuckoo.patched.cu inviJoin.patched.cu materialize.patched.cu groupBy.patched.cu orderBy.patched.cu scanImpl_merged.patched.cu
OBJ := tableScan.o  hashJoin.o  cuckoo.o  inviJoin.o  materialize.o  groupBy.o  orderBy.o  scanImpl_merged.o
OBJ_PATCHED := tableScan.patched.o  hashJoin.patched.o  cuckoo.patched.o  inviJoin.patched.o  materialize.patched.o  groupBy.patched.o  orderBy.patched.o  scanImpl_merged.patched.o

all: mpsdb

modules: $(SRC)
	 $(NVCC) $(GENCODE_FLAGS) -cubin -dlink -g $(SRC) $(CFLAGS) -L$(CUDA_LIB_PATH) $(LDFLAGS) -Xlinker -rpath=$(GMMPATH) -lgmm -o ops.cubin

mpsdb: gmm driver-mps
	 $(NVCC) --cudart=shared -g $(OBJ_PATCHED) driver.o -L$(CUDA_LIB_PATH) $(LDFLAGS) -Xlinker -rpath=$(GMMPATH) -lgmm -o GPUDATABASE

gmm: $(SRC_PATCHED)
	$(NVCC) -c -g $(GENCODE_FLAGS) $(CFLAGS) -DHAS_GMM $(SRC_PATCHED)

driver-mps: driver.cu
	$(NVCC) -c -g $(GENCODE_FLAGS) $(CFLAGS) -DHAS_GMM driver.cu

gmmdb: gmm driver
	 $(NVCC) --cudart=shared -g $(OBJ_PATCHED) driver.o -L$(CUDA_LIB_PATH) $(LDFLAGS) -Xlinker -rpath=$(GMMPATH) -lgmm -o GPUDATABASE

gpudb: gpu driver
	 $(NVCC) -g $(OBJ) driver.o -L$(CUDA_LIB_PATH) $(LDFLAGS) -o GPUDATABASE

driver: driver.cu 
	$(NVCC) -c -g $(GENCODE_FLAGS) $(CFLAGS) driver.cu

gpu: $(SRC)
	$(NVCC) -c -g $(GENCODE_FLAGS) $(CFLAGS) $(SRC)

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
