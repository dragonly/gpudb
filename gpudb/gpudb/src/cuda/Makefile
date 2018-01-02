
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

DATAPATH := /home/lyl/gpu/gpudb/data
GMMPATH := /home/lyl/gpu/gpudb/gmm
LDFLAGS += -L$(GMMPATH)
CFLAGS := -I$(GMMPATH)

all: gmmdb

gmmdb: gmm cpu
	 $(NVCC) -pg tableScan.o inviJoin.o hashJoin.o cuckoo.o materialize.o groupby.o orderBy.o driver.o -L$(CUDA_LIB_PATH) $(LDFLAGS) -Xlinker -rpath=$(GMMPATH) -lgmm -o GPUDATABASE

gpudb: gpu cpu
	 $(NVCC) -pg tableScan.o inviJoin.o hashJoin.o cuckoo.o materialize.o groupby.o orderBy.o driver.o -L$(CUDA_LIB_PATH) $(LDFLAGS) -o GPUDATABASE

cpu: driver.cu 
	$(NVCC) -c -g $(GENCODE_FLAGS) driver.cu

gpu: tableScan.cu hashJoin.cu cuckoo.cu inviJoin.cu materialize.cu groupby.cu orderBy.cu
	$(NVCC) -c -g $(GENCODE_FLAGS) $(CFLAGS) tableScan.cu hashJoin.cu cuckoo.cu inviJoin.cu materialize.cu groupby.cu orderBy.cu 

gmm: tableScan.cu hashJoin.cu cuckoo.cu inviJoin.cu materialize.cu groupby.cu orderBy.cu
	$(NVCC) -c -g $(GENCODE_FLAGS) $(CFLAGS) -DHAS_GMM tableScan.cu hashJoin.cu cuckoo.cu inviJoin.cu materialize.cu groupby.cu orderBy.cu 

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
