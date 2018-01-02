# Micro test cases
TESTS = test_init_fini test_malloc test_cow test_memcpy test_memset \
		test_launch test_ptarray test_evict_local

# Paths
CUDAPATH = /usr/local/cuda
MQXPATH := `pwd`/../../src
TMPPATH = ./tmp

# Compiler/linker settings
NVCC := $(CUDAPATH)/bin/nvcc
CFLAGS := -c --compiler-options -Wall -arch=sm_61 \
	-I$(CUDAPATH)/include -I$(MQXPATH)
LDFLAGS := -L$(CUDAPATH)/lib64 -L$(MQXPATH) -Xlinker -rpath=$(MQXPATH) -lmqx -lcudart

.DEFAULT_GOAL := all
.SECONDEXPANSION:
.PHONY : all test setup cleanup $(TESTS)

TESTBINS := $(addprefix $(TMPPATH)/,$(TESTS))

all : $(TMPPATH) $(TESTBINS)

$(TMPPATH) :
	@mkdir -p $(TMPPATH)

$(TESTBINS) : $$@.o
	@./tcgen.py $<
	@$(NVCC) $(CFLAGS) main.cu -o $(TMPPATH)/main.o
	$(NVCC) $(LDFLAGS) $(TMPPATH)/main.o $< -o $@
	-@rm $(TMPPATH)/main.o

$(TMPPATH)/%.o : %.cu
	$(NVCC) $(CFLAGS) $< -o $@

# No rules for source files
%.c : ;

$(TESTS) : $(TMPPATH)/$$@
	@echo "================================================================"
	@LD_PRELOAD=$(MQXPATH)/libmqx.so $(TMPPATH)/$@
	@echo ""

test : setup $(TESTS) cleanup

setup:
	@$(MQXPATH)/mqxctl --start -v
	@echo ""

cleanup:
	@$(MQXPATH)/mqxctl --stop -v

clean:
	-@rm $(TESTBINS) $(TMPPATH)/*.o testcases.h
	-@rm -r $(TMPPATH)
