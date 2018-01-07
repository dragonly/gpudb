#!/bin/bash
function gen_h() {
  echo "#ifndef __KERNEL_SYMBOLS_H_"
  echo "#define __KERNEL_SYMBOLS_H_"
  echo
  echo "static CUcontext context;"
  echo "static CUmodule mod_ops;"
  #cuobjdump -symbols ops.cubin | grep -v "STB_LOCAL" | grep "STO_ENTRY" | awk '{print $4}' | sort | awk '{print "static CUfunction F_" $1 ";"}'
  echo
  n=`cuobjdump -symbols ops.cubin | grep -v "STB_LOCAL" | grep "STO_ENTRY" | wc -l`
  echo "#define NUMFUNC $n"
  echo "static CUfunction fsym_table[NUMFUNC] = { NULL };"
  #echo
  #cuobjdump -symbols ops.cubin | grep -v "STB_LOCAL" | grep "STO_ENTRY" | awk '{print $4}' | sort | awk '{print "fsym_table["NR-1"] = &F_" $1 ";"}'
  echo
  echo "const static char* fname_table[] = {"
  cuobjdump -symbols ops.cubin | grep -v "STB_LOCAL" | grep "STO_ENTRY" | awk '{print $4}' | sort | awk '{print "  ["NR-1"] = \"" $1 "\","}'
  echo "};"
  echo
  echo "#endif"
}
function gen_c() {
  # this file is for embedding purpose
  # which means a weird `#include init_kernel_symbols.c`
  cuobjdump -symbols ops.cubin | grep -v "STB_LOCAL" | grep "STO_ENTRY" | awk '{print $4}' | sort | awk '{print "fsym_table["NR-1"] = &F_" $1 ";"}'
}

gen_h > kernel_symbols.h
#gen_c > init_kernel_symbols.c
