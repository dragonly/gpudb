#!/bin/bash
echo "#ifndef __KERNEL_SYMBOLS_H_"
echo "#define __KERNEL_SYMBOLS_H_"
echo
echo "const char* f_table[] = {"
cuobjdump -symbols ops.cubin | grep -v "STB_LOCAL" | grep "STO_ENTRY" | awk '{print $4}' | sort | awk '{print "  ["NR-1"] = \"" $1 "\","}'
echo "};"
echo
echo "#endif"
