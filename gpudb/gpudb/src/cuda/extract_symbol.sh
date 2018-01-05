#!/bin/bash
cuobjdump -symbols ops.cubin | grep "STB_GLOBAL STO_ENTRY" | awk '{print $4}'
