#!/bin/bash

#queries=(q1_1 q1_2 q1_3 q2_1 q2_2 q2_3 q3_1 q3_2 q3_3 q3_4 q4_1 q4_2 q4_3)
queries=(q1_2 q3_4 q2_1 q4_1 q3_1 q4_3 q2_2 q3_2 q3_3 q4_2)
for q in ${queries[*]}; do
  echo $q
  eval "LD_PRELOAD=/home/lyl/gpudb/gpudb/gmm/libgmm.so ../query_progs/${q} --datadir /home/lyl/gpudb/data_s10"
done

# LD_PRELOAD=/home/lyl/gpudb/gpudb/gmm/libgmm.so ../query_progs/q1_1 --datadir /home/lyl/gpudb/data_s10
# LD_PRELOAD=/home/lyl/gpudb/gpudb/gmm/libgmm.so ../query_progs/q1_2 --datadir /home/lyl/gpudb/data_s10
# LD_PRELOAD=/home/lyl/gpudb/gpudb/gmm/libgmm.so ../query_progs/q1_3 --datadir /home/lyl/gpudb/data_s10
# LD_PRELOAD=/home/lyl/gpudb/gpudb/gmm/libgmm.so ../query_progs/q2_1 --datadir /home/lyl/gpudb/data_s10
# LD_PRELOAD=/home/lyl/gpudb/gpudb/gmm/libgmm.so ../query_progs/q2_2 --datadir /home/lyl/gpudb/data_s10
# LD_PRELOAD=/home/lyl/gpudb/gpudb/gmm/libgmm.so ../query_progs/q2_3 --datadir /home/lyl/gpudb/data_s10
# LD_PRELOAD=/home/lyl/gpudb/gpudb/gmm/libgmm.so ../query_progs/q3_1 --datadir /home/lyl/gpudb/data_s10
# LD_PRELOAD=/home/lyl/gpudb/gpudb/gmm/libgmm.so ../query_progs/q3_2 --datadir /home/lyl/gpudb/data_s10
# LD_PRELOAD=/home/lyl/gpudb/gpudb/gmm/libgmm.so ../query_progs/q3_3 --datadir /home/lyl/gpudb/data_s10
# LD_PRELOAD=/home/lyl/gpudb/gpudb/gmm/libgmm.so ../query_progs/q3_4 --datadir /home/lyl/gpudb/data_s10
# LD_PRELOAD=/home/lyl/gpudb/gpudb/gmm/libgmm.so ../query_progs/q4_1 --datadir /home/lyl/gpudb/data_s10
# LD_PRELOAD=/home/lyl/gpudb/gpudb/gmm/libgmm.so ../query_progs/q4_2 --datadir /home/lyl/gpudb/data_s10
# LD_PRELOAD=/home/lyl/gpudb/gpudb/gmm/libgmm.so ../query_progs/q4_3 --datadir /home/lyl/gpudb/data_s10
