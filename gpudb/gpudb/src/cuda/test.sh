#!/bin/sh
export LD_PRELOAD=/home/lyl/gpudb/gpudb/gmm/libgmm.so
# preload data
./GPUDATABASE --datadir /home/lyl/gpudb/data_s10
sleep 1
# launch 10 simultaneous query
for i in {1..10}; do
  ./GPUDATABASE --datadir /home/lyl/gpudb/data_s10 > "log$i" 2>&1 &
done
echo DONE
