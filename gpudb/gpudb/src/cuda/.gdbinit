set exec-wrapper env "LD_PRELOAD=/home/lyl/gpudb/gpudb/gmm/libgmm.so"
file GPUDATABASE
tbreak main
r --datadir /home/lyl/gpudb/data
