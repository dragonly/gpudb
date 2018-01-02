set exec-wrapper env "LD_PRELOAD=/home/lyl/gpu/gpudb/gmm/libgmm.so"
file GPUDATABASE
tbreak main
r --datadir /home/lyl/gpu/data
