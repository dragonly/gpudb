define client
  set exec-wrapper env "LD_PRELOAD=/home/lyl/gpudb/mqx/src/libmps.so"
  file vectorAdd
  tbreak main
  r
end
