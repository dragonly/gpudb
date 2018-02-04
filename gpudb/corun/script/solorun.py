#!/usr/bin/python
import os
import shutil
import time
import subprocess

os.chdir("../../")
rootpath = os.getcwd()

outpath = rootpath + r'/trace/file/'
querypath = rootpath + r'/corun/query_progs/'
datapath = rootpath + r'/../data_s10/'

rep = 1
preloadlib=r'LD_PRELOAD='+rootpath+r'/gmm/libgmm.so '

if os.path.exists(outpath):
    shutil.rmtree(outpath)
os.mkdir(outpath)

print querypath
for query in os.listdir(querypath):
    print query
    #for i in range(0, rep):
    cmd = preloadlib + querypath + query + ' --datadir ' + datapath + ' >> ' + outpath + query + '.solo'
    #print cmd
    subprocess.call(cmd, shell=True)
