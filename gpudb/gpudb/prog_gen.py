#!/usr/bin/python
import os
from multiprocessing import Pool, Process
import uuid
import shutil


def gen_program(arg):
    path, prog_name = arg
    os.chdir(path + '/cuda')
    os.system(make_command)
    cmd = r'cp GPUDATABASE ' + rootpath + r'/corun/query_progs/' + prog_name
    ret = os.system(cmd)
    if ret != 0:
        print('oops')
        exit(1)
    return ret

TEMP_DIR = 'tmp-' + str(uuid.uuid4())
os.chdir("../")
rootpath = os.getcwd()

LOAD_GMM = 1

if LOAD_GMM:
    ldpreload=r'LD_PRELOAD='+rootpath+r'/gmm/libgmm.so '
    make_command = 'make gmmdb'
else:
    ldpreload=''
    make_command = 'make gpudb'
#elif LOAD_LIBICEPT:
#	ldpreload=r'LD_PRELOAD='+rootpath+r'/lib-intercept/libicept.so '
#	make_command = 'make gpudb'


if not os.path.exists(rootpath+'/corun/query_progs/'):
    os.mkdir(rootpath+'/corun/query_progs/')

if not os.path.exists(rootpath+'/trace/'):
    os.mkdir(rootpath+'/trace/')
if not os.path.exists(rootpath+'/trace/file/'):
    os.mkdir(rootpath+'/trace/file/')

sqlfiles = os.listdir(rootpath+"/gpudb/test/ssb_test/")
sqlfiles = filter(lambda x: x[-3:] == 'sql', sqlfiles)
ps = []
tmpdirs = []
for sqlfile in sqlfiles:
    tmpdirs.append('tmp-' + str(uuid.uuid4()))
for sqlfile, tmpdir in zip(sqlfiles, tmpdirs):
    os.chdir(rootpath+"/gpudb/")
    cmd = rootpath + r'/gpudb/translate.py' + r' ' + rootpath+'/gpudb/test/ssb_test/'+sqlfile + r' '+ rootpath+r'/gpudb/test/ssb_test/ssb.schema'
    os.system(cmd)
    shutil.copytree(rootpath + '/gpudb/src/cuda', tmpdir + '/cuda')
    shutil.copytree(rootpath + '/gpudb/src/include', tmpdir + '/include')
prog_names = map(lambda x: x[:-4], sqlfiles)
pool = Pool(len(sqlfiles))
pool.map(gen_program, zip(tmpdirs, prog_names))
print('cleaning up')
for tmpdir in tmpdirs:
    shutil.rmtree(tmpdir)

