#!/usr/bin/python
import os
import time
import shutil

os.chdir("../../")
rootpath = os.getcwd()

outpath = rootpath + r'/corun/output/'
querypath = rootpath + r'/corun/query_progs/'
datapath = rootpath + r'/../data_s10/'

# corun for #rep times
rep = '9'
CORUN = 9

preloadlib=r'LD_PRELOAD='+rootpath+r'/gmm/libgmm.so '

if os.path.exists(outpath):
    shutil.rmtree(outpath)
os.mkdir(outpath)

queries = [
    'q1_1', 'q1_2', 'q1_3',
    'q2_1', 'q2_2', 'q2_3',
    'q3_1', 'q3_2', 'q3_3', 'q3_4',
    'q4_1', 'q4_2', 'q4_3'
]
queries = queries[queries.index('q2_2'):]

for query in queries:
    output = outpath + query + '/'
    if os.path.exists(output):
        print "output path already exists"
        sys.exit(0)
    cmd = 'mkdir ' + output
    os.system(cmd)

    os.chdir(querypath)
    print query
    # Solorun the querys to load data into memory first
    cmd = preloadlib + ' ./' + query + ' --datadir ' + datapath
    os.system(cmd)
    # Now corun the querys
    for i in range(CORUN):
        oo = output + query + '.' + str(i)
        cmd = preloadlib + ' ./' + query + ' --datadir ' + datapath + ' >> ' + oo + ' 2>>' + output+'error'
        thread = rootpath+'/corun/script/reprun.py ' + rep + ' ' + cmd + ' &'
        if i == 0:
            print thread, 'x', CORUN
        os.system(thread)

    time.sleep(5)
    #cmd=' '
    #os.system(cmd) # like press an enter for the last '&'
    #for query in querys:
    #    cmd = r'ps -C ' + query + ' -o pid=|xargs'
    #    pid = os.popen(cmd).read().strip()
    #    if pid:
    #        cmd ='kill -9 ' + pid
    #        os.system(cmd)
    #        oo = query+' is killed, ' + pid
    #        print oo
    #    #cmd = oo+ ' > ' + output + '/'
    #    #os.system(oo)
