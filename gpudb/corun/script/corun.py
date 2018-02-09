#!/usr/bin/python
import os
import time
import shutil

os.chdir("../../")
rootpath = os.getcwd()

plan_file = rootpath + r'/corun/exec_plan/2q.plan'
outpath = rootpath + r'/corun/output/'
querypath = rootpath + r'/corun/query_progs/'
datapath = rootpath + r'/../data_s10/'

# corun for #rep times
rep = '9'

preloadlib=r'LD_PRELOAD='+rootpath+r'/gmm/libgmm.so '

if os.path.exists(outpath):
    shutil.rmtree(outpath)
os.mkdir(outpath)

plans = open(plan_file, "r").readlines()
plans = [p.strip() for p in plans if p.find('q3_1') != -1]
print plans
#i = plans.index('q3_1 q4_2')
#plans = plans[i:]
#print i
#print plans

for plan in plans:
    output = outpath + plan.strip().replace(' ', '.') + '/'
    if os.path.exists(output):
        print "Corun for the second time"
        sys.exit(0)
    cmd = 'mkdir ' + output
    os.system(cmd)
    running_query = {}

    querys = plan.strip().split(' ')
    os.chdir(querypath)
    print plan
    # Solorun the querys to load data into memory first
    for query in querys:
        cmd = preloadlib + ' ./' + query + ' --datadir ' + datapath
        os.system(cmd)
    # Now we corun the querys
    for query in querys:
        if running_query.has_key(query) is True:
            oo = output + query + '_' + str(running_query[query])
            running_query[query] += 1
        else:
            oo = output + query
            running_query[query] = 1

        cmd = preloadlib + ' ./' + query + ' --datadir ' + datapath + ' >> ' + oo + ' 2>>' + output+'error'
        script = rootpath+'/corun/script/reprun.py ' + rep + ' ' + cmd + ' &'
        #print script
        os.system(script)

    time.sleep(3)
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
