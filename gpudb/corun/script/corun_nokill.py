#!/usr/bin/python
import os
import time
import shutil

os.chdir("../../")
rootpath = os.getcwd()

#plan_file = rootpath + r'/corun/exec_plan/4q.plan'
plan_file = rootpath + r'/corun/exec_plan/2q.plan'
outpath = rootpath + r'/corun/output/'
querypath = rootpath + r'/corun/query_progs/'
datapath = rootpath + r'/../data/'

# corun for #rep times
rep = '9'

LOAD_GMM = 1
if LOAD_GMM:
	preloadlib=r'LD_PRELOAD='+rootpath+r'/gmm/libgmm.so '
else:
	preloadlib=''
#preloadlib = ''
#preloadlib = 'LD_PRELOAD=' + rootpath + r'/lib-intercept/libicept.so'
#preloadlib = 'LD_PRELOAD=' + rootpath + r'/gmm/libgmm.so'


if os.path.exists(outpath):
	shutil.rmtree(outpath)
os.mkdir(outpath)

plans = open(plan_file, "r").readlines()

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
	i = 0
	# Now we corun the querys
	for query in querys:
		if running_query.has_key(query) is True:
			oo = output + query + '_' + str(running_query[query])
			running_query[query] += 1
		else:
			oo = output + query
			running_query[query] = 1

		cmd = preloadlib + ' ./' + query + ' --datadir ' + datapath + ' >> ' + oo + ' 2>>' + output+'error'
		i += 1
		if i == len(querys):
			script = rootpath+'/corun/script/reprun.py ' + rep + ' ' + cmd
		else:
			script = rootpath+'/corun/script/reprun.py ' + rep + ' ' + cmd + ' &'
		print script
		os.system(script)
