#!/usr/bin/python

fd = open('../query_progs/log', 'rt')
time_cuda = 0
time_query = 0
lines = fd.readlines()
for line in lines:
    if line.find('uninitialized') != -1:
        print line
        exit()
for line in lines:
    if line.startswith('S CUDA Context'):
        time_cuda += float(line[14:].strip())
    elif line.startswith('S MQX Total'):
        time_query += float(line[11:].strip())

print 'cuda context, ', time_cuda, ', query, ', time_query, ', total, ', time_cuda + time_query
