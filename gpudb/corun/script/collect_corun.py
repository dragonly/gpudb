#!/bin/python2

import os
import sys


queries = [
    'q1_1', 'q1_2', 'q1_3',
    'q2_1', 'q2_2', 'q2_3',
    'q3_1', 'q3_2', 'q3_3', 'q3_4',
    'q4_1', 'q4_2', 'q4_3'
]
qmap = {}
for i, q in enumerate(queries):
    qmap[q] = 'q' + str(i + 1)
    qmap[q] = q.replace('_', '')

def gen(resultdir):
    cwd = os.getcwd()
    os.chdir('../result/{}/'.format(resultdir))
    data_dirs = os.listdir('.')
    data_dirs.sort()
    result = []
    for d in data_dirs:
        rep = 9
        try:
            assert os.stat('{}/error'.format(d)).st_size == 0
        except Exception as e:
            print 'error'
            print d
            exit()
        q1, q2 = d.split('.')
        if (q1 not in queries) or (q2 not in queries):
            print "excluding", d
            continue
        result_item = [qmap[q1], qmap[q2]]
        #result_item = [q1, q2]
        if q1 == q2:
            q2 = q2 + '_1'
        f1 = open('{}/{}'.format(d, q1), 'rt').readlines()
        f2 = open('{}/{}'.format(d, q2), 'rt').readlines()

        time = 0
        ntime = 0
        skip = SKIP = 1
        if resultdir == 'mps_result':
            for line in f1:
                if line.startswith('Total Time: '):
                    time += float(line[12:].strip())
                    ntime += 1
        elif resultdir == 'mqx_result':
            for line in f1:
                if line.startswith('S MQX Total'):
                    if skip == 0:
                        time += float(line[21:].strip())
                if line.startswith('S CUDA Context'):
                    ntime += 1
                    if skip == 0:
                        time += float(line[21:].strip())
                    if skip > 0: skip -= 1

        result_item.append(round(time / (rep-SKIP), 2))

        time = 0
        ntime = 0
        skip = 1
        if resultdir == 'mps_result':
            for line in f2:
                if line.startswith('Total Time: '):
                    time += float(line[12:].strip())
                    ntime += 1
        elif resultdir == 'mqx_result':
            for line in f2:
                if line.startswith('S MQX Total'):
                    if skip == 0:
                        time += float(line[21:].strip())
                if line.startswith('S CUDA Context'):
                    ntime += 1
                    if skip == 0:
                        time += float(line[21:].strip())
                    if skip > 0: skip -= 1

        assert ntime == rep

        result_item.append(round(time / rep, 2))
        result.append(result_item)

    os.chdir(cwd)
    return result

#print result
#print len(result)

mqx_result = gen('mqx_result')
mps_result = gen('mps_result')
result = []
for mqx, mps in zip(mqx_result, mps_result):
    r = []
    r.append('{}.{}'.format(mps[0], mps[1]))
    r.append( (mqx[2] + mqx[3]) / (mps[2] + mps[3]) )
    result.append(r)
for r in result:
    print '{}, {}'.format(r[0], r[1])
