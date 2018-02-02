#!/bin/python2

import os

rep = 9
os.chdir('../output/')
data_dirs = os.listdir('.')
data_dirs.sort()
result = []
for d in data_dirs:
    assert os.stat('{}/error'.format(d)).st_size == 0
    q1, q2 = d.split('.')
    if q1 == q2:
        q2 = q2 + '_1'
    #result_item = [q1, q2]
    result_item = [d]
    f1 = open('{}/{}'.format(d, q1), 'rt')
    f2 = open('{}/{}'.format(d, q2), 'rt')

    time = 0
    for line in f1:
        if line.startswith('Total Time: '):
            time += float(line[12:].strip())
    result_item.append(round(time / rep, 2))
    time = 0
    for line in f2:
        if line.startswith('Total Time: '):
            time += float(line[12:].strip())
    result_item.append(round(time / rep, 2))
    result.append(result_item)

    f1.close()
    f2.close()

print result
print len(result)
