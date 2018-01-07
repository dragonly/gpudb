#!/usr/bin/python2
import os
import re


headerfile = 'kernel_symbols.h'
pkt = re.compile(r'\[(\d+)]\ =\ \"(.+)\"')
kernel_table = {}
fdh = open(headerfile, 'rt').readlines()
start = None
for i, line in enumerate(fdh):
    if line.find('fname_table') != -1:
        start = i
        break
fdh = fdh[start:]
for line in fdh:
    m = pkt.search(line)
    if m != None:
        index = m.group(1)
        kernel = m.group(2)
        kernel_table[kernel] = index
assert len(kernel_table) == 138

kernel_sources = ['tableScan.cu', 'groupBy.cu', 'orderBy.cu', 'materialize.cu', 'hashJoin.cu', 'inviJoin.cu', 'cuckoo.cu', 'scanImpl_merged.cu']
kernel_sources = kernel_sources[:1]
pkcall = re.compile(r'(\s*)([_a-zA-Z0-9]+)<<<.+>>>')
for src in kernel_sources:
    fdin = open(src, 'rt')
    fdout = open(src[:-3] + '.patched.cu', 'wt')
    for line in fdin:
        m = pkcall.search(line)
        if m != None:
            tab = m.group(1)
            kernel = m.group(2)
            index = kernel_table[kernel]
            fdout.write('{}GMM_CALL(cudaSetFunction({}));\n'.format(tab, index))
        fdout.write(line)
    fdin.close()
    fdout.close()
