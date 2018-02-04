#!/usr/bin/python
import sys
import os

if __name__ == '__main__':
    rep = int(sys.argv[1])
    cmd = ''
    for i in range(2, len(sys.argv)):
        cmd = cmd + ' ' + sys.argv[i]
    print 'REP:', cmd
    for i in range(0, rep):
        os.system(cmd)
    print "REP DONE"
