#!/usr/bin/env python3
import numpy
import sys


def read_topo_file(topofile, start = 0, intv = 1, stop = -1, last = False):
    tail = {}
    loop = {}
    bridge = {}
    count = -1
    map_steps_count = {}
    with open(topofile, 'r') as f:
        if last:
            find_last_occurence(f)
            start = 0
            intv = 1
            stop = -1
        for line in f:
            line = line.strip().split()
            if line[0] == 'Step:':
                step = int(line[1])
                if step not in map_steps_count:
                    count += 1
                    map_steps_count[step] = count
                else:
                    count = map_steps_count[step]

                if count < start: 
                    read = False
                    continue
                elif stop > 0 and count > stop: 
                    read = False
                    continue
                elif (count - start)%intv: 
                    read = False
                    continue
                else:
                    read = True
                for seg in [tail, loop, bridge]: seg[step] = []

            elif read and line[0] == 'Number':
                if line[2] == 'tails:':
                    num_of_tails = int(line[3])
                elif line[2] == 'loops:':
                    num_of_loops = int(line[3])
                elif line[2] == 'bridges:':
                    num_of_bridges = int(line[3])
                else:
                    print("Something is wrong!")
                    print(line)
                    sys.exit(1)
            elif read and line[0] == 'tail_length':
                tail_lengths, loop_lengths, bridge_lengths = \
                        numpy.genfromtxt(f, max_rows = max(num_of_tails, 
                                                           num_of_loops, 
                                                           num_of_bridges)).T
                tail[step] = tail_lengths[numpy.logical_not(numpy.isnan(tail_lengths))]
                loop[step] = loop_lengths[numpy.logical_not(numpy.isnan(loop_lengths))]
                bridge[step] = bridge_lengths[numpy.logical_not(numpy.isnan(bridge_lengths))]
                assert (len(tail[step]) == num_of_tails) and \
                       (len(loop[step]) == num_of_loops) and \
                       (len(bridge[step]) == num_of_bridges)
    return tail, loop, bridge


def find_last_occurence(fid, s='Step: ', block_size=2048):
    ''' Returns an open file handle at the last occurance of string s. '''
    fid.seek(0, 2)
    loc = fid.tell()
    while loc > 0:
        loc = max(loc - block_size, 0)
        fid.seek(loc, 0)
        data = fid.read(block_size + len(s))
        m = data.find(s)
        if m >= 0:
            fid.seek(loc+m, 0)
            return fid
