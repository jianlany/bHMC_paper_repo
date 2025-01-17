#!/usr/bin/env python3
"""
Calculate the lbt information of a lammps data file generated by hmc procedures.
"""
import sys
import numpy
from mpi4py import MPI
import lammps

class Sc_system:
    pass

def calculate_lbt_info(datafile, cryst_ids_path, amorph_ids_path):

    lmp = lammps.lammps(cmdargs = ['-log', 'none', '-screen', 'none'])
    lmp_numpy = lmp.numpy
    sc_sys = Sc_system()
    sc_sys.cryst_ids, sc_sys.amorph_ids = read_ids(cryst_ids_path, amorph_ids_path)
    lmp.commands_string(setup_command.format(datafile = datafile))
    sc_sys.natoms = lmp.get_natoms()

    sc_sys.distance_from_crystalline = numpy.zeros(sc_sys.natoms)
    sc_sys.segment_lengths_atom = numpy.zeros(sc_sys.natoms)
    sc_sys.cryst_anchor = numpy.ones(sc_sys.natoms)
    for i, cryst_layer_id in enumerate(sc_sys.cryst_ids):
        for j in cryst_layer_id:
            sc_sys.distance_from_crystalline[j-1] = -i

    bonds = lmp_numpy.gather_bonds().astype(int)
    sc_sys.conn = build_conn(bonds, sc_sys.natoms)
    sc_sys.tail_ends = set()
    for i in sc_sys.amorph_ids:
        if len(sc_sys.conn[i-1]) == 1:
            sc_sys.tail_ends.add(i)
    find_topologies(sc_sys)
    print(len(sc_sys.tails))
    print(len(sc_sys.loops))
    print(len(sc_sys.bridges))
    for segs, seg_type in zip([sc_sys.tails, sc_sys.loops, sc_sys.bridges],
                             ['tails', 'loops', 'bridges']):
        with open('{}_atom_ids.txt'.format(seg_type), 'w') as f:
            for segment in sorted(segs, key = len):
                for atom in segment:
                    f.write('{} '.format(atom))
                f.write('\n')


def read_ids(cryst_ids_path, amorph_ids_path):
    amorph_ids = \
            numpy.genfromtxt(amorph_ids_path, dtype = int)
    cryst_ids = []
    with open(cryst_ids_path, 'r') as f:
        for line in f:
            cryst_ids.append(numpy.array(
                [int(i) for i in line.strip().split()]
                ))
    return cryst_ids, amorph_ids


def build_conn(bonds, natoms):
    """ Build connectivity matrix for they system"""
    conn = [[] for _ in range(natoms)]
    for _, b1, b2 in bonds:
        conn[b1-1].append(b2)
        conn[b2-1].append(b1)
    for i, _ in enumerate(conn):
        conn[i] = sorted(conn[i])
    return conn


def find_topologies(sc_sys):
    """ Find the topological belongings of amorphous atoms"""
    amorph_ids_set = set(sc_sys.amorph_ids)
    sc_sys.tail_atoms = set()
    sc_sys.tails = []
    seen = set()
    for i in sc_sys.tail_ends:
        if i in seen: continue
        if i not in amorph_ids_set: continue
        seen.add(i)
        stack = [i]
        tail = []
        anchor = None
        while stack:
            j = stack.pop()
            tail.append(j)
            for jj in sc_sys.conn[j-1]:
                if jj in seen: continue
                if jj not in amorph_ids_set: 
                    anchor = sc_sys.distance_from_crystalline[jj-1]
                    continue
                seen.add(jj)
                stack.append(jj)
        # reverse the tail list, record the distance from crystalline domain
        sc_sys.tails.append(tail)
        assert anchor is not None
        for d, j in enumerate(tail[::-1]):
            sc_sys.tail_atoms.add(j)
            sc_sys.distance_from_crystalline[j-1] = d + 1
            sc_sys.segment_lengths_atom[j-1] = len(tail)
            sc_sys.cryst_anchor[j-1] = anchor

    sc_sys.bridge_atoms = set()
    sc_sys.loop_atoms = set()
    sc_sys.bridges = []
    sc_sys.loops = []
    for i in sc_sys.amorph_ids:
        # Start with an amorphous bead that connects to a crystalline phase
        if i in sc_sys.tail_atoms: continue
        if i in seen: continue
        assert len(sc_sys.conn[i-1]) == 2
        if (sc_sys.conn[i-1][0] in amorph_ids_set) == (sc_sys.conn[i-1][1] in amorph_ids_set): continue
        # the crystalline bead that i connects to
        end1 = sc_sys.conn[i-1][0] \
            if sc_sys.conn[i-1][0] not in amorph_ids_set \
            else sc_sys.conn[i-1][1]
        end2 = None
        stack = [i]
        seen.add(i)
        segment = []
        while stack:
            j = stack.pop()
            segment.append(j)
            for jj in sc_sys.conn[j-1]:
                if jj in seen: continue
                if jj == end1: continue
                if jj not in amorph_ids_set:
                    end2 = jj
                    continue
                seen.add(jj)
                stack.append(jj)
        # if two ends are in the same crystalline layer, it is a loop
        # else it is a bridge
        assert end2 is not None
        anchor1 = sc_sys.distance_from_crystalline[end1-1]
        anchor2 = sc_sys.distance_from_crystalline[end2-1]
        if anchor1 == anchor2:
            segments = sc_sys.loops
            segments_set = sc_sys.loop_atoms
        else:
            segments = sc_sys.bridges
            segments_set = sc_sys.bridge_atoms
        segments.append(segment)
        for d, j in enumerate(segment):
            segments_set.add(j)
            sc_sys.distance_from_crystalline[j-1] = min(d+1, len(segment)-d)
            sc_sys.segment_lengths_atom[j-1] = len(segment)
            if segments is sc_sys.loops:
                sc_sys.cryst_anchor[j-1] = anchor1


setup_command = """
units           real
atom_style      angle
read_data       {datafile} nocoeff

compute         all_ids all property/atom id
"""
if __name__ == "__main__":
    calculate_lbt_info(sys.argv[1],sys.argv[2],sys.argv[3])

