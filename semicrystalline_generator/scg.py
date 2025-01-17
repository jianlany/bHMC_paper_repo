#!/usr/bin/env python3
import sys
import os
import numpy
from math import degrees
import logging
from scipy.constants import Avogadro as avog
from generator import domain_builder
import lammps
from mpi4py import MPI
comm = MPI.COMM_WORLD

def generate_crystalline_system(options, me):
    if me == 0:
        nxyz = (options.nx, options.ny, options.nz)
        abc = (options.a, options.b, options.c)
        rho_amorphous = options.rho_amorphous
        # abc = (7.4, 4.93, 2.54) # this is 300 K
        # crystal_regions = [[52.35, 157.0]]
        crystal_regions = options.crystal_regions
        logging.info("The crystalline region in z-direction: {}".format(crystal_regions))
        domain = domain_builder.atomic_system(*nxyz, *abc)
        domain.define_crystal_regions(crystal_regions)
        # domain.plot_system()
        domain_builder.test_bond_lengths(domain)
        a, b, c = abc
        m_cell = 28.0543e-3*2/avog # in kg
        v_cell = a*b*c*1e-30 # in m^3
        rho_crystal = m_cell/v_cell # in kg*m^-3
        domain.remove_atoms(1-options.rho_amorphous/rho_crystal, num_of_chains = options.num_of_chains)
        domain.write_lammps('initial_crystal.lammps')
        logging.info('Stem angle is {:.2f} degrees.'.format(degrees(domain.theta)))

    lmp = lammps.lammps()
    lmp_numpy = lmp.numpy

    cryst_group_command = ''
    cryst_id_compute_command = ""
    sub_command = \
    """
    region          cryst{i} block EDGE EDGE EDGE EDGE {bottom} {top}
    group           cryst_atom{i} region cryst{i}
    """
    for i, (bot, top) in enumerate(options.crystal_regions):
        cryst_group_command += sub_command.format(i = i, bottom = bot*options.lz + options.box[4],
                                                            top = top*options.lz + options.box[4])
        cryst_id_compute_command += 'compute         cryst_ids{i} cryst_atom{i} property/atom id\n'.format(i = i)

    cryst_group_command += "group cryst_atoms union" + \
        ''.join([' cryst_atom{}'.format(i) for i in range(options.num_crystal)])
    lmp.commands_string(commands.format(
                        cryst_group_command = cryst_group_command,
                        cryst_id_compute_command = cryst_id_compute_command))
    amorph_ids = lmp_numpy.extract_compute('amorph_ids', 
            lammps.LMP_STYLE_ATOM, lammps.LMP_TYPE_VECTOR).flatten().astype(int)
    amorph_ids = amorph_ids[amorph_ids != 0]
    amorph_ids_gather = comm.gather(amorph_ids, root = 0)
    cryst_ids = []
    for i in range(options.num_crystal):
        c_layer_ids = lmp_numpy.extract_compute('cryst_ids{}'.format(i), 
                                            lammps.LMP_STYLE_ATOM, 
                                            lammps.LMP_TYPE_VECTOR).flatten().astype(int)
        cryst_ids.append(c_layer_ids)
    cryst_ids_gather = comm.gather(cryst_ids, root = 0)
    
    if me == 0:
        amorph_ids_gather = numpy.concatenate(amorph_ids_gather)
        with open('amorph_ids.txt', 'w') as f:
            numpy.savetxt(f,sorted(amorph_ids_gather), fmt = '%3d', newline = ' ')
        cryst_ids_gather = numpy.concatenate(cryst_ids_gather, axis = 1)
        if os.path.exists('cryst_ids.txt'):
            os.remove('cryst_ids.txt')
        with open('cryst_ids.txt', 'a') as f:
            for i, _ in enumerate(cryst_ids_gather):
                c_layer = cryst_ids_gather[i]
                c_layer = c_layer[c_layer!=0]
                numpy.savetxt(f, sorted(c_layer), fmt = '%3d', newline = ' ')
                f.write('\n')

    lmp.close()


commands = """
atom_style      angle
read_data       initial_crystal.lammps
comm_modify     cutoff 16
reset_atom_ids  sort yes
reset_mol_ids   all compress yes
comm_modify     cutoff 0
write_data      initial_crystal.lammps
{cryst_group_command}
{cryst_id_compute_command}
group           amorph_atoms subtract all cryst_atoms
compute         amorph_ids amorph_atoms property/atom id
"""
if __name__ == '__main__':
    generate_crystalline_system()
