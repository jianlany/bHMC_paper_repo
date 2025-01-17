import sys
from math import cos, sin, atan
import numpy
import itertools
import logging
import random
from copy import deepcopy


class atomic_system:
    def __init__(self, nx, ny, nz, a, b, c, theta = None):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.num_atoms = 2*self.nx*self.ny*self.nz
        self.atom_ids = numpy.arange(self.num_atoms)
        self.atoms = numpy.zeros((self.num_atoms, 3))
        # Lattice constants
        self.a = a
        self.b = b
        self.c = c
        # Angle between c axis and the normal of crystal plane., in rad
        if theta is not None: self.theta = theta
        else: self.theta = atan(2.0*c/a)
        # distance between unit cells
        self.dx = self.a/cos(self.theta)
        self.dz = self.c*cos(self.theta)
        self.dy = self.b
        self.box = [0.0, self.nx*self.dx, 
                    0.0, self.ny*self.b, 
                    0.0, self.nz*self.dz]
        logging.info("The system dimensions: dx {:.4f}, dy {:.4f}, dz {:.4f}.".format(*self.box[1::2]))
        self.connection = [[] for i in range(len(self.atoms))]

        self.__build_atoms()
        self.__build_bonds()


    def __build_atoms(self):
        ''' Adds atoms to system but does not yet wrap to the box. '''
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    x = -k*self.c*sin(self.theta) + i*self.dx
                    idx = 2*(i + self.nx*j + self.nx*self.ny*k)
                    self.atoms[idx,:]    = (x,                   j*self.b, k*self.dz)
                    self.atoms[idx+1, :] = (x+0.5*self.dx, (j+0.5)*self.b, k*self.dz)
        logging.info("{} atoms before delete.".format(len(self.atoms)))


    def __build_bonds(self):
        ''' Adds bonds to system (including across periodic boundaries). '''
        def wrap_x(x):
            lx = self.box[1] - self.box[0]
            while x < self.box[0] or x > self.box[1]:
                if x < self.box[0]:
                    x += lx
                if x > self.box[1]:
                    x -= lx
            return x

        self.bonds = []
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    idx = 2*(i + self.nx*j + self.nx*self.ny*k)
                    idx2 = idx + 2*self.nx*self.ny
                    if idx2 >= len(self.atoms):
                        x = self.atoms[idx,0] - self.c*numpy.sin(self.theta)
                        x = wrap_x(x)
                        i_bottom = int(round(x/self.dx))%self.nx
                        idx2 = 2*self.nx*j + 2*i_bottom
                    self.bonds.append((idx, idx2))
                    self.bonds.append((idx+1, idx2+1))
                    self.connection[idx].append(idx2)
                    self.connection[idx2].append(idx)
                    self.connection[idx+1].append(idx2+1)
                    self.connection[idx2+1].append(idx+1)
        self.bonds = numpy.array(self.bonds)

        self.angles = []
        for i,a in enumerate(self.connection):
            for j1, j2 in itertools.combinations(a, 2):
                self.angles.append((j1, i, j2))
        self.angles = numpy.array(self.angles)

        logging.info("{} bonds before delete.".format(len(self.bonds)))
        logging.info("{} angles before delete.".format(len(self.angles)))


    def define_crystal_regions(self, crystal_regions):
        ''' Defines which regions of atoms (z-slices) will be crystalline
        then applies a shear in the x-z plane in the amorphous regions
        so that the bonds between the top and bottom of the domain are not
        distorted. '''
        z_length = self.box[5] - self.box[4]
        is_crystal = numpy.logical_or.reduce([(self.atoms[:,2] >= self.box[4] + c[0]*z_length) &
                                              (self.atoms[:,2] <= self.box[4] + c[1]*z_length)
                                              for c in crystal_regions])
        self.amorphous_atoms = numpy.nonzero(numpy.logical_not(is_crystal))[0]
        self.crystal_atoms = numpy.nonzero(is_crystal)[0]

        # Shift amorphous atoms so that domain is periodic along z-direction.
        na = len(self.amorphous_atoms) // (2*self.nx*self.ny)
        x_next = self.atoms[-2,0] - self.c*sin(self.theta)
        # First row of atoms won't be shifted, so don't count it in na.
        if self.amorphous_atoms[0] == 0:
            shift = (self.dx*round(x_next/self.dx) - x_next) / (na - 1)
        else:
            shift = (self.dx*round(x_next/self.dx) - x_next) / na
        for k in range(1, self.nz):
            idx = 2*self.ny*self.nx*k
            if idx in self.amorphous_atoms:
                self.atoms[idx:,0] += shift
        # Wrapping atoms to the box.
        self.wrap_atoms()


    def remove_atoms(self, fraction, num_of_chains, seed = None):
        '''remove a fraction of atoms in the amorphous region, only one atom will 
        be removed from each selected stem'''
        if seed:
            random.seed(seed)
        amorph_atom_id = list(self.amorphous_atoms)
        amorph_atom_id_set = set(amorph_atom_id)
        # use DFS to find all the atom ids in each stem in amorphous region
        stems = []
        seen = set()
        while amorph_atom_id:
            i = amorph_atom_id.pop()
            if i in seen: continue
            seen.add(i)
            stem = [i]
            queue = deepcopy(self.connection[i])
            while queue:
                j = queue.pop()
                if j not in amorph_atom_id_set: continue
                if j in seen: continue
                seen.add(j)
                stem.append(j)
                for k in self.connection[j]:
                    if k not in seen:
                        queue.append(k)
            stems.append(stem)
        assert(len(stems) == 2*self.nx*self.ny)

        atoms_to_remove = []
        num_of_atoms_to_remove = int(len(self.amorphous_atoms)*fraction)
        # break n stems in an infinite chain would result in n finite molecule chains
        num_of_stems_to_remove_from = num_of_chains
        num_of_atoms_to_remove_stem = int(num_of_atoms_to_remove/num_of_stems_to_remove_from)
        logging.info("Stem length in amorphous phase: {}.".format(len(stems[0]))) 
        logging.debug("Number of atoms needs to be removed: {}.".format(num_of_atoms_to_remove)) 
        logging.debug("Number of stems to remove atoms from: {}.".format(num_of_stems_to_remove_from))
        # # make sure after deletion, the tails have at least 1 bead
        assert (len(stems[0]) - num_of_atoms_to_remove_stem >= 2), "deleting too many beads from each stem, only {} beads remain in the stems.".format(len(stems[0]) - num_of_atoms_to_remove_stem)

        # pick stems to remove from, and pick a starting point to remove a sequence of bonded atoms,
        # avoiding short chains in amorphous region
        for s in random.sample(stems, k = num_of_stems_to_remove_from):
            # we should have at least 1 atom on each side after removing atoms)
            assert(len(s) - num_of_atoms_to_remove_stem >= 2)
            start = random.choice(range(1,len(s)- 1 - num_of_atoms_to_remove_stem))
            for i in range(start, start + num_of_atoms_to_remove_stem):
                atoms_to_remove.append(s[i])
        logging.debug("{} atoms to remove.".format(len(atoms_to_remove)))

        # Remove the atom coords and atom ids
        self.atoms = numpy.delete(self.atoms, atoms_to_remove, axis = 0)
        self.atom_ids = numpy.delete(self.atom_ids, atoms_to_remove, axis = 0)

        atoms_to_remove_set = set(atoms_to_remove)

        # Remove bond records
        bonds_to_remove = []
        for i, b in enumerate(self.bonds):
            b1, b2 = b
            if (b1 in atoms_to_remove_set) or (b2 in atoms_to_remove_set):
                bonds_to_remove.append(i)
        logging.debug("{} bonds to remove.".format(len(bonds_to_remove)))
        self.bonds = numpy.delete(self.bonds, bonds_to_remove, axis = 0)
        # remove angle records
        angles_to_remove = []
        for i, a in enumerate(self.angles):
            a1, a2, a3 = a
            if (a1 in atoms_to_remove_set) or \
               (a2 in atoms_to_remove_set) or \
               (a3 in atoms_to_remove_set):
                angles_to_remove.append(i)
        logging.debug("{} angles to remove.".format(len(angles_to_remove)))
        self.angles = numpy.delete(self.angles, angles_to_remove, axis = 0)
        logging.info("{} atoms after delete.".format(len(self.atoms)))
        logging.info("{} bonds after delete.".format(len(self.bonds)))
        logging.info("{} angles after delete.".format(len(self.angles)))


    def unwrapped_distance_vector(self, i, j):
        ''' Returns the distance vector between atoms i and j. '''
        def wrap(v, l):
            while v > 0.5*l: 
                v -= l
            while v < -0.5*l: 
                v += l 
            return v
        r = self.atoms[j,:] - self.atoms[i,:]
        r[0] = wrap(r[0], self.box[1]-self.box[0])
        r[1] = wrap(r[1], self.box[3]-self.box[2])
        r[2] = wrap(r[2], self.box[5]-self.box[4])
        return r

    def wrap_atoms(self):
        for a in self.atoms:
            while a[0] < 0.0:
                a[0] += self.box[1]
            while a[0] >= self.box[1]:
                a[0] -= self.box[1]

    def write_lammps(self, path):
        with open(path, 'w') as fid:
            fid.write('LAMMPS data file built by scg\n\n')
            fid.write(' {} atoms\n'.format(len(self.atoms)))
            fid.write(' {} atom types\n\n'.format(1))
            fid.write(' {} bonds\n'.format(len(self.bonds)))
            fid.write(' {} bond types\n\n'.format(1))
            # This only works on linear chains
            fid.write(' {} angles\n'.format(len(self.atoms)-2*(len(self.atoms) - len(self.bonds))))
            fid.write(' {} angle types\n\n'.format(1))
            fid.write(' {:.6f} {:.6f} xlo xhi\n'.format(*self.box[:2]))
            fid.write(' {:.6f} {:.6f} ylo yhi\n'.format(*self.box[2:4]))
            fid.write(' {:.6f} {:.6f} zlo zhi\n'.format(*self.box[4:6]))
            fid.write('\nMasses\n\n')
            fid.write('1 28.0543\n')

            fid.write('\nAtoms\n\n')
            for i,a in zip(self.atom_ids, self.atoms):
                mol = 1
                atom_type = 1
                fid.write('{} {} {} {} {} {}\n'.format(i+1, mol, atom_type, *a))

            fid.write('\nBonds\n\n')
            bond_type = 1
            for i,b in enumerate(self.bonds):
                fid.write('{} {} {} {}\n'.format(i+1, bond_type, b[0]+1, b[1]+1))

            fid.write('\nAngles\n\n')
            angle_type = 1
            for i,a in enumerate(self.angles):
                fid.write('{} {} {} {} {}\n'.format(i+1, angle_type, a[0]+1, a[1]+1, a[2]+1))


    def plot_system(self):
        ''' Draws the xz plane of the system showing the amorphous and
        crystalline domains. '''
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot
        fig, ax = pyplot.subplots(1, 1, figsize=(4,8))

        x = self.atoms
        c, a = self.crystal_atoms, self.amorphous_atoms
        pp = dict(s=2)
        ax.scatter(x[c,0], x[c,2], **pp)
        ax.scatter(x[a,0], x[a,2], **pp)
        # ax.axvline(x=x[0,0])
        # ax.axvline(x=x[2,0])
        # ax.axvline(x=10.73)
        ax.set_ylabel('y ($\mathrm{\AA})$')
        ax.set_xlabel('x ($\mathrm{\AA})$')

        # m = x[:,2] < 10.0
        # ax.scatter(x[m,0], x[m,2]+self.box[-1], **pp)  # fc='gray',
        # m = x[:,0] < 10.0
        # ax.scatter(x[m,0] + self.box[1], x[m,2], **pp) #  fc='gray',
        ax.set_aspect('equal')
        fig.tight_layout()
        pyplot.savefig('system.png', dpi=300)


def test_bond_lengths(system):
    ''' Checks that range of bond lengths is not too large.  
    Bonds should slighly vary in length due to the shearing of the domain 
    needed to maintain perodicity. '''
    rr = set()
    for b in system.bonds:
        r = numpy.linalg.norm(system.unwrapped_distance_vector(*b))
        rr.add(round(r, 6))

    deviation = max(rr)/min(rr) - 1.0
    if deviation > 0.1: 
        logging.warning('Bond lengths vary by {:.1f}%. Max: {:.2f}, min: {:.2f}'.format(100.0*deviation, max(rr), min(rr)))

