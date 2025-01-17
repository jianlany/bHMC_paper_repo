import os
import shutil
import re
import numpy
from math import cos, sin, atan
from mpi4py import MPI
import glob
comm = MPI.COMM_WORLD


class HMC_config:
    def __init__(self, inifile = None):
        self.debug = False
        self.num_states = 20

        self.repo_path = os.path.dirname(os.path.realpath(__file__)) + "/../"

        self.HB_pair_table = os.path.join(self.repo_path, "lammps_input/pair.table.CGHB")
        self.HB_angle_table = os.path.join(self.repo_path, "lammps_input/angle.table.CGHB")

        self.BA_pair_table = os.path.join(self.repo_path, "lammps_input/pair.table.EE.w1.8.1x")
        self.BA_angle_table = os.path.join(self.repo_path, "lammps_input/bond_angle.table.EEE.w1.8.1x")

        self.tail_mean_target = 100
        self.loop_mean_target = 30
        self.bridge_mean_target = 0

        self.tail_std = 100
        self.loop_std = 100
        self.bridge_std = 100

        # number of unit cells in x, y, and z direction
        self.nx = 15
        self.ny = 15
        self.nz = 100
        # lattice parameters in angstroms
        self.a, self.b, self.c = (8.99180, 5.18985, 2.47976)
        self.theta = atan(2.0*self.c/self.a)

        self.rho_amorphous = 752.0 # kg/m^3 752 is the value at 460K
        self.num_of_chains = int(self.nx*self.ny*0.24)
        # Angle between c axis and the normal of crystal plane., in rad
        # Crystal regions in relative dimensions
        self.crystal_regions = [[0, 0.3], [0.7, 1]]


        self.max_steps = 1000

        self.max_melt_num_steps = 60000
        self.max_anneal_num_steps = 80000
        self.max_sample_num_steps = 100000

        self.timestep = 5 # fs
        self.nvt_num_steps = 400
        self.nve_num_steps = 100
        self.nve_max_disp = 5 # angstrom
        self.nve_temp = 460 # K
        self.nvt_temp = 460
        self.Tsa = 3.5e5 # K

        self.search_radius = 6 # angstrom
        self.save_intv = 100
        self.topo_save_intv = 100
        self.forcefield = 'HB'

        if inifile:
            self.read_config_file(inifile)

        self.num_crystal = len(self.crystal_regions)

        self.lx = self.nx*self.a/cos(self.theta)
        self.ly = self.ny*self.b
        self.lz = self.nz*self.c*cos(self.theta)

        self.box = [0, self.lx, 0, self.ly, 0, self.lz]




    def read_config_file(self, config_file):
        with open(config_file, 'r') as f:
            try:
                for line in f:
                    if line.startswith('#'): continue
                    line = line.strip()
                    if not line: continue
                    line = line.split()
                    if line[0] == 'crystal':
                        if line[1] == 'a': self.a = float(line[2])
                        elif line[1] == 'b': self.b = float(line[2])
                        elif line[1] == 'c': self.c = float(line[2])
                        elif line[1] == 'nx': self.nx = int(line[2])
                        elif line[1] == 'ny': self.ny = int(line[2])
                        elif line[1] == 'nz': self.nz = int(line[2])
                        elif line[1] == 'theta': self.theta = numpy.deg2rad(float(line[2]))
                        elif line[1] == 'rho_amorph': self.rho_amorphous = float(line[2])
                        elif line[1] == 'regions':
                            assert len(line) % 2 == 0
                            self.crystal_regions = [[float(lo), float(hi)] for lo, hi in zip(line[2::2], line[3::2])]
                    elif line[0] == 'hmc':
                        if line[1] == 'max_melt_num_steps': self.max_melt_num_steps = int(line[2])
                        elif line[1] == 'max_anneal_num_steps': self.max_anneal_num_steps = int(line[2])
                        elif line[1] == 'max_sample_num_steps': self.max_sample_num_steps = int(line[2])
                        elif line[1] == 'Tsa': self.Tsa = float(line[2])
                        elif line[1] == 'nve_num_steps': self.nve_num_steps = int(line[2])
                        elif line[1] == 'nvt_num_steps': self.nvt_num_steps = int(line[2])
                        elif line[1] == 'timestep': self.timestep = float(line[2])
                        elif line[1] == 'radius': self.search_radius = float(line[2])
                        elif line[1] == 'tail_mean_target': self.tail_mean_target = int(line[2])
                        elif line[1] == 'loop_mean_target': self.loop_mean_target = int(line[2])
                        elif line[1] == 'bridge_mean_target': self.bridge_mean_target = int(line[2])
                        elif line[1] == 'tail_std': self.tail_std = float(line[2])
                        elif line[1] == 'loop_std': self.loop_std = float(line[2])
                        elif line[1] == 'bridge_std': self.bridge_std = float(line[2])
                        elif line[1] == 'forcefield': self.forcefield = line[2]
                        elif line[1] == 'save_interval': self.save_intv = int(line[2])
                        elif line[1] == 'topo_save_interval': self.topo_save_intv = int(line[2])
                    elif line[0] == 'debug':self.debug = True
                    else:
                        print("Unknown input command!")
                        print(line)
                        raise Exception
            except Exception as e:
                print(e)
                print(line)
                comm.Abort(2)

    def writer(self, filename):
        with open(filename, 'w') as f:
            for attr in dir(self):
                if attr.startswith('__'): continue
                if callable(getattr(self, attr)): continue
                f.write('{} {}\n'.format(attr, getattr(self, attr)))


dir_list = ['logs', 'states']
def setup_directories():
    """Make directories for the run."""
    for d in dir_list:
        if not os.path.exists(d):
            os.mkdir(d)


def remove_directories():
    """Remove files and directories to start fresh."""
    for d in dir_list:
        shutil.rmtree(d, ignore_errors = True)
    shutil.rmtree('trial.restart', ignore_errors = True)
    shutil.rmtree('initial_crystal.lammps', ignore_errors = True)
    shutil.rmtree('in.lammps', ignore_errors = True)
    shutil.rmtree('log.lammps', ignore_errors = True)
    for f in glob.glob('*txt'):
        os.remove(f)


def get_most_recent_data(data_files):
    """ Find the state with the largest step when starting a job"""
    get_step_num = lambda f: int(re.match('states/hmc_step(\d+).lammps', f).group(1))
    data = max(data_files, key = get_step_num)
    step = get_step_num(data)
    return data, step

