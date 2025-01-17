import os
from mpi4py import MPI
from scipy.spatial import cKDTree
import time
import logging
comm = MPI.COMM_WORLD
import numpy
import lammps
import random
from itertools import zip_longest

from .topo_states import State

kB = 1.987204259e-3 # kcal/(mol*K)


class Semicrystalline_hmc:
    def __init__(self, options):
        self.lmp = lammps.lammps(cmdargs = ['-log', 'none'])# cmdargs = ['-screen', 'none']
        self.options = options
        self.lmp_numpy = self.lmp.numpy
        self.me = comm.Get_rank()
        # The simulation box bounds
        self.box = options.box
        # The dimension of the simulation box
        self.boxsizes = [self.box[1] - self.box[0],
                         self.box[3] - self.box[2],
                         self.box[5] - self.box[4]]

        self.r = options.search_radius
        self.cmd_to_write = ''

        self.accepted = None

        # Numpy array of integers, contains the amorphous bead ids extracted from LAMMPS, 1-based. Do not change over the steps.
        self.amorph_ids_gather = None
        # Numpy array of numpy array of integers, each array inside represents a crystalline layers.
        # and contains the 1-based bead ids. Do not change over the steps.
        self.cryst_ids_gather = None
        # Numpy array of integers, contains the negative values of the id of the crystalline phase a bead is connected to, for bridges and crystalline beads values are 1.
        self.cryst_anchor = None
        # 2d Numpy array, contains the bead coordinates of the beads in the system, are updated every step
        self.coords_gather = None
        # Python set, contains the ids of beads that belong to tails, changes when the topology changes
        self.tail_atoms = None
        # Python set, contains the ids of beads that belong to loops, changes when the topology changes
        self.loop_atoms = None
        # Python set, contains the ids of beads that belong to bridges, changes when the topology changes
        self.bridge_atoms = None
        # List of List of integers, each nested list represents a tail, which contains the ids of beads belong to that tail, changes when the topology changes
        self.tails = None
        # List of List of integers, each nested list represents a loop, which contains the ids of beads belong to that loop, changes when the topology changes
        self.loops = None
        # List of List of integers, each nested list represents a bridge, which contains the ids of beads belong to that bridge, changes when the topology changes
        self.bridges = None
        # Numpy array of integers, length is the same as the number of beads in the system
        # The shortest topological distance of an amorphous bead from the crystalline phases, 
        # For crystalline beads, it contains the negative value of the ids of the crystalline phase, 0-based 
        # changes when the topology changes
        self.distance_from_crystalline = None
        # Numpy array of integers, length is the same as the number of beads in the system
        # The length of the segment that a bead belongs to
        # 0 for crystalline beads
        # changes when the topology changes
        self.segment_lengths_atom = None
        # Python set, contains the ids of the end beads of tails, changes when the topology changes
        self.tail_ends = None

    def setup(self, datafile):
        """
        Set up the LAMMPS simulations, such as read_data, set force fields, etc.
        """

        self.read_ids()
        all_cryst_ids = numpy.concatenate(self.cryst_ids_gather, axis = 0)
        assert len(all_cryst_ids) == sum([array.size for array in self.cryst_ids_gather])
        cryst_ids_str = ' '.join(all_cryst_ids.astype(str))
        cryst_group_command = 'group cryst_atoms id {}'.format(cryst_ids_str)
        setup_cmd = setup_command.format(datafile = datafile, 
                                         cryst_group_command = cryst_group_command)

        self.lmp.commands_string(setup_cmd)


        if self.options.forcefield == 'HB':
            self.force_field_command = HB_forcefield_command.format(
                    pair_table = self.options.HB_pair_table,
                    angle_table = self.options.HB_angle_table)
        elif self.options.forcefield == 'BA':
            self.force_field_command = BA_forcefield_command.format(
                    pair_table = self.options.BA_pair_table,
                    angle_table = self.options.BA_angle_table)

        self.lmp.commands_string(self.force_field_command)
        self.lmp.commands_string(initial_run_command.format(
            md_dt = self.options.timestep))
        if self.me == 0:
            self.cmd_to_write+=setup_cmd \
                    + self.force_field_command \
                    + initial_run_command

        self.natoms = self.lmp.get_natoms()
        self.distance_from_crystalline = numpy.zeros(self.natoms)
        self.segment_lengths_atom = numpy.zeros(self.natoms)
        self.cryst_anchor = numpy.ones(self.natoms)


        if self.me == 0:
            # Record the crystalline layer where the crystalline atoms belong to
            # To distinguish from the distance from crystalline atoms
            # The crystalline layers are labeled with 0 or negative number
            for i, cryst_layer_id in enumerate(self.cryst_ids_gather):
                for j in cryst_layer_id:
                    self.distance_from_crystalline[j-1] = -i

    def hmc_step1(self, step, substep):
        if step < self.options.max_melt_num_steps:
            nve_setting = "nve/limit {}".format(self.options.nve_max_disp)
            temp_setting= 'fix 3 amorph_atoms temp/rescale 1 {0} {0} 10 1.0'.format(self.options.nve_temp),
            temp_unfix = 'unfix 3'
        else:
            nve_setting = "nve"
            temp_setting = ''
            temp_unfix = ''

        step1_cmd = run_commands_step1.format(velocity_seed = random.randrange(1, int(1e5)),
                                          step = step,
                                          substep = substep,
                                          md_dt = self.options.timestep,
                                          nvt_run_steps = self.options.nvt_num_steps,
                                          nve_run_steps = self.options.nve_num_steps,
                                          nve_setting = nve_setting,
                                          temp_setting= temp_setting,
                                          temp_unfix= temp_unfix)
        self.lmp.commands_string(step1_cmd)

        pe = self.lmp.extract_compute('thermo_pe', lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_SCALAR)
        ke = self.lmp.extract_compute('thermo_ke', lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_SCALAR)
        if self.me == 0:
            self.initial_te = pe + ke
            logging.info('Initial total energy: {:.2f} kcal/mol.'.format(self.initial_te))

        if self.me == 0 and step == 0 and substep == 0:
            self.cmd_to_write+=step1_cmd


    def attack_bonds(self, step, substep):
        # When accepted, the atom coords and atom id does not change from last gather
        # in evaluate acceptance
        if not self.accepted:
            self.gather_coords_and_ids()
        bonds = self.lmp_numpy.gather_bonds().astype(int)
        if self.me == 0:
            logging.info('*****************************')
            self.build_conn(bonds)
            self.tail_ends = set()
            for i in self.amorph_ids_gather:
                if len(self.conn[i-1]) == 1:
                    self.tail_ends.add(i)
            logging.info("Number of tail ends: {}.".format(len(self.tail_ends)))
            assert len(self.tail_ends) == 2*self.options.num_of_chains, \
                    "Number of tail ends is wrong,"\
                    " current number: {}".format(len(self.tail_ends))

            self.find_topologies()
            logging.info("There are {} tails, {} loops, and {} bridges "\
                    "before bond attack at step {}, substep {}.".format(
                    len(self.tails), 
                    len(self.loops), 
                    len(self.bridges), 
                    step, substep))
            assert len(self.tails) + len(self.loops) + len(self.bridges) == \
                    2*self.options.nx*self.options.ny + self.options.num_of_chains, \
                    "Total number of segments is wrong."

            logging.info("Before bond attack, mean tail length: {:.2f}, "\
                                             "mean loop length: {:.2f}, "\
                                             "mean bridge length: {:.2f}.".\
                    format(len(self.tail_atoms)/len(self.tails), 
                           len(self.loop_atoms)/len(self.loops) if len(self.loops) else 0, 
                           len(self.bridge_atoms)/len(self.bridges) if len(self.bridges) else 0))
            logging.info("There are {} tail atoms, {} loops atoms , and {} bridge atoms "\
                    "before bond attack at step {} substep {}.".format(
                    len(self.tail_atoms), 
                    len(self.loop_atoms), 
                    len(self.bridge_atoms), 
                    step, substep))

            states = []
            ps = []
            state_lengths = []
            state_nums = []
            seen = set()
            start = time.perf_counter()
            kdt = cKDTree(self.coords_gather, boxsize = self.boxsizes)
            logging.info("Building KD tree takes {:.2f} ms.".format(1000*(time.perf_counter() - start)))
            for j in range(1, self.options.num_states+1):# self.options.num_states+1
                logging.info("state {}:".format(j))
                ba, bt, bs = self.generate_attack_state(kdt)
                if (ba is None) or (bt is None) or (bs is None): continue
                if (ba, bt, bs) in seen: continue
                state = State(ba, bt, bs)
                p, mean_tail_length, mean_loop_length, mean_bridge_length \
                        = state.calc_probability(self, self.options)
                logging.info("\tba:{}, bt:{}, bs:{}, relative p: {}.".format(ba, bt, bs, p))
                seen.add((ba, bt, bs))
                states.append((ba, bt, bs))
                state_lengths.append((mean_tail_length, mean_loop_length, mean_bridge_length))
                ps.append(p)
                state_nums.append(j)
            ps = numpy.array(ps)
            chosen_number = random.choices(range(len(state_nums)), weights = ps, k = 1)[0]

            self.ba, self.bt, self.bs = states[chosen_number]
            selected_lengths = state_lengths[chosen_number]

            # Update the connectivity
            del self.conn[self.bt-1][self.conn[self.bt-1].index(self.bs)]
            del self.conn[self.bs-1][self.conn[self.bs-1].index(self.bt)]
            self.tail_ends.remove(self.ba)
            self.tail_ends.add(self.bs)
            self.conn[self.bt-1].append(self.ba)
            self.conn[self.bt-1] = sorted(self.conn[self.bt-1])
            self.conn[self.ba-1].append(self.bt)
            self.conn[self.ba-1] = sorted(self.conn[self.ba-1])
            neigh_ba = self.conn[self.ba-1]
            neigh_bt = self.conn[self.bt-1]
            assert(len(self.conn[self.bs-1]) == 1)
            assert(len(neigh_ba) == 2)
            neigh_bs = self.conn[self.bs-1][0]

            # After changing topology lets just recheck the tails, loops, and bridges
            self.find_topologies()
            logging.info("There are {} tails, {} loops, and {} bridges "\
                    "after bond attack at step {}, substep {}.".format(
                        len(self.tails), 
                        len(self.loops), 
                        len(self.bridges), 
                        step, substep))
            logging.info("After bond attack, mean tail length: {:.2f}, "\
                                            "mean loop length: {:.2f}, "\
                                            "mean bridge length: {:.2f}.".\
                    format(len(self.tail_atoms)/len(self.tails), 
                           len(self.loop_atoms)/len(self.loops) if len(self.loops) else 0, 
                           len(self.bridge_atoms)/len(self.bridges) if len(self.bridges) else 0))
            logging.info("There are {} tail atoms, {} loops atoms, and {} bridge atoms"\
                    " after bond attack at step {} substep {}.".format(
                        len(self.tail_atoms), 
                        len(self.loop_atoms), 
                        len(self.bridge_atoms), 
                        step, substep))

            logging.info('State {} is selected: ba {}, bt {}, bs {}.'.format(state_nums[chosen_number], self.ba, self.bt, self.bs))
            logging.info('Resulting mean tail length: {}, mean loop length: {}, and mean bridge length: {}'.format(*selected_lengths))
            assert self.atom_ids_gather[self.ba-1] == self.ba
            assert self.atom_ids_gather[self.bt-1] == self.bt
            assert self.atom_ids_gather[self.bs-1] == self.bs
            logging.debug('ba coord: {}, bt coord: {}.'.format(self.coords_gather[self.ba-1], self.coords_gather[self.bt-1]))
            logging.info('*****************************')
        else:
            self.ba = self.bt = self.bs = neigh_ba = neigh_bt = neigh_bs = None
        self.ba, self.bt, self.bs, neigh_ba, neigh_bt, neigh_bs = \
                comm.bcast((self.ba, self.bt, self.bs, neigh_ba, neigh_bt, neigh_bs), root = 0)
        angle_todelete_ids = '{} {} {} '.format(
                self.bt, self.bs, neigh_bs) + \
                ' '.join([str(i) for i in neigh_bt if i!=self.ba])
        atk_cmds = attack_commands.format(ba = self.ba,
                                          bt = self.bt,
                                          bs = self.bs,
                                          ba_neigh1 = neigh_ba[0],
                                          ba_neigh2 = neigh_ba[1],
                                          angle_todelete_ids = angle_todelete_ids)
        if len(neigh_bt) == 2:
            atk_cmds += bt_angle_cmd.format(bt = self.bt,
                                            bt_neigh1 = neigh_bt[0],
                                            bt_neigh2 = neigh_bt[1])
        if self.me == 0 and step == 0 and substep == 0:
            self.cmd_to_write+=atk_cmds

        self.lmp.commands_string(atk_cmds)
        bonds = self.lmp_numpy.gather_bonds()
        if self.me == 0 and self.options.debug:
            # Making sure everything is correct
            # can be removed if slowing down the code with large systems
            conn = self.build_conn(bonds, overwrite = False)
            assert conn == self.conn
            tail_ends = set()
            for i in self.amorph_ids_gather:
                if len(self.conn[i-1]) == 1:
                    tail_ends.add(i)
            if tail_ends != self.tail_ends:
                print("Tail ends checking failed!.")
                print("What I have:", self.tail_ends)
                print("From lammps:", tail_ends)
                comm.Abort(2)



    def hmc_step2(self, step, substep):
        step2_cmd = run_commands_step2.format(md_dt = self.options.timestep,
                                              velocity_seed = random.randrange(1, int(1e5)),
                                              step = step,
                                              substep = substep,
                                              nvt_run_steps = self.options.nvt_num_steps,
                                              nve_run_steps = self.options.nve_num_steps,
                                              nve_setting = "nve/limit {}".format(self.options.nve_max_disp),
                                              temp_setting= 'fix 3 amorph_atoms temp/rescale 1 {0} {0} 10 1.0'.format(self.options.nve_temp),
                                              temp_unfix= 'unfix 3')
        if self.me == 0 and step == 0 and substep == 0:
            self.cmd_to_write+= step2_cmd
            with open('in.lammps', 'w') as f:
                f.write(self.cmd_to_write)
        self.lmp.commands_string(step2_cmd)
        pe = self.lmp.extract_compute('thermo_pe', lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_SCALAR)
        ke = self.lmp.extract_compute('thermo_ke', lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_SCALAR)
        if self.me == 0:
            self.final_te = pe + ke
            logging.info('Final total energy of step {} substep {}: {:.2f} kcal/mol.'.format(step, substep, self.final_te))


    def evaluate_acceptance(self, step, substep, Tsa):
        """Evaluate if the current move should be accepted"""
        self.gather_coords_and_ids()
        if self.me == 0:
            current_tail = self.find_current_segment(self.bs)
            xs = self.coords_gather[self.bs-1]
            start = time.perf_counter()
            kdt = cKDTree(self.coords_gather, boxsize = self.boxsizes)
            self.ns = 0
            for ii in kdt.query_ball_point(xs, self.r):
                i = ii+1
                if i in current_tail: continue
                if self.distance_from_crystalline[i-1] < 2: continue
                self.ns+=1
            logging.info("KD tree seaching for available targets in eval takes {:.2f} ms".format((time.perf_counter() - start)*1000))

            if self.options.debug == True:
                start = time.perf_counter()
                ns_regular = 0
                for i in self.amorph_ids_gather:
                    if i in current_tail: continue
                    if self.distance_from_crystalline[i-1] < 2: continue
                    if self.in_distance(self.coords_gather[i-1] - xs, self.r):
                        ns_regular += 1
                print("Regular seaching for available targets takes {:.2f} ms".format((time.perf_counter() - start)*1000))
                assert ns_regular == self.ns

            if self.ns != 0:
                te_drop = self.initial_te - self.final_te
                p = min(1, self.na/self.ns*numpy.exp(te_drop/kB/Tsa))
                logging.debug("total energy drop {:.2f} kcal/mol, Number of available target ratio before/after: {:.2f}.".format(te_drop, float(self.na)/self.ns))
                logging.info('Probability of acceptance: {:.2f}%.'.format(p*100))
                self.accepted = random.random() < p
            else: 
                logging.warning("0 available targets after NVT at step {} substep {}, state rejected.".format(step, substep))
                self.accepted = False
        else:
            self.accepted = None
        self.accepted = comm.bcast(self.accepted, root = 0)
        if not self.accepted:
            if self.me == 0:
                logging.info("State rejected! Reading restart...")
            self.lmp.close()
            self.lmp = lammps.lammps(cmdargs = ['-log', 'none'])
            self.lmp_numpy = self.lmp.numpy
            self.lmp.command("read_restart trial.restart")
            self.lmp.commands_string(re_setup_commands + self.force_field_command)
        else:
            if self.me == 0:
                logging.info("State accepted!")
            if (step % self.options.save_intv == 0):
                self.lmp.command("write_data states/hmc_step{}.lammps".format(step))
            if step % self.options.topo_save_intv == 0 and self.me == 0:
                self.write_topology_info(step)
        return self.accepted


    def build_conn(self, bonds, overwrite = True):
        """ Build connectivity matrix for they system"""
        conn = [[] for _ in range(self.natoms)]
        for _, b1, b2 in bonds:
            conn[b1-1].append(b2)
            conn[b2-1].append(b1)
        for i, _ in enumerate(conn):
            conn[i] = sorted(conn[i])
        if overwrite: self.conn = conn.copy()
        return conn


    def find_topologies(self):
        """ Find the topological belongings of amorphous atoms"""
        amorph_ids_set = set(self.amorph_ids_gather)
        self.tail_atoms = set()
        self.tails = []
        seen = set()
        for i in self.tail_ends:
            if i in seen: continue
            if i not in amorph_ids_set: continue
            seen.add(i)
            stack = [i]
            tail = []
            anchor = None
            while stack:
                j = stack.pop()
                tail.append(j)
                for jj in self.conn[j-1]:
                    if jj in seen: continue
                    if jj not in amorph_ids_set: 
                        anchor = self.distance_from_crystalline[jj-1]
                        continue
                    seen.add(jj)
                    stack.append(jj)
            # reverse the tail list, record the distance from crystalline domain
            self.tails.append(tail)
            assert anchor is not None
            for d, j in enumerate(tail[::-1]):
                self.tail_atoms.add(j)
                self.distance_from_crystalline[j-1] = d + 1
                self.segment_lengths_atom[j-1] = len(tail)
                self.cryst_anchor[j-1] = anchor

        self.bridge_atoms = set()
        self.loop_atoms = set()
        self.bridges = []
        self.loops = []
        for i in self.amorph_ids_gather:
            # Start with an amorphous bead that connects to a crystalline phase
            if i in self.tail_atoms: continue
            if i in seen: continue
            assert len(self.conn[i-1]) == 2
            if (self.conn[i-1][0] in amorph_ids_set) == (self.conn[i-1][1] in amorph_ids_set): continue
            # the crystalline bead that i connects to
            end1 = self.conn[i-1][0] \
                if self.conn[i-1][0] not in amorph_ids_set \
                else self.conn[i-1][1]
            end2 = None
            stack = [i]
            seen.add(i)
            segment = []
            while stack:
                j = stack.pop()
                segment.append(j)
                for jj in self.conn[j-1]:
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
            anchor1 = self.distance_from_crystalline[end1-1]
            anchor2 = self.distance_from_crystalline[end2-1]
            if anchor1 == anchor2:
                segments = self.loops
                segments_set = self.loop_atoms
            else:
                segments = self.bridges
                segments_set = self.bridge_atoms
            segments.append(segment)
            for d, j in enumerate(segment):
                segments_set.add(j)
                self.distance_from_crystalline[j-1] = min(d+1, len(segment)-d)
                self.segment_lengths_atom[j-1] = len(segment)
                if segments is self.loops:
                    self.cryst_anchor[j-1] = anchor1


        if self.options.debug:
            assert len(seen) == len(amorph_ids_set)
            for i in self.amorph_ids_gather:
                assert self.distance_from_crystalline[i-1] > 0
            for cryst_layer_atom_id in self.cryst_ids_gather:
                for j in cryst_layer_atom_id:
                    assert self.distance_from_crystalline[j-1] <= 0


    def find_current_segment(self, i):
        """
        Find the segment containing the given atom id
        """
        assert (i in self.amorph_ids_gather)
        stack = [i]
        seen = set([i])
        segment = set()
        while stack:
            j = stack.pop()
            segment.add(j)
            for jj in self.conn[j-1]:
                if jj in seen: continue
                if jj not in self.amorph_ids_gather: continue
                seen.add(jj)
                stack.append(jj)
        return segment


    def in_distance(self, vec, r):
        """ Calculates if the length of a vector is shorter than radius r."""
        for i, d in enumerate(self.boxsizes):
            while vec[i] >  0.5*d: vec[i] -= d
            while vec[i] < -0.5*d: vec[i] += d
        ###########
        # if any(abs(v)>r) for v in vec
        ###########
        for v in vec:
            if abs(v) > r: return False
        if numpy.dot(vec,vec) < r*r: return True
        return False


    def wrap_coords(self):
        """
        vectorize it
        m = self.coords_gather[:,0] > self.boxsize[0]
        self.coords_gather[m,0] -= self.boxsize[0]
        """

        start = time.perf_counter()
        for i, _ in enumerate(self.coords_gather):
            for j, d in enumerate(self.boxsizes):
                while self.coords_gather[i,j] > d: self.coords_gather[i,j] -= d
                while self.coords_gather[i,j] < 0: self.coords_gather[i,j] += d
        logging.info("Wrap coords take {:.2f} ms.".format((time.perf_counter() - start)*1000))


    def gather_coords_and_ids(self):
        coords = self.lmp_numpy.extract_atom('x')
        coords_gather = comm.gather(coords, root = 0)
        atom_ids = self.lmp_numpy.extract_compute('all_ids', 
                lammps.LMP_STYLE_ATOM, lammps.LMP_TYPE_VECTOR).flatten().astype(int)
        atom_ids = atom_ids[atom_ids != 0]
        atom_ids_gather = comm.gather(atom_ids, root = 0)
        if self.me == 0:
            self.atom_ids_gather = numpy.concatenate(atom_ids_gather)
            self.coords_gather = numpy.concatenate(coords_gather)
            self.coords_gather = numpy.array([x for _, x in sorted(zip(self.atom_ids_gather, self.coords_gather), key = lambda pair: pair[0])])
            self.atom_ids_gather = sorted(self.atom_ids_gather)
            self.wrap_coords()


    def generate_attack_state(self, kdt):
        """ Randomly generate ba, bt, and bs to satisfy the restriction."""
        # ba is the attacking bead, bt is the target, bs is bt's neighbor
        # ba and bt will form new bond, bs will be the new tail end.
        ba = random.choice(list(self.tail_ends))
        current_tail = self.find_current_segment(ba)
        xa = self.coords_gather[ba-1]

        start = time.perf_counter()
        available_targets = []
        for ii in kdt.query_ball_point(xa, self.r):
            i = ii+1
            # bt cannot be on the same tail as ba
            if i in current_tail: continue
            # Make sure the resulting loop length >= 3 and tail length >= 1
            if self.distance_from_crystalline[i-1] < 2: continue
            # assert self.in_distance(xa - self.coords_gather[i-1], self.r), 'atom id {}, coord {}'.format(i, self.coords_gather[i-1])
            available_targets.append(i)
        logging.info("\tkdtree searching takes {:.5f} ms.".format(1000*(time.perf_counter() - start)))
        if self.options.debug:
            start = time.perf_counter()
            available_targets_regular = []
            for i in self.amorph_ids_gather:
                # bt cannot be on the same tail as ba
                if i in current_tail: continue
                # Make sure the resulting loop length >= 3 and tail length >= 1
                if self.distance_from_crystalline[i-1] < 2: continue
                if self.in_distance(self.coords_gather[i-1] - xa, self.r):
                    available_targets_regular.append(i)
            print("\tRegular searching takes {:.5f} seconds.".format(time.perf_counter() - start))
            assert sorted(available_targets)  == sorted(available_targets_regular)

        if not available_targets:
            logging.warning("0 available targets found in this state, skipping")
            return None, None, None

        logging.info("\tThere are {} available target beads for attacking in a radius of {} angstrom.".format(len(available_targets), self.r))
        self.na = len(available_targets)
        bt = random.choice(available_targets)
        # We do not choose bs that is farther from crystalline phase than bt if they belong to a tail
        # to avoid having short chains floating around in the amorphous phase
        if bt in self.tail_atoms:
            if len(self.conn[bt-1]) == 2:
                neigh1, neigh2 = self.conn[bt-1]
                bs = neigh1 if self.distance_from_crystalline[neigh1-1] \
                             < self.distance_from_crystalline[neigh2-1] \
                               else neigh2
            else:
                bs = self.conn[bt-1][0]
        else:
            bs = random.choice(self.conn[bt-1])
        return ba, bt, bs


    def write_topology_info(self, step):
        """ Record the topology information to a file."""
        with open('logs/topology.txt', 'a') as f:
            f.write('Step: {}\n'.format(step))
            f.write('Number of tails: {}\n'.format(len(self.tails)))
            f.write('Number of loops: {}\n'.format(len(self.loops)))
            f.write('Number of bridges: {}\n'.format(len(self.bridges)))
            f.write('tail_length loop_length bridge_length\n')
            for tail, loop, bridge in zip_longest(self.tails, self.loops, self.bridges):
                f.write('{} {} {}\n'.format(len(tail)   if tail   else None,
                                            len(loop)   if loop   else None,
                                            len(bridge) if bridge else None))

    def read_ids(self):
        self.amorph_ids_gather = numpy.genfromtxt('amorph_ids.txt', dtype = int)
        self.cryst_ids_gather = []
        with open('cryst_ids.txt', 'r') as f:
            for line in f:
                self.cryst_ids_gather.append(
                        numpy.array([int(i) for i in line.strip().split()]))




setup_command = """
units           real
atom_style      angle
read_data       {datafile} extra/bond/per/atom 1 nocoeff
{cryst_group_command}
group           amorph_atoms subtract all cryst_atoms

compute         all_ids all property/atom id
compute         thermo_ke amorph_atoms ke

neigh_modify    delay 0 every 1 check yes exclude group cryst_atoms cryst_atoms
fix             1 cryst_atoms setforce 0.0 0.0 0.0

special_bonds   lj/coul 0.0 0.0 1.0
thermo_style    custom step temp press etotal ke pe epair ebond eangle vol
thermo          100
"""


HB_forcefield_command = \
"""
bond_style      harmonic
pair_style      table linear 1001
angle_style     table linear 1001

bond_coeff      1 113.746321 2.573622
angle_coeff     1 {angle_table} EEE
pair_coeff      1 1 {pair_table} EE

"""

BA_forcefield_command = \
"""
pair_style      table linear 1600
pair_coeff      1 1 {pair_table} EE

bond_style      zero
bond_coeff      *
angle_style     cgangle bicubic 63001
angle_coeff     1 {angle_table} EEE
"""

initial_run_command = \
"""
velocity        amorph_atoms create 460 1111 dist gaussian
timestep        {md_dt}
fix             2 amorph_atoms nvt temp 460 460 500
run             1000
unfix           2
"""


re_setup_commands = \
"""
compute         all_ids all property/atom id
compute         thermo_ke amorph_atoms ke
neigh_modify    delay 0 every 1 check yes exclude group cryst_atoms cryst_atoms
fix             1 cryst_atoms setforce 0.0 0.0 0.0

thermo_style    custom step temp press etotal ke pe epair ebond eangle vol
thermo          100
"""


run_commands_step1 = \
"""
# Equilibration step - NVT
write_restart   trial.restart
reset_timestep  0
# log             logs/hmc_step{step}-{substep}.log
velocity        amorph_atoms create 460 {velocity_seed} dist gaussian
run             0
"""

# This is not used anymore, step1 only saves the state and initialize velocity
"""
# fix             2 amorph_atoms nvt temp 460 460 100
# # fix             4 amorph_atoms ave/time 1 5 10 c_thermo_ke c_thermo_pe file logs/te_step{step}-{substep}.ave
# timestep        {md_dt}
# run             {nvt_run_steps}
# unfix           2
# 
# # Equilibration step - NVE
# fix             2 amorph_atoms {nve_setting}
# {temp_setting}
# timestep        1
# run             {nve_run_steps}
# unfix           2
# {temp_unfix}
"""

attack_commands = \
"""
group bond_todelete  id {bs} {bt}
delete_bonds    bond_todelete bond 1 remove special

group angle_todelete id {angle_todelete_ids}
delete_bonds    angle_todelete angle 1 remove special

group           bond_todelete clear
group           angle_todelete clear

create_bonds    single/bond 1 {ba} {bt}
create_bonds    single/angle 1 {ba_neigh1} {ba} {ba_neigh2}
neigh_modify    delay 0 every 1 check yes exclude group cryst_atoms cryst_atoms
"""

bt_angle_cmd = \
"""
create_bonds single/angle 1 {bt_neigh1} {bt} {bt_neigh2}
"""

run_commands_step2 = """
# Equilibration step - NVE

timestep        1
fix             2 amorph_atoms {nve_setting}
{temp_setting}
run             {nve_run_steps}
unfix 2
{temp_unfix}

# Equilibration step - NVT
# compute         msd amorph_atoms msd com yes
fix             2 amorph_atoms nvt temp 460 460 500
# fix             5 amorph_atoms ave/time 1 1 10 c_msd[*] file logs/g1_{step}-{substep}.ave
timestep        {md_dt}
run             {nvt_run_steps}
unfix           2
# unfix           5
# unfix           4
# uncompute       msd
"""
