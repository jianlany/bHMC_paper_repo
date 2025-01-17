#!/usr/bin/env python3
import sys
from lammps import lammps, MPIAbortException
import random
import logging
import glob
import time
import argparse
from mpi4py import MPI
comm = MPI.COMM_WORLD
me = comm.Get_rank()

from scg import generate_crystalline_system
from hmc.hmc import Semicrystalline_hmc
from hmc.hmc_config import HMC_config, setup_directories, remove_directories, get_most_recent_data



def main():
    if me == 0:
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '-in', dest = 'input')
        parser.add_argument('--clean', action = 'store_true')
        args = parser.parse_args()
        options = HMC_config(args.input)
        if args.clean:
            remove_directories()
        options.writer('options.txt')
        setup_directories()
    else:
        args = options = None

    options, args = comm.bcast((options, args), root = 0)
    set_logger(options)
    sc_hmc_run = Semicrystalline_hmc(options)

    if me == 0:
        start = time.perf_counter()
        states = glob.glob('states/hmc_step*.lammps')
        if args.clean or len(states) <= 1:
            datafile = 'initial_crystal.lammps'
            step = 0
        else:
            datafile, step = get_most_recent_data(states)
            logging.info("Restarting at step {}, reading datafile {}.".format(step, datafile))
    else:
        datafile = step = None

    datafile, step = comm.bcast((datafile, step), root = 0)
    if step >= options.max_sample_num_steps:
        if me == 0:
            logging.info("HMC run already finished, " \
                    "recent step {} is greater or equal to the max step ({}) "\
                    "of sampling stage."\
                    .format(step, options.max_sample_num_steps))
        sys.exit(0)
    if datafile == 'initial_crystal.lammps':
        generate_crystalline_system(options, me)
    comm.Barrier()

    sc_hmc_run.setup(datafile)
    if me == 0:
        logging.info("Setup takes {:.2f} seconds.".format(time.perf_counter() - start))


    substep = 0
    while step < options.max_melt_num_steps:
        accepted = HMC_move(sc_hmc_run, step, substep, options.Tsa)
        if accepted:
            step += 1
            substep = 0
        else:
            substep += 1
    if me == 0:
        logging.info("Melting stage finished.")

    while step < options.max_anneal_num_steps:
        progress =  (step - options.max_melt_num_steps)/ \
            (options.max_anneal_num_steps - options.max_melt_num_steps)
        Tsa = options.Tsa - progress * (options.Tsa - options.nvt_temp)
        if me == 0:
            logging.info("Step: {}, Tsa: {:.2f} K".format(step, Tsa))
        accepted = HMC_move(sc_hmc_run, step, substep, Tsa)
        if accepted:
            step += 1
            substep = 0
        else:
            substep += 1
    if me == 0:
        logging.info("Annealing stage finished.")

    while step <= options.max_sample_num_steps:
        accepted = HMC_move(sc_hmc_run, step, substep, options.nvt_temp)
        if accepted:
            step += 1
            substep = 0
        else:
            substep += 1

    if me == 0:
        logging.info("Sampling stage finished.")
        logging.info("HMC run finished!!! Hooray!!!!")


def HMC_move(sc_hmc_run, step, substep, Tsa):
    if me == 0:
        logging.info('***********************')
        logging.info('Step {}, substep {}.'.format(step, substep))
        logging.info('***********************')
        start = time.perf_counter()
    sc_hmc_run.hmc_step1(step, substep)
    if me == 0:
        logging.info("Step1 takes {:.2f} seconds.".format(time.perf_counter() - start))
        start = time.perf_counter()
    sc_hmc_run.attack_bonds(step, substep)
    if me == 0:
        logging.info("Bond attack takes {:.2f} seconds.".format(time.perf_counter() - start))
        start = time.perf_counter()
    try:
        sc_hmc_run.hmc_step2(step, substep)
    except Exception as e:
        logging.error("Exception caught on rank {}!!!!!!".format(me))
        comm.Abort(7)
    if me == 0:
        logging.info("Step2 takes {:.2f} seconds.".format(time.perf_counter() - start))
        start = time.perf_counter()
    accepted = sc_hmc_run.evaluate_acceptance(step, substep, Tsa)
    if me == 0:
        logging.info("Evaluate acceptance/re-setup takes {:.2f} seconds.".format(time.perf_counter() - start))
        start = time.perf_counter()
    return accepted


def set_logger(options):
    logging.basicConfig(filename = 'logs/hmc.log', 
                        filemode = 'a',
                        format = '%(asctime)s-%(levelname)s-%(message)s',

                        datefmt = '%Y-%m-%d %H:%M',
                        level = logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)



if __name__ == "__main__":
    main()
