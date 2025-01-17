#!/usr/bin/env python3
import sys
import numpy
from log_reader import log_reader


def main(logfile, simple_log, **log_reader_kwargs):
    hmc_steps = log_reader(logfile, ignore_state_candidates = True, **log_reader_kwargs)
    print('Log reading finished.')
    with open(simple_log, 'w') as f:
        f.write('step substep initial_energy (kcal/mol) final_energy (kcal/mol) num_tails num_loops'\
                ' num_bridges mean_tail_length (number of beads) mean_loop_length mean_bridge_length\n')
        for hs in hmc_steps:
            try:
                hs.mean_tail_length
            except:
                print("Step {} substpe {} does not have tail_length".format(hs.step,hs.substep))
                sys.exit()
            f.write('{:7d} {:4d} {:10.2f} {:10.2f} {:5d} {:5d} {:5d} {:10.5f} {:10.5f} {:10.5f}\n'.\
                    format(hs.step, hs.substep, hs.initial_te, hs.final_te, 
                           hs.num_tails, hs.num_loops, hs.num_bridges, 
                           hs.mean_tail_length, 
                           hs.mean_loop_length, 
                           hs.mean_bridge_length))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
