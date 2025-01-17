#!/usr/bin/env python3
import sys
import re
import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from subprocess import PIPE, Popen
p = Popen(['latex', '--version'], stdout=PIPE, stderr=PIPE)
stdout, stderr = p.communicate()
if p.returncode: pass
else:
    plt.style.use('paper')

def main(logfile):
    final_energy = {}
    with open(logfile, 'r') as f:
        for line in f:
            if 'Step' in line and 'substep' in line:
                m = re.match('.*Step (\d+), substep (\d+).*', line)
                step, substep = [int(s) for s in m.groups()]
            if 'Final' in line:
                fe = float(re.match('.*Final total energy: (\d+\.\d+) kcal\/mol', line).group(1))
                final_energy[step] = fe
    fig, ax = plt.subplots()
    ax.plot(final_energy.keys(), final_energy.values())
    ax.set_xlabel('Steps')
    ax.set_ylabel('Total energy (kcal/mol)')
    fig.tight_layout()
    fig.savefig('te_evo.png', dpi = 300)


if __name__ == "__main__":
    main(sys.argv[1])


