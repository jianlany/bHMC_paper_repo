#!/usr/bin/env python3
import sys
import numpy
from read_topology_file import read_topo_file
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from subprocess import PIPE, Popen
p = Popen(['latex', '--version'], stdout=PIPE, stderr=PIPE)
stdout, stderr = p.communicate()
if p.returncode: pass
else:
    plt.style.use('paper')

def plot_final_distribution(topofile):
    tail, loop, bridge = read_topo_file(topofile, last = True)
    fig, [ax_tail, ax_loop, ax_bridge] = plt.subplots(3, 1, figsize = (3.5, 4))
    for seg, ax, xtag, tag in zip([tail, loop, bridge],
                                  [ax_tail, ax_loop, ax_bridge],
                                  ['$n_t$','$n_l$','$n_b$'],
                                  ['tail', 'loop', 'bridge']):
        lengths = list(seg.values())[0]
        if tag == 'bridge':
            bins = numpy.logspace(0, 3, 10)
        else:
            bins = numpy.logspace(0, 2, 10)
        hist, bin_edges = numpy.histogram(lengths, bins = bins, density = True)
        x = numpy.array([0.5*(e1+e2) for e1, e2 in zip(bin_edges[:-1], bin_edges[1:])])
        ax.loglog(x, hist, 'o')
        ax.set_ylabel('$P(n)$')
        ax.set_xlabel(xtag)
    fig.tight_layout()
    fig.savefig('final_seg_distribution.png', dpi = 600)



if __name__ == "__main__":
    plot_final_distribution(sys.argv[1])
