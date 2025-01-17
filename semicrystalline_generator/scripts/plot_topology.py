#!/usr/bin/env python3
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from read_topology_file import read_topo_file


from subprocess import PIPE, Popen
p = Popen(['latex', '--version'], stdout=PIPE, stderr=PIPE)
stdout, stderr = p.communicate()
if p.returncode: pass
else:
    plt.style.use('paper')


def main(topofile):
    tail_lengths, loop_lengths, bridge_lengths = read_topo_file(topofile)
    fig, [[ax1_top, ax2_top, ax3_top],\
          [ax1_bot, ax2_bot, ax3_bot]]= plt.subplots(2,3, figsize = (7, 3), sharex = 'col')
    for seg, ax in zip([tail_lengths, loop_lengths, bridge_lengths], 
                         [[ax1_top, ax1_bot], [ax2_top, ax2_bot], [ax3_top, ax3_bot]]):
        for step, lengths in seg.items():
            ax[0].plot(step, lengths.mean() if len(lengths) else 0, 'o', color = 'C0')
            ax[1].plot(step, len(lengths), 'o', color = 'C0')
        ax[1].set_xlabel('steps')
    ax1_top.set_ylabel(r'$\bar{l_t}$ (number of beads)')
    ax2_top.set_ylabel(r'$\bar{l_l}$ (number of beads)')
    ax3_top.set_ylabel(r'$\bar{l_b}$ (number of beads)')

    ax1_bot.set_ylabel(r'$n_t$')
    ax2_bot.set_ylabel(r'$n_l$')
    ax3_bot.set_ylabel(r'$n_b$')

    fig.tight_layout()
    fig.savefig('segment_mean_length_evo.png', dpi = 600)


if __name__ == "__main__":
    main(sys.argv[1])


