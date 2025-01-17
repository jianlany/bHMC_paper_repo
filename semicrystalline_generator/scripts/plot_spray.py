#!/usr/bin/env python3
import sys
import numpy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes

import log_reader


from subprocess import PIPE, Popen
p = Popen(['latex', '--version'], stdout=PIPE, stderr=PIPE)
stdout, stderr = p.communicate()
if p.returncode: pass
else:
    plt.style.use('paper')


def main(logfile):
    hmc_steps = log_reader.log_reader(logfile, end = 1000)
    for i, hmc_step in enumerate(hmc_steps):
        print('step', hmc_step.step)
        if hmc_step.step%100 != 0: continue
        prev_step = hmc_steps[i-1]
        fig, ax = plt.subplots()
        ax.set_xlabel('$l_l$  (number of beads)')
        ax.set_ylabel('$l_b$  (number of beads)')
        ax.plot(hmc_step.mean_loop_length,
                hmc_step.mean_bridge_length, 'x', ms = 7)
        loop_means = []
        bridge_means = []
        ps = []
        for i,state in enumerate(hmc_step.state_candidates):
            if (not state.mean_lengths) or \
               (not state.target_means): 
                   print("State {} is invalid.".format(state.state_num))
                   continue
            loop_means.append(state.mean_lengths['loops'])
            bridge_means.append(state.mean_lengths['bridges'])
            ps.append(state.p)
            if i == 0:
                ax.plot(state.target_means['loops'],
                        state.target_means['bridges'], '*')

        ax.plot(loop_means, bridge_means, 'o', ms = 3, fillstyle = 'none',color = 'C0')
        axins = inset_axes(ax, width = '40%', 
                               height = '40%', 
                               bbox_to_anchor = [0.1, 0.1, 1.0, 1.0],
                               bbox_transform=ax.transAxes,
                               loc = 'lower left')
        cax = inset_axes(axins,
                         width="5%",
                         height="100%",
                         loc='lower left',
                         bbox_to_anchor=(1.05, 0., 1, 1),
                         bbox_transform=axins.transAxes,
                         borderpad=0,
                         )

        ps = numpy.array(ps)
        ps = ps/(ps[ps!=0].min()) if (ps!=0).any() else (0*ps)
        im = axins.scatter(loop_means, bridge_means, c = ps, cmap = 'inferno')
        cbar = plt.colorbar(mappable = im, cax = cax)
        cbar.ax.set_ylabel('$p_\mathrm{max}/p_\mathrm{min}$', fontsize = 6)
        axins.plot(hmc_step.mean_loop_length,
                   hmc_step.mean_bridge_length, 'x')
        selected_state = hmc_step.state_candidates[hmc_step.selected_state_num-1]
        axins.plot(selected_state.mean_lengths['loops'], 
                   selected_state.mean_lengths['bridges'], 
                   'o', color = 'C0')
        axins.set_title("{} loops, {} bridges ".format(hmc_step.num_loops, hmc_step.num_bridges), fontsize = 6)



        # fig.tight_layout()
        fig.savefig('spray_plot_step{}.png'.format(hmc_step.step), dpi = 600)
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1])
