import numpy
import logging

class State:
    def __init__(self, ba, bt, bs):
        self.ba = ba
        self.bt = bt
        self.bs = bs

        self.tail_mean = 0
        self.loop_mean = 0
        self.bridge_mean = 0



    def calc_probability(self, hmc_run, options):
        mean_tail_length, mean_loop_length, mean_bridge_length = \
                self.calc_segment_length(hmc_run)
        logging.info('\tResulting mean length of tails:   {:.2f}, target: {:.2f}'.format(mean_tail_length,   options.tail_mean_target))
        logging.info('\tResulting mean length of loops:   {:.2f}, target: {:.2f}'.format(mean_loop_length,   options.loop_mean_target))
        logging.info('\tResulting mean length of bridges: {:.2f}, target: {:.2f}'.format(mean_bridge_length, options.bridge_mean_target))
        gaussian = lambda l, mean, std: \
            numpy.exp(-(l-mean)*(l-mean)/(2*std*std))
        p = 1
        if options.tail_mean_target:
            p*=gaussian(mean_tail_length, options.tail_mean_target, options.tail_std)
        if options.loop_mean_target:
            p*=gaussian(mean_loop_length, options.loop_mean_target, options.loop_std)
        if options.bridge_mean_target:
            p*=gaussian(mean_bridge_length, options.bridge_mean_target, options.bridge_std)
        return p, mean_tail_length, mean_loop_length, mean_bridge_length


    def calc_segment_length(self, hmc_run):
        """ This function calculates the total length of tails, loops, and bridges
            based on the topology changes."""
        num_tails = len(hmc_run.tails)
        num_loops = len(hmc_run.loops)
        num_bridges = len(hmc_run.bridges)
        bs_anchor, bs_distance = self.find_seg_anchor_length(self.bs, self.bt, hmc_run)
        assert (bs_anchor <= 0) & (bs_distance >= 0)
        if self.bt not in hmc_run.tail_atoms:
            bt_anchor, bt_distance = self.find_seg_anchor_length(self.bt, self.bs, hmc_run)
            assert (bt_anchor <= 0) & (bt_distance >= 0)
        ba_anchor, ba_distance = hmc_run.cryst_anchor[self.ba-1], \
                                 hmc_run.distance_from_crystalline[self.ba-1]
        assert (ba_anchor <= 0) & (ba_distance >= 0)

        assert ba_distance == hmc_run.segment_lengths_atom[self.ba-1]
        if self.bt not in hmc_run.tail_atoms:
            assert bt_distance + bs_distance == hmc_run.segment_lengths_atom[self.bt-1]

        tail_length   = len(hmc_run.tail_atoms)
        loop_length   = len(hmc_run.loop_atoms)
        bridge_length = len(hmc_run.bridge_atoms)
        # tail + tail does not change number of tail atoms or number of tails, 
        # hence does not change average length
        tail_length += bs_distance - ba_distance
        if self.bt in hmc_run.tail_atoms:
            pass
        # tail + loop
        elif self.bt in hmc_run.loop_atoms:
            assert hmc_run.cryst_anchor[self.bt-1] <= 0
            # attacker and target are from the same crystal
            # tail + loop =  tail + loop
            if hmc_run.cryst_anchor[self.bt-1] == \
               hmc_run.cryst_anchor[self.ba-1]:
                loop_length += ba_distance - bs_distance
                """
                if (distance_from_crystalline[self.bs-1] <
                    distance_from_crystalline[self.bt-1]):
                    loop_length +=  attacker_length \
                                  - distance_from_crystalline[self.bs-1]
                else:
                    loop_length += attacker_length\
                                - (segment_lengths_atom[self.bs-1] - distance_from_crystalline[self.bt-1])
                """
            # attacker and target are from different crystal
            # tail + loop =  tail + bridge 
            else:
                loop_length -= hmc_run.segment_lengths_atom[self.bt-1]
                bridge_length += bt_distance + ba_distance
                num_loops -= 1
                num_bridges += 1
                """
                if (distance_from_crystalline[self.bs-1] <
                    distance_from_crystalline[self.bt-1]):
                    bridge_length += attacker_length \
                                    +(segment_lengths_atom[self.bs-1] - distance_from_crystalline[self.bs-1])
                else:
                    bridge_length += distance_from_crystalline[self.bt-1] \
                                   + attacker_length
                """
        # tail + bridge
        elif self.bt in hmc_run.bridge_atoms:
            # tail + bridge = tail + bridge
            if bs_anchor == ba_anchor:
                bridge_length += ba_distance - bs_distance
            # tail + bridge = tail + loop
            else:
                loop_length += ba_distance + bt_distance
                bridge_length -= hmc_run.segment_lengths_atom[self.bt-1]
                num_loops += 1
                num_bridges -= 1
        else: 
            logging.info("The target bead is not in tails, loops, or bridges, something is wrong.")
            comm.Abort(2)

        mean_tail_length   = tail_length / num_tails if num_tails else 0
        mean_loop_length   = loop_length / num_loops if num_loops else 0
        mean_bridge_length = bridge_length / num_bridges if num_bridges else 0
        return mean_tail_length, mean_loop_length, mean_bridge_length


    def find_seg_anchor_length(self, i, j, hmc_run):
        """ This function finds the number of beads from i (inclusive) to 
            the crystalline phase (excluding crystalline atoms), 
            not in the j direction."""
        i_distance = hmc_run.distance_from_crystalline[i-1]
        j_distance = hmc_run.distance_from_crystalline[j-1]
        seg_length = hmc_run.segment_lengths_atom[i-1]
        assert hmc_run.segment_lengths_atom[i-1] == hmc_run.segment_lengths_atom[j-1], \
                print(i, j)

        if i in hmc_run.tail_atoms:
            assert i_distance < j_distance, "i {}, j {}, i distance: {}, j distance: {}. {}".format(i, j, i_distance, j_distance, j in hmc_run.conn[i-1] )
            return hmc_run.cryst_anchor[i-1], hmc_run.distance_from_crystalline[i-1]
        elif i in hmc_run.loop_atoms:
            if i_distance < j_distance:
                return hmc_run.cryst_anchor[i-1], i_distance
            else:
                return hmc_run.cryst_anchor[i-1], seg_length - j_distance
        else:
            stack = [i]
            seen = set()
            length = 0
            while stack:
                ii = stack.pop()
                length += 1
                seen.add(ii)
                for jj in hmc_run.conn[ii-1]:
                    if jj in seen: continue
                    if jj == j: continue
                    if jj not in hmc_run.amorph_ids_gather:
                        return hmc_run.distance_from_crystalline[jj-1], length
                    seen.add(jj)
                    stack.append(jj)
