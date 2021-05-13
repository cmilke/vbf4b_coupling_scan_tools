import sys
import argparse
import numpy
import itertools
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#import pdb

from combination_utils import is_valid_combination
from fileio_utils import read_coupling_file, get_events, retrieve_reco_weights
from negative_weight_map import get_negative_weight_grid, draw_error_map


def main():
    var_edges = numpy.linspace(200, 1200, 31)
    kv_val = 1.0
    num_kappa_bins = 100
    k2v_val_range = numpy.linspace(-2,4,num_kappa_bins+1)
    kl_val_range = numpy.linspace(-14,16,num_kappa_bins+1)
    grid_pixel_area = (k2v_val_range[1] - k2v_val_range[0]) * (kl_val_range[1] - kl_val_range[0])

    all_variations = #TODO

    valid_bases = []
    total = 0
    for couplings in itertools.combinations(all_variations,6):
        # Unwrap each combination
        if (1.0,1.0,1.0) not in couplings: continue
        if not is_valid_combination(couplings): continue

        weights, errors = numpy.array( list(zip(*histograms)) )
        nWeight_integral = get_Nweight_sum(couplings, weights, kv_val, k2v_val_range, kl_val_range)
        valid_bases.append( (nWeight_integral, couplings, weights) )
        total += 1
        if total % 10 == 0: print(total)
    print('Integrals computed, sorting and printing...')
    valid_bases.sort()
    for rank, (integral, couplings, weight) in enumerate(valid_bases): print(rank, int(integral), couplings)

    ranks_to_draw = 0, int(len(valid_bases)/2), 27#, len(valid_bases)-1
    draw_rankings(ranks_to_draw, valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, '')
    draw_rankings([0,1,2], valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, 'top')





if __name__ == '__main__': main()
