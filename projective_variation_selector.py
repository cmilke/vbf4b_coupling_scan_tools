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


def draw_rankings(ranks_to_draw, valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, name_infix):
    print('Drawing basis ranks ' + str(ranks_to_draw))
    max_negative_weight = max([ numpy.max(valid_bases[rank][2]) for rank in ranks_to_draw ])
    for rank in ranks_to_draw:
        print('Drawing rank ' + str(rank) + '...')
        basis = valid_bases[rank]
        draw_error_map(basis[1], var_edges, kv_val, k2v_val_range, kl_val_range, basis[2], 
                vmax = max_negative_weight, name_suffix=f'_{name_infix}rank{rank:03d}', 
                title_suffix=f'Rank {rank+1}/{len(valid_bases)}, Integral={int(basis[0])}')


def main():
    var_edges = numpy.linspace(200, 1200, 31)
    kv_val = 1.0
    k2v_val_range = numpy.linspace(-2,4,101)
    kl_val_range = numpy.linspace(-14,16,101)

    data_files = read_coupling_file('basis_files/nnt_coupling_file.dat')
    all_events = get_events(data_files.keys(), data_files)
    all_histograms = [ retrieve_reco_weights(var_edges,events) for events in all_events ]
    # Wrap all variations up together with their histograms so I can find combinations
    all_variations = list(zip(data_files.keys(), all_histograms))
    print('Histograms loaded, proceeding to integrate Nweight grids...')

    valid_bases = []
    total = 0
    for basis_set in itertools.combinations(all_variations,6):
        # Unwrap each combination
        couplings, histograms = list(zip(*basis_set))
        if not is_valid_combination(couplings): continue
        if (1.0,1.0,1.0) not in couplings: continue
        weights, errors = numpy.array( list(zip(*histograms)) )
        grid_pixel_area = (k2v_val_range[1] - k2v_val_range[0]) * (kl_val_range[1] - kl_val_range[0])
        negative_weight_grid = get_negative_weight_grid(couplings, weights, errors, kv_val, k2v_val_range, kl_val_range)
        grid_integral = numpy.sum( negative_weight_grid * grid_pixel_area )
        valid_bases.append( (grid_integral, couplings, negative_weight_grid) )
        total += 1
        if total % 10 == 0: print(total)
    print('Integrals computed, sorting and printing...')
    valid_bases.sort()
    for rank, (integral, couplings, grid) in enumerate(valid_bases): print(rank, int(integral), couplings)


    ranks_to_draw = 0, int(len(valid_bases)/2), 27#, len(valid_bases)-1
    draw_rankings(ranks_to_draw, valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, '')
    draw_rankings([0,1,2], valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, 'top')





if __name__ == '__main__': main()
