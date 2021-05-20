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
from negative_weight_map import get_Nweight_sum, draw_error_map


def draw_rankings(ranks_to_draw, valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, name_infix):
    print('Drawing basis ranks ' + str(ranks_to_draw))
    max_negative_weight = 0
    for rank in ranks_to_draw:
        basis = valid_bases[rank]
        nWeight_grid = get_Nweight_sum(basis[1], basis[2], kv_val, k2v_val_range, kl_val_range, grid=True)
        max_for_rank = numpy.max(nWeight_grid)
        if max_for_rank > max_negative_weight:
            max_negative_weight = max_for_rank

    for rank in ranks_to_draw:
        print('Drawing rank ' + str(rank) + '...')
        basis = valid_bases[rank]
        nWeight_grid = get_Nweight_sum(basis[1], basis[2], kv_val, k2v_val_range, kl_val_range, grid=True)
        draw_error_map(basis[1], var_edges, kv_val, k2v_val_range, kl_val_range, nWeight_grid, 
                vmax = max_negative_weight, name_suffix=f'_{name_infix}rank{rank:03d}', 
                title_suffix=f'Rank {rank+1}/{len(valid_bases)}, Integral={int(basis[0])}')


def optimize_reco():
    var_edges = numpy.linspace(200, 1200, 31)
    kv_val = 1.0
    num_kappa_bins = 100
    k2v_val_range = numpy.linspace(-2,4,num_kappa_bins+1)
    kl_val_range = numpy.linspace(-14,16,num_kappa_bins+1)
    grid_pixel_area = (k2v_val_range[1] - k2v_val_range[0]) * (kl_val_range[1] - kl_val_range[0])

    data_files = read_coupling_file('basis_files/nnt_coupling_file_2021May.dat')
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



def main():
    # Sort out command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( "--mode", required = False, default = 'reco', type=str,
        help = "Options are: 'truth' or 'reco'",) 

    args = parser.parse_args()

    #pdb.set_trace()
    #numpy.set_printoptions(precision=None, linewidth=400, threshold=10000, sign=' ', formatter={'float':lambda f: f'{int(f):2d}'}, floatmode='fixed')
    #numpy.set_printoptions(precision=1, linewidth=400, threshold=10000, sign=' ', floatmode='fixed')
    if args.mode == 'reco': optimize_reco()
    else:
        print('Mode - '+str(args.mode)+' - is not valid.')
        print('Please choose from:\ntruth\nrwgt_truth\nreweight\nreco\n')
        print('Aborting')
        exit(1)


if __name__ == '__main__': main()
