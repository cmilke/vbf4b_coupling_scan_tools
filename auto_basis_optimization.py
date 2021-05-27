import sys
import argparse
import numpy
import itertools
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#import pdb

import combination_utils
import fileio_utils
from negative_weight_map import get_Nweight_sum, draw_error_map
import validate_linear_combinations


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
                vmax = max_negative_weight, name_suffix=f'_{name_infix}rank{int(rank)}', 
                title_suffix=f'Rank {rank+1}/{len(valid_bases)}, Integral={int(basis[0])}')
    comp_couplings = [ valid_bases[ranks_to_draw[0]][1], valid_bases[ranks_to_draw[1]][1] ]
    num_kappa_bins = 3
    data_files = fileio_utils.read_coupling_file(fileio_utils.coupling_file)
    validate_linear_combinations.compare_bases_reco_method(comp_couplings, list(data_files.keys()),
         name_suffix=f'_auto_{name_infix}_3D_{ranks_to_draw[0]}-{ranks_to_draw[1]}', labels=(f'Rank {ranks_to_draw[0]}', f'Rank {ranks_to_draw[1]}') )
    k2v_vals = [-1.5, 0.5, 2, 3.5]
    kl_vals = [-9, -3, 5, 14]
    preview_couplings = []
    for k2v in k2v_vals:
        for kl in kl_vals:
            preview_couplings.append( (k2v, kl, 1) )
    validate_linear_combinations.compare_bases_reco_method(comp_couplings, preview_couplings,
         name_suffix='_preview_auto_'+name_infix+'_3D_'f'{ranks_to_draw[0]}-{ranks_to_draw[1]}', labels=(f'Rank {ranks_to_draw[0]}', f'Rank {ranks_to_draw[1]}'), is_verification=False)


def optimize_reco():
    var_edges = numpy.linspace(200, 1200, 31)
    kv_val = 1.0
    num_kappa_bins = 100
    k2v_val_range = numpy.linspace(-2,4,num_kappa_bins+1)
    kl_val_range = numpy.linspace(-14,16,num_kappa_bins+1)
    grid_pixel_area = (k2v_val_range[1] - k2v_val_range[0]) * (kl_val_range[1] - kl_val_range[0])

    data_files = fileio_utils.read_coupling_file(fileio_utils.coupling_file)
    all_events = fileio_utils.get_events(data_files.keys(), data_files)
    all_histograms = [ fileio_utils.retrieve_reco_weights(var_edges,events) for events in all_events ]
    # Wrap all variations up together with their histograms so I can find combinations
    all_variations = list(zip(data_files.keys(), all_histograms))
    print('Histograms loaded, proceeding to integrate Nweight grids...')

    valid_bases = []
    total = 0
    for basis_set in itertools.combinations(all_variations,6):
        # Unwrap each combination
        couplings, histograms = list(zip(*basis_set))
        if (1.0,1.0,1.0) not in couplings: continue
        if not combination_utils.is_valid_combination(couplings): continue

        weights, errors = numpy.array( list(zip(*histograms)) )
        nWeight_integral = get_Nweight_sum(couplings, weights, kv_val, k2v_val_range, kl_val_range)
        valid_bases.append( (nWeight_integral, couplings, weights) )
        total += 1
        if total % 10 == 0: print(total)
    print('Integrals computed, sorting and printing...')
    valid_bases.sort()
    for rank, (integral, couplings, weight) in enumerate(valid_bases): print(rank, int(integral), couplings)

    ranks_to_draw = 0, int(len(valid_bases)/2), 27#, len(valid_bases)-1
    #draw_rankings(ranks_to_draw, valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, '')
    draw_rankings([0,1], valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, 'top')
    #draw_rankings([0,1], valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, 'alttop')
    #draw_rankings([0,2], valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, 'alttop')
    #draw_rankings([0,127], valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, 'badcomp')
    #draw_rankings([0,220], valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, 'baddercomp')
    combination_utils.get_amplitude_function(valid_bases[0][1], base_equations=combination_utils.full_scan_terms, name='optimal_3DR0', output='tex')
    combination_utils.get_amplitude_function(valid_bases[1][1], base_equations=combination_utils.full_scan_terms, name='optimal_3DR1', output='tex')



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
