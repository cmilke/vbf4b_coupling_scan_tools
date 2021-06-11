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
import validate_linear_combinations
import weight_contribution_map


def draw_rankings(ranks_to_draw, valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, name_infix):
    print('Drawing basis ranks ' + str(ranks_to_draw))
    for rank in ranks_to_draw:
        print('Drawing rank ' + str(rank) + '...')
        basis = valid_bases[rank]
        #solidarity_grid = weight_contribution_map.get_theoretical_solidarity_map(basis[1], kv_val, k2v_val_range, kl_val_range, grid=True)
        #solidarity_integral = weight_contribution_map.get_theoretical_solidarity_map(basis[1], kv_val, k2v_val_range, kl_val_range, grid=False)
        effective_stats_grid = weight_contribution_map.get_theory_effective_stats_map(basis[1], kv_val, k2v_val_range, kl_val_range, grid=True)
        effective_stats_integral = weight_contribution_map.get_theory_effective_stats_map(basis[1], kv_val, k2v_val_range, kl_val_range, grid=False)
        weight_contribution_map.draw_contribution_map(basis[1], kv_val, k2v_val_range, kl_val_range, effective_stats_grid, vmin=0, vmax=None,
                name_suffix=f'solidarity_{name_infix}rank{int(rank)}', 
                #title_suffix=f'Rank {rank+1}/{len(valid_bases)}, Integral={int(basis[0])}')
                title_suffix=f'Rank {rank+1}/{len(valid_bases)}, Integral={int(effective_stats_integral)}')


def optimize_reco():
    var_edges = numpy.linspace(200, 1200, 31)
    kv_val = 1.0
    num_kappa_bins = 100
    k2v_val_range = numpy.linspace(-2,4,num_kappa_bins+1)
    kl_val_range = numpy.linspace(-14,16,num_kappa_bins+1)
    grid_pixel_area = (k2v_val_range[1] - k2v_val_range[0]) * (kl_val_range[1] - kl_val_range[0])

    all_variations = [
        (1    ,   1  ,  1     ),
        (0    ,   1  ,  1     ),
        (0.5  ,   1  ,  1     ),
        (1.5  ,   1  ,  1     ),
        (2    ,   1  ,  1     ),
        (3    ,   1  ,  1     ),
        (1    ,   0  ,  1     ),
        (1    ,   2  ,  1     ),
        (1    ,   10 ,  1     ),
        (1    ,   1  ,  0.5   ),
        (1    ,   1  ,  1.5   ),
        (0    ,   0  ,  1     ),
        #(2.5  ,  10  ,  1     )
        #(-0.5 ,   8  ,  0.5   )
        (3    ,  -9  ,  1     )
        #(2    ,   7  ,  1     )
    ]

    valid_bases = []
    total = 0
    for couplings in itertools.combinations(all_variations,6):
        # Unwrap each combination
        if (1.0,1.0,1.0) not in couplings: continue
        if not combination_utils.is_valid_combination(couplings, base_equations=combination_utils.full_scan_terms): continue
        #solidarity_integral = weight_contribution_map.get_theoretical_solidarity_map(couplings, 1, k2v_val_range, kl_val_range,
                #mask=lambda k2v, kl: ((k2v-1)/1)**2 + ((kl-1)/10)**2 < 1 )
        #solidarity_integral = weight_contribution_map.get_theoretical_solidarity_map(couplings, 1, k2v_val_range, kl_val_range)
        #solidarity_integral = (weight_contribution_map.get_theory_effective_stats_map(couplings, 1, k2v_val_range, kl_val_range)).sum()
        #valid_bases.append( (solidarity_integral, couplings) )
        effective_stat_integral = weight_contribution_map.get_theory_effective_stats_map(couplings, 1, k2v_val_range, kl_val_range)
                #mask=lambda k2v, kl: ((k2v-1)/1)**2 + ((kl-1)/10)**2 < 1 )
        valid_bases.append( (effective_stat_integral, couplings) )
        total += 1
        if total % 10 == 0: print(total)
    print('Integrals computed, sorting and printing...')
    valid_bases.sort(reverse=True)
    for rank, (integral, couplings) in enumerate(valid_bases): print(rank, int(integral), couplings)

    #draw_rankings([0,1,2,3], valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, 'masktop')
    #draw_rankings([0,1,2,3], valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, 'neomasktop')
    draw_rankings([0], valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, 'neoeffectiveB')
    #draw_rankings([0,1,2,3], valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, 'neotop')



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
