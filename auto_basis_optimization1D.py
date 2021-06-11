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
import negative_weight_map
import validate_linear_combinations


def draw_rankings(ranks_to_draw, valid_bases, which_variations, var_edges, k2v_vals, kl_vals, kv_val, which_coupling, base_equations):
    print('Drawing basis ranks ' + str(ranks_to_draw))
    for rank in ranks_to_draw:
        nWeight_integral, couplings, weights = valid_bases[rank]
        #validate_linear_combinations.validate_reco_method(couplings, which_variations,
        #        base_equations=base_equations,
        #        name_suffix='_'+which_coupling+'_Nweight-rank'+str(rank),
        #        title_suffix=': '+which_coupling+' Nweight Rank '+str(rank)+', Integral = '+f'{int(nWeight_integral)}')
        validate_linear_combinations.draw_1D_mhh_heatmap(couplings, weights, var_edges, k2v_vals, kl_vals, kv_val,
                base_equations=base_equations,
                which_coupling=which_coupling,
                filename='auto-'+which_coupling+'_rank'+str(rank),
                title_suffix=' Nweight Rank '+str(rank+1)+'/'+str(len(ranks_to_draw))+', Integral = '+f'{int(nWeight_integral)}')

    comp_couplings = [ valid_bases[0][1], valid_bases[1][1] ]
    num_kappa_bins = 10
    if which_coupling == 'k2v':
        coupling_index = 0
        preview_vals = numpy.linspace(-2,4,num_kappa_bins+1)
        preview_couplings = [ (k2v, 1, 1) for k2v in preview_vals ]
    if which_coupling == 'kl':
        coupling_index = 1
        preview_vals = numpy.linspace(-14,16,num_kappa_bins+1)
        preview_couplings = [ (1, kl, 1) for kl in preview_vals ]
    labels = [ which_coupling+'='+', '.join([ f'{c[coupling_index]:.1f}' for c in couplings ]) for couplings in comp_couplings ]
    validate_linear_combinations.compare_bases_reco_method(comp_couplings, which_variations,
        base_equations=base_equations, name_suffix='_auto'+which_coupling+'01', labels=labels)
    validate_linear_combinations.compare_bases_reco_method(comp_couplings, preview_couplings,
        base_equations=base_equations, name_suffix='_preview_auto'+which_coupling+'01', labels=labels, is_verification=False)


def optimize_reco(which_coupling):
    var_edges = numpy.linspace(200, 1200, 31)
    kv_val = 1.0
    num_kappa_bins = 100
    data_files = fileio_utils.read_coupling_file()

    if which_coupling == 'k2v':
        hold_index = 1
        base_equations = combination_utils.k2v_scan_terms
        k2v_vals = numpy.linspace(-2,4,num_kappa_bins+1)
        kl_vals = 1
    elif which_coupling == 'kl':
        hold_index = 0
        base_equations = combination_utils.kl_scan_terms
        k2v_vals = 1
        kl_vals = numpy.linspace(-14,16,num_kappa_bins+1)
    else:
        print("What are you doing??")
        exit(1)
    which_variations = [ variation for variation in data_files.keys() if variation[2] == 1 and variation[hold_index] == 1 ]

    all_events = fileio_utils.get_events(which_variations, data_files)
    all_histograms = [ fileio_utils.retrieve_reco_weights(var_edges,events) for events in all_events ]
    # Wrap all variations up together with their histograms so I can find combinations
    all_variations = list(zip(which_variations, all_histograms))
    print('Histograms loaded, proceeding to integrate Nweight grids...')

    valid_bases = []
    total = 0
    for basis_set in itertools.combinations(all_variations,len(base_equations)):
        # Unwrap each combination
        couplings, histograms = list(zip(*basis_set))
        if (1.0,1.0,1.0) not in couplings: continue
        if not combination_utils.is_valid_combination(couplings, base_equations=base_equations): continue

        weights, errors = numpy.array( list(zip(*histograms)) )
        nWeight_integral = negative_weight_map.get_Nweight_sum1D(couplings, weights, k2v_vals, kl_vals, kv_val, base_equations=base_equations, which_coupling=which_coupling)
        valid_bases.append( (nWeight_integral, couplings, weights) )
        total += 1
        if total % 10 == 0: print(total)
    print('Integrals computed, sorting and printing...')
    valid_bases.sort()
    for rank, (integral, couplings, weight) in enumerate(valid_bases): print(rank, f'{integral:.9f}', couplings)

    ranks_to_draw = 0, 1, 2
    draw_rankings(ranks_to_draw, valid_bases, which_variations, var_edges, k2v_vals, kl_vals, kv_val, which_coupling, base_equations)
    combination_utils.get_amplitude_function(valid_bases[0][1], base_equations=base_equations, name='optimalR0_'+which_coupling, output='tex')
    combination_utils.get_amplitude_function(valid_bases[1][1], base_equations=base_equations, name='optimalR1_'+which_coupling, output='tex')



def main():
    # Sort out command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( "--mode", required = False, default = 'reco', type=str,
        help = "Options are: 'truth' or 'reco'",) 

    args = parser.parse_args()

    #pdb.set_trace()
    #numpy.set_printoptions(precision=None, linewidth=400, threshold=10000, sign=' ', formatter={'float':lambda f: f'{int(f):2d}'}, floatmode='fixed')
    #numpy.set_printoptions(precision=1, linewidth=400, threshold=10000, sign=' ', floatmode='fixed')
    if args.mode == 'reco':
        optimize_reco('k2v')
        optimize_reco('kl')
    else:
        print('Mode - '+str(args.mode)+' - is not valid.')
        print('Please choose from:\ntruth\nrwgt_truth\nreweight\nreco\n')
        print('Aborting')
        exit(1)


if __name__ == '__main__': main()
