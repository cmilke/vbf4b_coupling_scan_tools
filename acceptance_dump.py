import sys
import argparse
import math
import numpy
import statistics
import itertools
import pickle
import matplotlib
from matplotlib import pyplot as plt

import fileio_utils, combination_utils, negative_weight_map

#import pdb

def metric_accXeff_list(theory_function, couplings, cutflows):
    lumi_total = 3.2 + 24.6 + 43.65 + 58.45
    return [ (cuts['Final'] / (theory_function(kappas)*lumi_total) ) * 10**4 for cuts,kappas in zip(cutflows,couplings) ]
    #return [ (cuts['Final'] / cuts['Initial']) * 10**5 for cuts in cutflows ]
    #return [ cuts['FinalCount'] for cuts in cutflows ]

def get_sorted_acceptances():
    var_edges = numpy.linspace(200, 1200, 31)
    kv_val = 1.0
    num_bins = 21
    k2v_val_range = numpy.linspace(-2,4,num_bins)
    kl_val_range = numpy.linspace(-14,16,num_bins)

    data_files = fileio_utils.read_coupling_file()
    all_cutflows = fileio_utils.get_combined_cutflow_values(data_files.keys(), data_files).values() # It's a really good things that python dicts are ordered...
    all_events = fileio_utils.get_events(data_files.keys(), data_files)
    all_histograms = [ fileio_utils.retrieve_reco_weights(var_edges,events) for events in all_events ]
    # Wrap all variations up together with their histograms so I can find combinations
    all_variations = list(zip(data_files.keys(), all_histograms, all_cutflows, all_events))#[:7]
    print('All variations loaded, proceeding to retrieve metrics...')
    theory_function = combination_utils.get_theory_xsec_function()

    total = 0
    basis_metrics = {}
    Nweight_acceptance_list = []
    for basis_set in itertools.combinations(all_variations,6):
        # Unwrap each combination
        couplings, histograms, cutflows, events_list = list(zip(*basis_set))
        if not combination_utils.is_valid_combination(couplings): continue
        #if (1.0,1.0,1.0) not in couplings: continue

        weights, errors = numpy.array( list(zip(*histograms)) )
        Nweight_integral = negative_weight_map.get_Nweight_sum(couplings, weights, kv_val, k2v_val_range, kl_val_range)
        accXeff = metric_accXeff_list(theory_function, couplings, cutflows)

        total += 1
        if total % 10 == 0: print(total)
        if total % 20 != 0: continue
        Nweight_acceptance_list.append( [Nweight_integral, accXeff] )
    return Nweight_acceptance_list


def main():
    use_cached = len(sys.argv) > 1
    Nweight_acceptance_list = None
    cache_file = '.cached_scores.p'
    if use_cached:
        Nweight_acceptance_list = pickle.load(open(cache_file,'rb'))
    else:
        Nweight_acceptance_list = get_sorted_acceptances()
        pickle.dump(Nweight_acceptance_list, open(cache_file,'wb'))
    print('Data retrieved, plotting...')

    Nweight_acceptance_list.sort()
    for i,a in Nweight_acceptance_list: a.sort(reverse=True)
    #for i,a in Nweight_acceptance_list: print(i, a)
    Nweight_list, acceptance_list = zip(*Nweight_acceptance_list)
    acceptance_array = numpy.array(acceptance_list)
    numpy.set_printoptions(precision=2, linewidth=400, threshold=100, sign=' ')
    
    fig, ax = plt.subplots()
    #im = ax.imshow(acceptance_array, origin='lower', cmap='viridis')
    im = ax.imshow(acceptance_array, cmap='viridis', extent=(0,6,0,len(acceptance_list)))
    #im = ax.imshow(acceptance_array, cmap='viridis', extent=(0,6,0,len(acceptance_list)), norm=matplotlib.colors.LogNorm(1,10) )

    ax.set_xticks(ticks = range(6))
    ax.set_yticks(ticks = range(len(acceptance_list)))
    ax.set_xlabel('Sample')
    ax.set_ylabel('Nweight Integral')

    x_param_labels = [ str(i) for i in range(1,7) ]
    x_param_ticks = numpy.array(range(6))
    ax.set_xticks(x_param_ticks)
    ax.set_xticks(x_param_ticks+0.5, minor=True)
    ax.set_xticklabels('') # Clear major tick labels
    ax.set_xticklabels(x_param_labels, minor=True, fontsize=10)

    y_param_labels = [ f'{int(Nweight)}' for Nweight in Nweight_list[::-1] ]
    y_param_ticks = numpy.array(range(len(acceptance_list)))
    ax.set_yticks(y_param_ticks)
    ax.set_yticks(y_param_ticks+0.5, minor=True)
    ax.set_yticklabels('') # Clear major tick labels
    ax.set_yticklabels(y_param_labels, minor=True, fontsize=4)

    #ax.grid()
    ax.set_aspect('auto','box')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.89, 0.11, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Acceptance X Efficiency (x$10^{-5}$)')

    title  = 'Performance of Basis VS Acceptance of Constituent Samples'
    fig.suptitle(title, fontsize=10, fontweight='bold')
    dpi = 500
    figname = 'Nweight_acceptance_dump'
    #plt.savefig('plots/dump/'+figname+'.png',dpi=dpi)
    plt.savefig('plots/dump/'+figname+'.pdf',dpi=dpi)


if __name__ == '__main__': main()
