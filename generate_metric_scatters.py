import sys
import argparse
import numpy
import statistics
import itertools
import pickle
import matplotlib
from matplotlib import pyplot as plt

#import pdb

from fileio_utils import read_coupling_file, get_events, retrieve_reco_weights, get_combined_cutflow_values
from combination_utils import is_valid_combination
#from combination_utils import basis_full3D_old_minN as _reco_basis 
from combination_utils import basis_full3D_max as _reco_basis 
from reweight_utils import reco_reweight
from negative_weight_map import get_negative_weight_grid



def metric_Nweight_integral(couplings, weights, errors, kv_val, k2v_val_range, kl_val_range):
    grid_pixel_area = (k2v_val_range[1] - k2v_val_range[0]) * (kl_val_range[1] - kl_val_range[0])
    negative_weight_grid = get_negative_weight_grid(couplings, weights, errors, kv_val, k2v_val_range, kl_val_range)
    grid_integral = numpy.sum( negative_weight_grid * grid_pixel_area )
    return grid_integral

def metric_accXeff_sum(cutflows):
    total_accXeff = 0
    for cuts in cutflows:
        accXeff = cuts['Signal'] / cuts['Initial']
        total_accXeff += accXeff
    return total_accXeff

def metric_accXeff_rms(cutflows):
    rms = 0
    for cuts in cutflows:
        accXeff = cuts['Signal'] / cuts['Initial']
        rms += accXeff**2
    rms = rms**(1/2)
    return rms

def metric_accXeff_sigma(cutflows):
    accXeff_list = [ cuts['Signal'] / cuts['Initial'] for cuts in cutflows ]
    return statistics.stdev(accXeff_list)

def metric_accXeff_avg_stdev(cutflows):
    accXeff_list = [ cuts['Signal'] / cuts['Initial'] for cuts in cutflows ]
    average = statistics.mean(accXeff_list) 
    std_dev = statistics.stdev(accXeff_list)
    return average / std_dev

def metric_accXeff_min(cutflows):
    accXeff_min = 100
    for cuts in cutflows:
        accXeff = cuts['Signal'] / cuts['Initial']
        if accXeff < accXeff_min:
            accXeff_min = accXeff
    return accXeff_min

def metric_accXeff_harmonic_mean(cutflows):
    reciprocal_sum = 0
    for cuts in cutflows:
        accXeff = cuts['Signal'] / cuts['Initial']
        reciprocal_sum += 1 / accXeff
    reciprocal_average = reciprocal_sum / len(cutflows)
    harmonic_mean = 1 / reciprocal_average
    return harmonic_mean

def metric_eventCount_sum(events_list):
    total_events = 0
    for weights, errors in events_list:
        total_events += len(weights)
    return total_events



def generate_metric_values():
    var_edges = numpy.linspace(200, 1200, 31)
    kv_val = 1.0
    k2v_val_range = numpy.linspace(-2,4,31)
    kl_val_range = numpy.linspace(-14,16,31)

    data_files = read_coupling_file('basis_files/nnt_coupling_file.dat')
    all_cutflows = get_combined_cutflow_values(data_files.keys(), data_files).values() # It's a really good things that python dicts are ordered...
    all_events = get_events(data_files.keys(), data_files)
    all_histograms = [ retrieve_reco_weights(var_edges,events) for events in all_events ]
    # Wrap all variations up together with their histograms so I can find combinations
    all_variations = list(zip(data_files.keys(), all_histograms, all_cutflows, all_events))#[:7]
    print('All variations loaded, proceeding to retrieve metrics...')

    total = 0
    basis_metrics = {}
    for basis_set in itertools.combinations(all_variations,6):
        # Unwrap each combination
        couplings, histograms, cutflows, events_list = list(zip(*basis_set))
        if not is_valid_combination(couplings): continue
        #if (1.0,1.0,1.0) not in couplings: continue

        weights, errors = numpy.array( list(zip(*histograms)) )

        basis_metrics[couplings] = {
            'Nweight_integral': metric_Nweight_integral(couplings, weights, errors, kv_val, k2v_val_range, kl_val_range),
            'accXeff_sum': metric_accXeff_sum(cutflows),
            'accXeff_rms': metric_accXeff_rms(cutflows),
            'accXeff_avg_stdev': metric_accXeff_avg_stdev(cutflows),
            'accXeff_min': metric_accXeff_min(cutflows),
            'accXeff_sigma': metric_accXeff_sigma(cutflows),
            'accXeff_harmonic': metric_accXeff_harmonic_mean(cutflows),
            'eventCount_sum': metric_eventCount_sum(events_list)
        }

        total += 1
        if total % 10 == 0: print(total)
    print('Variations traversed, plotting...')
    metric_lists = { key:[] for key in list(basis_metrics.values())[0].keys() }
    for basis, metrics in basis_metrics.items():
        for key,val in metrics.items():
            metric_lists[key].append(val)
    return metric_lists


def main():
    use_cached = len(sys.argv) > 1
    metric_lists = None
    cache_file = '.cached_basis_metrics.p'
    if use_cached:
        metric_lists = pickle.load(open(cache_file,'rb'))
    else:
        metric_lists = generate_metric_values()
        pickle.dump(metric_lists, open(cache_file,'wb'))

    plot_specs = {
        'Nweight_integral': ( 'Negative Weight Integral', '' ),
        'accXeff_sum': ( 'Acceptance X Efficiency Sum', '' ),
        'accXeff_min': ( 'Acceptance X Efficiency Minimum', '' ),
        'accXeff_rms': ( 'Acceptance X Efficiency RMS', '' ),
        'accXeff_avg_stdev': ( 'Acceptance X Efficiency 'f'$\mu/\sigma$', '' ),
        'accXeff_sigma': ( 'Acceptance X Efficiency Standard Deviation', '' ),
        'accXeff_harmonic': ( 'Acceptance X Efficiency Harmonic Mean', '' ),
        'eventCount_sum': ( 'Event Count Sum', '' )
    }

    plot_list = [ (plot, 'Nweight_integral') for plot in plot_specs.keys() ][1:] #skip plotting Nweight against itself

    dpi=500
    for x,y in plot_list:
        plt.scatter(metric_lists[x], metric_lists[y])
        plt.xlabel(plot_specs[x][0]+plot_specs[x][1])
        plt.ylabel(plot_specs[y][0]+plot_specs[y][1])
        plt.title(plot_specs[x][0]+'\nVS '+plot_specs[y][0])
        plt.savefig('plots/metrics/'+x+'_VS_'+y+'.png', dpi=dpi)
        plt.close()


if __name__ == '__main__': main()
