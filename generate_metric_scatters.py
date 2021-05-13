import sys
import argparse
import math
import numpy
import statistics
import itertools
import pickle
import matplotlib
from matplotlib import pyplot as plt

#import pdb

from fileio_utils import read_coupling_file, get_events, retrieve_reco_weights, get_combined_cutflow_values
from combination_utils import is_valid_combination, orthogonality_metric, get_amplitude_function, get_theory_xsec_function
import combination_utils
#from combination_utils import basis_full3D_old_minN as _reco_basis 
from combination_utils import basis_full3D_max as _reco_basis 
from reweight_utils import reco_reweight
from negative_weight_map import get_Nweight_sum
from effective_stats_map import get_effective_stats_grid
from weight_contribution_map import get_theoretical_solidarity_map, get_reco_solidarity_map, get_theory_effective_stats_map, get_test_map



def metric_reco_effective_stats_integral(couplings, events_list, kv_val, k2v_val_range, kl_val_range):
    grid_pixel_area = (k2v_val_range[1] - k2v_val_range[0]) * (kl_val_range[1] - kl_val_range[0])
    effective_stats_grid = get_effective_stats_grid(couplings, events_list, kv_val, k2v_val_range, kl_val_range)
    grid_integral = numpy.sum( effective_stats_grid * grid_pixel_area )
    return 1/grid_integral

def metric_orthogonality(couplings):
    return orthogonality_metric(couplings)

def metric_theory_effective_stats_integral(couplings, kv_val, k2v_val_range, kl_val_range):
    grid_pixel_area = (k2v_val_range[1] - k2v_val_range[0]) * (kl_val_range[1] - kl_val_range[0])
    contribution_grid = get_theory_effective_stats_map(couplings, kv_val, k2v_val_range, kl_val_range)
    grid_integral = numpy.sum( contribution_grid * grid_pixel_area )
    #return math.log(grid_integral)
    return 1 / grid_integral
    #return grid_integral

def metric_theory_test_val(couplings):
    inverse_vector = combination_utils.get_inversion_vector(couplings)
    print(inverse_vector)

def metric_contribution_integral(couplings, kv_val, k2v_val_range, kl_val_range):
    grid_pixel_area = (k2v_val_range[1] - k2v_val_range[0]) * (kl_val_range[1] - kl_val_range[0])
    contribution_grid = get_test_map(couplings, kv_val, k2v_val_range, kl_val_range)
    grid_integral = numpy.sum( contribution_grid * grid_pixel_area )
    return math.log(grid_integral)

def metric_accXeff_list(cutflows):
    return [ cuts['Signal'] / cuts['Initial'] for cuts in cutflows ]

def metric_accXeff_sum(cutflows):
    total_accXeff = 0
    for cuts in cutflows:
        accXeff = cuts['Signal'] / cuts['Initial'] * 1e5
        total_accXeff += accXeff
    return total_accXeff

def metric_accXeff_rms(cutflows):
    rms = 0
    for cuts in cutflows:
        accXeff = cuts['Signal'] / cuts['Initial']
        rms += accXeff**2
    rms = rms**(1/2)
    return rms

def metric_accXeff_geometric_mean(cutflows):
    accXeff_list = [ cuts['Signal'] / cuts['Initial'] for cuts in cutflows ]
    return statistics.geometric_mean(accXeff_list)

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
        #print(len(weights))
        total_events += len(weights)
    #print()
    return total_events



def generate_metric_values():
    var_edges = numpy.linspace(200, 1200, 31)
    kv_val = 1.0
    num_bins = 100
    k2v_val_range = numpy.linspace(-2,4,num_bins+1)
    kl_val_range = numpy.linspace(-14,16,num_bins+1)

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
            'Nweight_integral': get_Nweight_sum(couplings, weights, kv_val, k2v_val_range, kl_val_range),
            #'orthogonality': metric_orthogonality(couplings),
            #'reco_effective_stats_integral': metric_reco_effective_stats_integral(couplings, events_list, kv_val, k2v_val_range, kl_val_range),
            #'theory_effective_stats_integral': metric_theory_effective_stats_integral(couplings, kv_val, k2v_val_range, kl_val_range),
            'reco_solidarity_integral': get_reco_solidarity_map(couplings, weights, kv_val, k2v_val_range, kl_val_range),
            'theory_solidarity_integral': get_theoretical_solidarity_map(couplings, kv_val, k2v_val_range, kl_val_range),
            #'theory_test_val': metric_theory_test_val(couplings),
            #'contribution_integral': metric_contribution_integral(couplings, kv_val, k2v_val_range, kl_val_range),
            #'accXeff_list': metric_accXeff_list(cutflows),
            #'accXeff_sum': metric_accXeff_sum(cutflows),
            #'accXeff_geometric': metric_accXeff_geometric_mean(cutflows),
            #'accXeff_rms': metric_accXeff_rms(cutflows),
            #'accXeff_avg_stdev': metric_accXeff_avg_stdev(cutflows),
            #'accXeff_min': metric_accXeff_min(cutflows),
            #'accXeff_sigma': metric_accXeff_sigma(cutflows),
            #'accXeff_harmonic': metric_accXeff_harmonic_mean(cutflows),
            #'eventCount_sum': metric_eventCount_sum(events_list)
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
    
    #raw_output = [ (nwi, acc) for nwi, acc in zip(metric_lists['Nweight_integral'], metric_lists['accXeff_list']) ]
    #raw_output.sort()
    #for nwi, acc in raw_output:
    #    basis_str = f'{int(nwi):4d}: '
    #    acc.sort()
    #    for a in acc: basis_str += f'{int(a*10**6):2d}, '
    #    print(basis_str)

    #exit()

    numpy.set_printoptions(precision=None, linewidth=400, threshold=100, sign=' ', formatter={'float':lambda n: f'{n: 4.1f}'}, floatmode='fixed')
    plot_specs = {
        'Nweight_integral': ( 'Negative Weight Integral', '' ),
        'orthogonality': ( 'Orthogonality', '' ),
        'reco_effective_stats_integral': ( 'Effective Statistics Integral', ''),
        'reco_solidarity_integral': ( 'Reconstructed Solidarity Integral', '' ),
        'theory_effective_stats_integral': ( 'Theoretical Effective Statistics Integral', ''),
        'theory_solidarity_integral': ( 'Theoretical Solidarity Integral', '' ),
        'contribution_integral': ( 'Contribution Count Integral', '' ),
        'accXeff_sum': ( 'Acceptance X Efficiency Sum', '' ),
        'accXeff_geometric': ( 'Acceptance X Efficiency Geometric Mean', '' ),
        'accXeff_min': ( 'Acceptance X Efficiency Minimum', '' ),
        'accXeff_rms': ( 'Acceptance X Efficiency RMS', '' ),
        'accXeff_avg_stdev': ( 'Acceptance X Efficiency 'f'$\mu/\sigma$', '' ),
        'accXeff_sigma': ( 'Acceptance X Efficiency Standard Deviation', '' ),
        'accXeff_harmonic': ( 'Acceptance X Efficiency Harmonic Mean', '' ),
        'eventCount_sum': ( 'Event Count Sum', '' )
    }

    plot_list = [ (plot, 'Nweight_integral') for plot in plot_specs.keys() if plot != 'Nweight_integral' ]
    plot_list += [ (plot, 'reco_effective_stats_integral') for plot in plot_specs.keys() if plot != 'reco_effective_stats_integral' ]

    dpi=500
    for x,y in plot_list:
        if not (x in metric_lists and y in metric_lists): continue

        #xy_tuples = list(zip(metric_lists[x], metric_lists[y]))
        #xvals, yvals = list(zip( *sorted(xy_tuples, reverse=True)[:10] ))
        xvals, yvals = metric_lists[x], metric_lists[y]
        plt.scatter(xvals, yvals)
        plt.xlabel(plot_specs[x][0]+plot_specs[x][1])
        plt.ylabel(plot_specs[y][0]+plot_specs[y][1])
        plt.title(plot_specs[y][0]+'\nVS '+plot_specs[x][0])
        plt.savefig('plots/metrics/'+y+'_VS_'+x+'.png', dpi=dpi)
        plt.close()


if __name__ == '__main__': main()
