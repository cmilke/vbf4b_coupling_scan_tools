import sys
import argparse
import math
import sympy
import numpy
import uproot
import inspect

#import pdb

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import reweight_utils

_scan_terms = reweight_utils.full_scan_terms
#_scan_terms = reweight_utils.k2v_scan_terms


def plot_histogram(hist_name, hist_title, edge_list, coupling_parameters,
        linearly_combined_weights, linearly_combined_errors,
        verification_weights, verification_errors, xlabel='Truth $m_{HH}$ (GeV)', normalize=False):

    print('Plotting '+hist_name+' for ' + str(coupling_parameters))
    fig, (ax_main, ax_ratio) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios':[4,1]} )

    if normalize:
        linear_normalization = linearly_combined_weights.sum()
        verification_normalization = verification_weights.sum()

        linearly_combined_weights /= linear_normalization
        #linearly_combined_errors /= linear_normalization
        linearly_combined_errors /= 10000
        verification_weights /= verification_normalization
        verification_errors /= verification_normalization


    xpositions = 0.5*(edge_list[1:]+edge_list[:-1])
    counts, bins, points = ax_main.errorbar( xpositions, linearly_combined_weights,
        yerr=linearly_combined_errors, label='Linear Combination',
        marker='+', markersize=2, capsize=2, color='blue', linestyle='none', linewidth=1, zorder=3)

    vcounts, vbins, vhists = ax_main.hist( [edge_list[:-1]]*2,
        weights=[verification_weights-verification_errors, 2*verification_errors],
        label=['Generated MC', 'MC Statistical Error'],
        bins=edge_list, fill=True, histtype='barstacked', zorder=1, alpha=0.5, color=['green','red'])
    plt.setp(vhists[1], hatch='/////')

    safe_error = verification_errors.copy()
    safe_error[ safe_error == 0 ] = float('inf')
    rcounts, rbins, rpoints = ax_ratio.errorbar( xpositions, (linearly_combined_weights-verification_weights)/safe_error,
        yerr=linearly_combined_errors/safe_error, label='MC Statistical Error check',
        marker='+', markersize=2, capsize=2, color='blue', linestyle='none', linewidth=1, zorder=3)
    
    zero_line = ax_ratio.hlines(0,xmin=edge_list[0],xmax=edge_list[-1],colors='black',zorder=2)

    kappa_labels = [ str(param) for param in coupling_parameters ]
    title  = hist_title+' for '
    title += '$\kappa_{2V}='+kappa_labels[0]+'$, '
    title += '$\kappa_{\lambda}='+kappa_labels[1]+'$, '
    title += '$\kappa_{V}='+kappa_labels[2]+'$'
    fig.suptitle(title)

    ax_main.set(ylabel='Bin Weight')
    ax_ratio.set_ylim([-2,2])
    ax_ratio.set_yticks(ticks=[-2,-1,0,1,2])
    ax_ratio.set(ylabel=r'$\frac{lin. comb. - gen.}{stat. error}$', xlabel=xlabel)
    ax_main.legend(prop={'size':7})
    ax_main.grid()
    ax_ratio.grid()

    dpi=500
    kappa_string_list = [ label.replace('.','p') for label in kappa_labels ]
    kappa_string = 'cvv'+kappa_string_list[0]+'cl'+kappa_string_list[1]+'cv'+kappa_string_list[2]
    fig.savefig('plots/validation/'+hist_name+'_'+kappa_string+'.png', dpi=dpi)
    plt.close()


def extract_truth_data(file_list, hist_key=b'HH_m'):
    weight_list = []
    error_list = []
    edge_list = []
    for f in file_list:
        directory = uproot.open(f)
        root_hist = directory[hist_key]
        weights, edges = root_hist.numpy()
        errors = numpy.sqrt(root_hist.variances)
        weight_list.append(weights)
        #linear_combination[ abs(linear_combination) == float('inf') ] = 0 # Just to avoid Nan issues
        error_list.append(errors)
        if len(edge_list) == 0: edge_list = edges
    return weight_list, error_list, edge_list



def retrieve_truth_combination(amplitude_function, basis_files, hist_key=b'HH_m'):
    basis_weight_list, basis_error_list, edge_list = extract_truth_data(basis_files, hist_key=hist_key)
    combination_function = lambda params: amplitude_function(*params, *basis_weight_list)
    error_function = lambda params: amplitude_function(*params, *basis_error_list)
    return combination_function, basis_weight_list, basis_error_list, edge_list


def extract_ntuple_events(ntuple, mhh_key='m_hh', unit_conversion=1):
    ttree = uproot.rootio.open(ntuple)[b'sig']
    frame = ttree.pandas.df(branches=[mhh_key,'mc_sf'])
    #frame = ttree.pandas.df(branches=[mhh_key,'mc_sf','pass_vbf_sel'])
    #frame = frame[ frame['pass_vbf_sel'] ]
    masses  = frame[mhh_key][:,0].values * unit_conversion
    weights = frame['mc_sf'][:,0].values
    events = (masses,weights)
    return events



def retrieve_reco_weights(mHH_edges, reco_events):
    event_weights = reco_events[1]
    reco_weights = numpy.histogram(reco_events[0], bins=mHH_edges, weights=event_weights)[0]
    reco_errors = numpy.zeros( len(reco_weights) )
    event_bins = numpy.digitize(reco_events[0],mHH_edges)
    for i in range(len(reco_errors)):
        binned_weights = event_weights[ event_bins == i ]
        error2_array = binned_weights**2
        error = math.sqrt( error2_array.sum() )
        reco_errors[i] = error

    return reco_weights, reco_errors


def truth_reweight( basis_parameters, combination_components, coupling_parameters, reco_base_bins, mHH_edges):
    combination_function, basis_weight_list, basis_error_list, edge_list = combination_components
    basis_list = [ [eval(n) for n in b ] for b in basis_parameters ]

    linear_combination = combination_function(coupling_parameters)
    reweight_vector = reweight_utils.get_amplitude_function(basis_parameters, base_equations=_scan_terms, as_scalar=False)(*coupling_parameters)[0]

    truth_base_weights = basis_weight_list[0].copy()
    truth_base_weights[ truth_base_weights == 0. ] = float('inf') # Just to avoid Nan issues
    reweight_array = linear_combination / truth_base_weights
    reco_truth_ratio = reco_base_bins[0] / truth_base_weights

    reweighted_reco_weights = reweight_array * reco_base_bins[0]

    base_error2 = ( reweight_array * reco_base_bins[1] )**2
    truth_error2 = reco_truth_ratio**2 * numpy.array([ (e*m)**2 for e,m in zip(basis_error_list[1:], reweight_vector[1:]) ]).sum(axis=0)
    base_truth_error2 = reco_truth_ratio**2 * ( reweight_vector[0] - reweight_array )**2
    combined_errors = numpy.sqrt( base_error2 + truth_error2 + base_truth_error2 )

    return reweighted_reco_weights, combined_errors



def reco_reweight(mHH_edges, reweight_vector, coupling_parameters, base_weights, base_errors):
    multiplier_vector = reweight_vector(*coupling_parameters)[0]

    reweighted_weights = numpy.array([ w*m for w,m in zip(base_weights, multiplier_vector) ])
    linearly_combined_weights = reweighted_weights.sum(axis=0)

    reweighted_errors2 = numpy.array([ (w*m)**2 for w,m in zip(base_errors, multiplier_vector) ])
    linearly_combined_errors = numpy.sqrt( reweighted_errors2.sum(axis=0) )

    return linearly_combined_weights, linearly_combined_errors



def validate_truth_combinations(basis_parameters, basis_files, verification_parameters, verification_files):
    amplitude_function = reweight_utils.get_amplitude_function(basis_parameters, base_equations=_scan_terms)
    combination_function, basis_weight_list, basis_error_list, edge_list = retrieve_truth_combination(amplitude_function, basis_files)
    verification_weight_list, verification_error_list = extract_truth_data(verification_files)[:2]

    for index, coupling_parameters in enumerate(verification_parameters):
        linearly_combined_weights = combination_function(coupling_parameters)
        linearly_combined_errors = error_function(coupling_parameters)

        plot_histogram('truth_mHH', 'Full-Truth Linear Combination:\n$m_{HH}$', edge_list, coupling_parameters,
            linearly_combined_weights, linearly_combined_errors,
            verification_weight_list, verification_error_list)



def validate_reweighting_method(basis_parameters, basis_files, verification_parameters, verification_files):
    amplitude_function = reweight_utils.get_amplitude_function(basis_parameters, base_equations=_scan_terms)
    combination_components = retrieve_truth_combination(amplitude_function, basis_files)

    verification_events_list = [ extract_ntuple_events(v,mhh_key='truth_mhh',unit_conversion=1/1000) for v in verification_files ]
    reco_base_events = verification_events_list[0] # 0th couplings for basis and verification must be the same

    mHH_edges = combination_components[-1]
    reco_base_bins = retrieve_reco_weights(mHH_edges, reco_base_events)
    for index, coupling_parameters in enumerate(verification_parameters):
        verification_weights, verification_errors = retrieve_reco_weights(mHH_edges, verification_events_list[index])
        combined_weights, combined_errors = truth_reweight( basis_parameters, combination_components, coupling_parameters, reco_base_bins, mHH_edges)
        plot_histogram('rwgt_mHH', 'Truth-Reweighted NNT:\n$m_{HH}$', mHH_edges, coupling_parameters,
                 combined_weights, combined_errors,
                 verification_weights, verification_errors, normalize=True
        )


def validate_reco_method(basis_parameters, basis_files, verification_parameters, verification_files):
    reweight_vector = reweight_utils.get_amplitude_function(basis_parameters, as_scalar=False, base_equations=_scan_terms)
    mHH_edges = numpy.arange(0, 2050, 50)

    verification_events_list = [ extract_ntuple_events(v) for v in verification_files ]
    base_events_list = [ extract_ntuple_events(b) for b in basis_files ]
    base_histograms = [ retrieve_reco_weights(mHH_edges, base_events) for base_events in base_events_list ]
    base_weights, base_errors = numpy.array(list(zip(*base_histograms)))

    for verification_events, coupling_parameters in zip(verification_events_list, verification_parameters):
        verification_weights, verification_errors = retrieve_reco_weights(mHH_edges, verification_events)
        combined_weights, combined_errors = reco_reweight(mHH_edges, reweight_vector, coupling_parameters, base_weights, base_errors)

        plot_histogram('reco_mHH', 'NNT-Based Linear Combination:\n$m_{HH}$', mHH_edges, coupling_parameters,
                 combined_weights, combined_errors,
                 verification_weights, verification_errors,
                 xlabel='Reconstructed $m_{HH}$ (GeV)'
        )



def main():
    # Sort out command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( "--basis", required = True, default = None, type=str,
        help = "File to provide basis states",)

    parser.add_argument( "--verify", required = True, default = None, type=str,
        help = "File containing a list of states to verify against",) 

    parser.add_argument( "--mode", required = True, default = None, type=str,
        help = "Options are: truth, reweight, or reco",) 

    args = parser.parse_args()

    # Read in base parameters file
    basis_parameters = []
    basis_files = []
    with open(args.basis) as basis_list_file:
        for line in basis_list_file:
            if line.strip().startswith('#'): continue
            linedata = line.split()
            if len(linedata) < 3: continue
            basis_parameters.append(linedata[:3])
            basis_files.append(linedata[3])

    # Read in parameter list to be generated by linear combination
    verification_parameters = []
    verification_files = []
    with open(args.verify) as plot_list_file:
        for line in plot_list_file:
            if line.strip().startswith('#'): continue
            linedata = line.split()
            params = [ float(p) for p in linedata[:3] ]
            verification_parameters.append(params)
            verification_files.append(linedata[3])

    #pdb.set_trace()
    if args.mode == 'truth':
        validate_truth_combinations(basis_parameters, basis_files, verification_parameters, verification_files)
    elif args.mode == 'reweight':
        validate_reweighting_method(basis_parameters, basis_files, verification_parameters, verification_files)
    elif args.mode == 'reco':
        validate_reco_method(basis_parameters, basis_files, verification_parameters, verification_files)
    else:
        print('Mode - '+str(args.mode)+' - is not valid.')
        print('Please choose from:\ntruth\nreweight\nreco\n')
        print('Aborting')
        exit(1)


if __name__ == '__main__': main()
