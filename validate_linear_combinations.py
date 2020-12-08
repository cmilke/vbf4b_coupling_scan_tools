import sys
import argparse
import math
import sympy
import numpy
import uproot
import inspect

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import reweight_utils


def plot_histogram(hist_name, hist_title, edge_list, coupling_parameters,
        linearly_combined_weights, linearly_combined_errors,
        verification_weights, verification_errors):

    print('Plotting '+hist_name+' for ' + str(coupling_parameters))
    fig, (ax_main, ax_ratio) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios':[4,1]} )

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
    ax_ratio.set(ylabel=r'$\frac{lin. comb. - gen.}{stat. error}$', xlabel='Truth $m_{HH}$ (GeV)')
    ax_main.legend(prop={'size':7})
    ax_main.grid()
    ax_ratio.grid()

    dpi=500
    kappa_string_list = [ label.replace('.','p') for label in kappa_labels ]
    kappa_string = 'cvv'+kappa_string_list[0]+'cl'+kappa_string_list[1]+'cv'+kappa_string_list[2]
    fig.savefig('plots/validation/'+hist_name+'_'+kappa_string+'.png', dpi=dpi)
    plt.close()


def retrieve_truth_data(file_list):
    hist_key = b'HH_m'
    weight_list = []
    error_list = []
    edge_list = []
    for f in file_list:
        directory = uproot.open(f)
        root_hist = directory[hist_key]
        weights, edges = root_hist.numpy()
        errors = numpy.sqrt(root_hist.variances)
        weight_list.append(weights)
        error_list.append(errors)
        if len(edge_list) == 0: edge_list = edges
    return weight_list, error_list, edge_list



def retrieve_truth_combination(amplitude_function, basis_files):
    basis_weight_list, basis_error_list, edge_list = retrieve_truth_data(basis_files)
    combination_function = lambda params: amplitude_function(*params, *basis_weight_list)
    error_function = lambda params: amplitude_function(*params, *basis_error_list)
    return combination_function, error_function, edge_list



def validate_truth_combinations(basis_parameters, basis_files, verification_parameters, verification_files):
    amplitude_function = reweight_utils.get_amplitude_function(basis_parameters)
    combination_function, error_function, edge_list = retrieve_truth_combination(amplitude_function, basis_files)
    verification_weight_list, verification_error_list = retrieve_truth_data(verification_files)[:2]

    for index, coupling_parameters in enumerate(verification_parameters):
        linearly_combined_weights = combination_function(coupling_parameters)
        linearly_combined_errors = error_function(coupling_parameters)

        plot_histogram('truth_mHH', 'Full-Truth Linear Combination:\n$m_{HH}$', edge_list, coupling_parameters,
            linearly_combined_weights, linearly_combined_errors,
            verification_weight_list, verification_error_list)



def extract_ntuple_events(ntuple, mhh_key='m_hh', unit_conversion=1):
    ttree = uproot.rootio.open(ntuple)[b'sig']
    frame = ttree.pandas.df(branches=[mhh_key,'mc_sf'])
    masses  = frame[mhh_key][:,0].values * unit_conversion
    weights = frame['mc_sf'][:,0].values
    events = (masses,weights)
    return events



def retrieve_reco_weights(mHH_edges, reco_events, reweight_array=None):
    event_weights = reco_events[1]
    if type(reweight_array) != type(None): event_weights = reco_events[1]*reweight_array

    reco_weights = numpy.histogram(reco_events[0], bins=mHH_edges, weights=event_weights)[0]

    reco_errors = numpy.zeros( len(reco_weights) )
    event_bins = numpy.digitize(reco_events[0],mHH_edges)
    for i in range(len(reco_errors)):
        #binned_weights = reco_events[1][ event_bins == i ]
        binned_weights = event_weights[ event_bins == i ]
        error2_array = binned_weights**2
        error = math.sqrt( error2_array.sum() )
        reco_errors[i] = error

    return reco_weights, reco_errors



def truth_reweight( combination_components, coupling_parameters, sm_events, mHH_edges):
    combination_function, error_function, edge_list = combination_components

    linear_combination = combination_function(coupling_parameters)
    sm_truth_weights = combination_function([1,1,1])
    sm_truth_weights[ sm_truth_weights == 0 ] = float('inf') # Just to avoid Nan issues
    reweight_array = linear_combination / sm_truth_weights

    binned_SM = numpy.digitize(sm_events[0], edge_list)
    event_weights_multiplier = reweight_array[binned_SM]
    reweighted_reco_weights, reweighted_reco_error = retrieve_reco_weights(mHH_edges, sm_events, reweight_array=event_weights_multiplier)

    linearly_combined_errors = error_function([1,1,1])
    #combined_errors = linearly_combined_errors
    combined_errors = reweighted_reco_error

    return reweighted_reco_weights, combined_errors



def validate_reweighting_method(basis_parameters, basis_files, verification_parameters, verification_files):
    amplitude_function = reweight_utils.get_amplitude_function(basis_parameters)
    combination_components = retrieve_truth_combination(amplitude_function, basis_files)

    #verification_events_list = [ extract_ntuple_events(v,mhh_key='truth_mhh',unit_conversion=1/1000) for v in verification_files ]
    verification_events_list = [ extract_ntuple_events(v) for v in verification_files ]
    sm_events = verification_events_list[0] #Assume SM file is the zeroth
    #sm_events = extract_ntuple_events(verification_files[0], mhh_key='truth_mhh', unit_conversion=1/1000)

    mHH_edges = numpy.arange(0, 2050, 50)
    for index, coupling_parameters in enumerate(verification_parameters):
        verification_weights, verification_errors = retrieve_reco_weights(mHH_edges, verification_events_list[index])
        combined_weights, combined_errors = truth_reweight( combination_components, coupling_parameters, sm_events, mHH_edges)

        plot_histogram('rwgt_mHH', 'Truth-Reweighted NNT:\n$m_{HH}$', mHH_edges, coupling_parameters,
                 combined_weights, combined_errors,
                 verification_weights, verification_errors
        )


def reco_reweight(mHH_edges, reweight_vector, coupling_parameters, base_events_list):

    multiplier_vector = reweight_vector(*coupling_parameters)[0]

    reweighted_events_list = []
    for multiplier, events in zip(multiplier_vector, base_events_list):
        reweighted_events_list.append( numpy.array( [events[0], events[1]*multiplier] ) )
    reweighted_events = numpy.concatenate(reweighted_events_list, axis=1)

    #combined_weights = numpy.histogram(reweighted_events[0], bins=mHH_edges, weights=reweighted_events[1])[0]
    return retrieve_reco_weights(mHH_edges, reweighted_events)



def validate_reco_method(basis_parameters, basis_files, verification_parameters, verification_files):
    reweight_vector = reweight_utils.get_amplitude_function(basis_parameters, as_scalar=False)

    base_events_list = [ extract_ntuple_events(b) for b in basis_files ]
    verification_events_list = [ extract_ntuple_events(v) for v in verification_files ]

    mHH_edges = numpy.arange(0, 2050, 50)
    for verification_events, coupling_parameters in zip(verification_events_list, verification_parameters):
        verification_weights, verification_errors = retrieve_reco_weights(mHH_edges, verification_events)
        combined_weights, combined_errors = reco_reweight(mHH_edges, reweight_vector, coupling_parameters, base_events_list)

        plot_histogram('reco_mHH', 'NNT-Based Linear Combination:\n$m_{HH}$', mHH_edges, coupling_parameters,
                 combined_weights, combined_errors,
                 verification_weights, verification_errors
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
