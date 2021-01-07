import sys
import argparse
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#import pdb

import combination_utils
import reweight_utils

_scan_terms = combination_utils.full_scan_terms
#_scan_terms = combination_utils.k2v_scan_terms


def plot_histogram(hist_name, hist_title, edge_list, coupling_parameters,
        linearly_combined_weights, linearly_combined_errors,
        verification_weights, verification_errors,
        range_specs=None, xlabel='Truth $m_{HH}$ (GeV)', normalize=False,
        generated_label='Linear Combination', generated_color='blue'):

    print('Plotting '+hist_name+' for ' + str(coupling_parameters))
    fig, (ax_main, ax_ratio) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios':[4,1]} )

    if type(range_specs) != type(None):
        linearly_combined_weights = linearly_combined_weights[range_specs[0]:range_specs[1]]
        linearly_combined_errors = linearly_combined_errors[range_specs[0]:range_specs[1]]
        verification_weights = verification_weights[range_specs[0]:range_specs[1]]
        verification_errors = verification_errors[range_specs[0]:range_specs[1]]
        edge_list = edge_list[range_specs[0]:range_specs[1]+1]

    if normalize:
        linear_normalization = linearly_combined_weights.sum()
        verification_normalization = verification_weights.sum()

        linearly_combined_weights /= linear_normalization
        linearly_combined_errors /= linear_normalization
        verification_weights /= verification_normalization
        verification_errors /= verification_normalization


    xpositions = 0.5*(edge_list[1:]+edge_list[:-1])
    counts, bins, points = ax_main.errorbar( xpositions, linearly_combined_weights,
        yerr=linearly_combined_errors, label=generated_label,
        marker='+', markersize=2, capsize=2, color=generated_color, linestyle='none', linewidth=1, zorder=3)

    vcounts, vbins, vhists = ax_main.hist( [edge_list[:-1]]*2,
        weights=[verification_weights-verification_errors, 2*verification_errors],
        label=['Generated MC', 'MC Statistical Error'],
        bins=edge_list, fill=True, histtype='barstacked', zorder=1, alpha=0.5, color=['green','red'])
    plt.setp(vhists[1], hatch='/////')

    safe_error = verification_errors.copy()
    safe_error[ safe_error == 0 ] = float('inf')
    safe_combination = linearly_combined_weights.copy()
    safe_combination[ safe_combination == 0 ] = float('inf')
    #rcounts, rbins, rpoints = ax_ratio.errorbar( xpositions, (linearly_combined_weights-verification_weights)/safe_error,
    #    yerr=linearly_combined_errors/safe_error, label='MC Statistical Error check',
    #    marker='+', markersize=2, capsize=2, color=generated_color, linestyle='none', linewidth=1, zorder=3)
    rcounts, rbins, rpoints = ax_ratio.errorbar( xpositions, verification_weights/safe_combination,
        yerr=verification_errors/safe_combination, label='MC Statistical Error check',
        marker='+', markersize=2, capsize=2, color=generated_color, linestyle='none', linewidth=1, zorder=3)
    
    zero_line = ax_ratio.hlines(1,xmin=edge_list[0],xmax=edge_list[-1],colors='black',zorder=2)

    kappa_labels = [ str(param) for param in coupling_parameters ]
    title  = hist_title+' for '
    title += '$\kappa_{2V}='+kappa_labels[0]+'$, '
    title += '$\kappa_{\lambda}='+kappa_labels[1]+'$, '
    title += '$\kappa_{V}='+kappa_labels[2]+'$'
    fig.suptitle(title)

    ax_main.set(ylabel='Bin Weight')
    ax_ratio.set_ylim([0.5,1.5])
    ax_ratio.set_yticks(ticks=[0.5,1,1.5])
    #ax_ratio.set(ylabel=r'$\frac{lin. comb. - gen.}{stat. error}$', xlabel=xlabel)
    ax_ratio.set(ylabel=r'$\frac{generated}{lin. combination}$', xlabel=xlabel)
    ax_main.legend(prop={'size':7})
    ax_main.grid()
    ax_ratio.grid()

    dpi=500
    kappa_string_list = [ label.replace('.','p') for label in kappa_labels ]
    kappa_string = 'cvv'+kappa_string_list[0]+'cl'+kappa_string_list[1]+'cv'+kappa_string_list[2]
    fig.savefig('plots/validation/'+hist_name+'_'+kappa_string+'.png', dpi=dpi)
    plt.close()



def plot_dual_histogram(hist_name, hist_title, edge_list, coupling_parameters,
        linearly_combined_weights, linearly_combined_errors,
        reweighted_sm_weights, reweighted_sm_errors,
        verification_weights, verification_errors,
        range_specs=None, xlabel='Truth $m_{HH}$ (GeV)', normalize=False):

    print('Plotting '+hist_name+' for ' + str(coupling_parameters))
    fig, (ax_main, ax_ratio) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios':[4,1]} )

    if type(range_specs) != type(None):
        linearly_combined_weights = linearly_combined_weights[range_specs[0]:range_specs[1]]
        linearly_combined_errors = linearly_combined_errors[range_specs[0]:range_specs[1]]
        reweighted_sm_weights = reweighted_sm_weights[range_specs[0]:range_specs[1]]
        reweighted_sm_errors = reweighted_sm_errors[range_specs[0]:range_specs[1]]
        verification_weights = verification_weights[range_specs[0]:range_specs[1]]
        verification_errors = verification_errors[range_specs[0]:range_specs[1]]
        edge_list = edge_list[range_specs[0]:range_specs[1]+1]

    if normalize:
        linear_normalization = linearly_combined_weights.sum()
        verification_normalization = verification_weights.sum()

        linearly_combined_weights /= linear_normalization
        linearly_combined_errors /= linear_normalization
        reweighted_sm_weights /= linear_normalization
        reweighted_sm_errors /= linear_normalization
        verification_weights /= verification_normalization
        verification_errors /= verification_normalization


    xpositions = 0.5*(edge_list[1:]+edge_list[:-1])
    counts, bins, points = ax_main.errorbar( xpositions, linearly_combined_weights,
        yerr=linearly_combined_errors, label='Linear Combination',
        marker='+', markersize=2, capsize=2, color='blue', linestyle='none', linewidth=1, zorder=3)

    counts2, bins2, points2 = ax_main.errorbar( xpositions, reweighted_sm_weights,
        yerr=reweighted_sm_errors, label='Reweighted SM',
        marker='+', markersize=2, capsize=2, color='orange', linestyle='none', linewidth=1, zorder=3)

    vcounts, vbins, vhists = ax_main.hist( [edge_list[:-1]]*2,
        weights=[verification_weights-verification_errors, 2*verification_errors],
        label=['Generated MC', 'MC Statistical Error'],
        bins=edge_list, fill=True, histtype='barstacked', zorder=1, alpha=0.5, color=['green','red'])
    plt.setp(vhists[1], hatch='/////')

    safe_error = verification_errors.copy()

    safe_error[ safe_error == 0 ] = float('inf')
    rcounts, rbins, rpoints = ax_ratio.errorbar( xpositions, (linearly_combined_weights-verification_weights)/safe_error,
        yerr=linearly_combined_errors/safe_error, label='MC Statistical Error Check',
        marker='+', markersize=2, capsize=2, color='blue', linestyle='none', linewidth=1, zorder=3)

    rcounts2, rbins2, rpoints2 = ax_ratio.errorbar( xpositions, (reweighted_sm_weights-verification_weights)/safe_error,
        yerr=reweighted_sm_errors/safe_error, label='MC Statistical Error Check 2',
        marker='+', markersize=2, capsize=2, color='orange', linestyle='none', linewidth=1, zorder=3)
    
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






def validate_truth_combinations(basis_parameters, basis_files, verification_parameters, verification_files):
    basis_weight_list, basis_error_list, edge_list = reweight_utils.extract_truth_data(basis_files)
    amplitude_function = combination_utils.get_amplitude_function(basis_parameters, base_equations=_scan_terms)
    vector_function = combination_utils.get_amplitude_function(basis_parameters, base_equations=_scan_terms, as_scalar=False)
    combination_function = lambda params: amplitude_function(*params, *basis_weight_list)
    coefficient_function = lambda params: vector_function(*params)[0]

    verification_weight_list, verification_error_list = reweight_utils.extract_truth_data(verification_files)[:2]
    for index, coupling_parameters in enumerate(verification_parameters):
        linearly_combined_weights, linearly_combined_errors = reweight_utils.obtain_linear_combination(coupling_parameters, combination_function, coefficient_function, basis_error_list)
        plot_histogram('truth_mHH', 'Full-Truth Linear Combination:\n$m_{HH}$', edge_list, coupling_parameters,
            linearly_combined_weights, linearly_combined_errors,
            verification_weight_list[index], verification_error_list[index])



def validate_truth_reweighting_method(basis_parameters, basis_files, verification_parameters, verification_files):
    mHH_edges = numpy.linspace(0, 6000, num=600)
    amplitude_function = combination_utils.get_amplitude_function(basis_parameters, base_equations=_scan_terms)
    combination_components = reweight_utils.retrieve_lhe_truth_combination(amplitude_function, basis_files, mHH_edges)

    kinematic_variables = [
        ['HH_m', '$m_{HH}$', (0,6000,100, (0,2000)), 'GeV' ],
        ['HH_pt', '$p_{T,HH}$', (0,1000,50, (0,1000)), 'GeV'],
        ['HH_eta', r'$\Delta\eta_{HH}$', (-8,8,32, (-8,8)), ''],
        ['jj_M', '$m_{jj}$', (0,10000,100, (0,6000)), 'GeV' ],
        ['jj_pT', '$p_{T,jj}$', (0,1000,50, (0,1000)), 'GeV'],
        ['jj_eta', r'$\Delta\eta_{jj}$', (-8,8,32, (-8,8)), '']
    ]
    kinematic_keys = list(zip(*kinematic_variables))[0]
    verification_events_list = [ reweight_utils.extract_lhe_events(v,kinematic_keys) for v in verification_files ]

    base_distro = 'HH_m'
    for key, title, bin_specs, units in kinematic_variables:
        bin_edges = numpy.linspace(bin_specs[0], bin_specs[1], num=bin_specs[2])
        combination_components2 = reweight_utils.retrieve_lhe_truth_combination(amplitude_function, basis_files, bin_edges)
        for index, coupling_parameters in enumerate(verification_parameters):
            verification_weights, verification_errors = reweight_utils.retrieve_lhe_weights(verification_events_list[index], key, bin_edges)
            reweighted_sm_weights, reweighted_sm_errors = reweight_utils.truth_truth_reweight( basis_parameters, combination_components, coupling_parameters, 
                    verification_events_list[0], base_distro, key, bin_edges)

            plot_histogram('truth_rwgt_'+str(key), 'Truth-Reweighted SM LHE:\n'+title, bin_edges, coupling_parameters,
                     reweighted_sm_weights, reweighted_sm_errors,
                     verification_weights, verification_errors,
                     range_specs=numpy.digitize(bin_specs[3], bin_edges),
                     xlabel='Truth '+title+( ' ('+units+')' if units != '' else '' ),
                     generated_label='Reweighted SM',
                     generated_color='orange'
            )

            #linearly_combined_weights = combination_components2[0](coupling_parameters)
            #linearly_combined_errors = amplitude_function(*coupling_parameters,*combination_components2[2])
            #plot_dual_histogram('truth_rwgt_DUAL_'+str(key), 'Truth-Reweighted SM LHE:\n'+title, bin_edges, coupling_parameters,
            #         linearly_combined_weights, linearly_combined_errors,
            #         reweighted_sm_weights, reweighted_sm_errors,
            #         verification_weights, verification_errors,
            #         range_specs=numpy.digitize(bin_specs[3], bin_edges),
            #)


def validate_reweighting_method(basis_parameters, basis_files, verification_parameters, verification_files):
    amplitude_function = combination_utils.get_amplitude_function(basis_parameters, base_equations=_scan_terms)
    basis_weight_list, basis_error_list, mHH_edges = reweight_utils.extract_truth_data(basis_files, hist_key=hist_key)
    combination_function = lambda params: amplitude_function(*params, *basis_weight_list)
    combination_components = combination_function, basis_weight_list, basis_error_list

    verification_events_list = [ reweight_utils.extract_ntuple_events(v,mhh_key='truth_mhh',unit_conversion=1/1000) for v in verification_files ]
    reco_base_events = verification_events_list[0] # 0th couplings for basis and verification must be the same

    reco_base_bins = reweight_utils.retrieve_reco_weights(mHH_edges, reco_base_events)
    for index, coupling_parameters in enumerate(verification_parameters):
        verification_weights, verification_errors = reweight_utils.retrieve_reco_weights(mHH_edges, verification_events_list[index])
        combined_weights, combined_errors = reweight_utils.truth_reweight( basis_parameters, combination_components, coupling_parameters, reco_base_bins, mHH_edges)
        plot_histogram('rwgt_mHH', 'Truth-Reweighted NNT:\n$m_{HH}$', mHH_edges, coupling_parameters,
                 combined_weights, combined_errors,
                 verification_weights, verification_errors
        )


def validate_reco_method(basis_parameters, basis_files, verification_parameters, verification_files):
    reweight_vector = combination_utils.get_amplitude_function(basis_parameters, as_scalar=False, base_equations=_scan_terms)
    mHH_edges = numpy.arange(0, 2050, 50)

    verification_events_list = [ reweight_utils.extract_ntuple_events(v) for v in verification_files ]
    base_events_list = [ reweight_utils.extract_ntuple_events(b) for b in basis_files ]
    base_histograms = [ reweight_utils.retrieve_reco_weights(mHH_edges, base_events) for base_events in base_events_list ]
    base_weights, base_errors = numpy.array(list(zip(*base_histograms)))

    for verification_events, coupling_parameters in zip(verification_events_list, verification_parameters):
        verification_weights, verification_errors = reweight_utils.retrieve_reco_weights(mHH_edges, verification_events)
        combined_weights, combined_errors = reweight_utils.reco_reweight(mHH_edges, reweight_vector, coupling_parameters, base_weights, base_errors)

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
    elif args.mode == 'rwgt_truth':
        validate_truth_reweighting_method(basis_parameters, basis_files, verification_parameters, verification_files)
    elif args.mode == 'reweight':
        validate_reweighting_method(basis_parameters, basis_files, verification_parameters, verification_files)
    elif args.mode == 'reco':
        validate_reco_method(basis_parameters, basis_files, verification_parameters, verification_files)
    else:
        print('Mode - '+str(args.mode)+' - is not valid.')
        print('Please choose from:\ntruth\nrwgt_truth\nreweight\nreco\n')
        print('Aborting')
        exit(1)


if __name__ == '__main__': main()
