import sys
import argparse
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#import pdb

from fileio_utils import read_coupling_file, get_events, retrieve_reco_weights
from  combination_utils import get_amplitude_function
from reweight_utils import reco_reweight

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
    #safe_error = linearly_combined_errors.copy()
    safe_error[ safe_error == 0 ] = float('inf')
    safe_combination = linearly_combined_weights.copy()
    safe_combination[ safe_combination == 0 ] = float('inf')
    rcounts, rbins, rpoints = ax_ratio.errorbar( xpositions, (linearly_combined_weights-verification_weights)/safe_error,
        yerr=linearly_combined_errors/safe_error, label='MC Statistical Error check',
        marker='+', markersize=2, capsize=2, color=generated_color, linestyle='none', linewidth=1, zorder=3)
    ax_ratio.set(ylabel=r'$\frac{lin. comb. - gen.}{stat. error}$', xlabel=xlabel)
    #ax_ratio.set(ylabel=r'$\frac{lin. comb. - gen.}{comb. error}$', xlabel=xlabel)
    ax_ratio.set_ylim([-3,3])
    #ax_ratio.set_yticks(ticks=range(-3,4))
    ax_ratio.set_yticks(ticks=[-2,-1,0,1,2])
    zero_line = ax_ratio.hlines(0,xmin=edge_list[0],xmax=edge_list[-1],colors='black',zorder=2)

    #rcounts, rbins, rpoints = ax_ratio.errorbar( xpositions, linearly_combined_weights/verification_weights,
    #    yerr=linearly_combined_errors/verification_weights, label='MC Statistical Error check',
    #    marker='+', markersize=2, capsize=2, color=generated_color, linestyle='none', linewidth=1, zorder=3)
    #ax_ratio.set(ylabel=r'$\frac{generated}{lin. combination}$', xlabel=xlabel)
    #ax_ratio.set_ylim([0.5,1.5])
    #ax_ratio.set_yticks(ticks=[0.5,1,1.5])
    #zero_line = ax_ratio.hlines(1,xmin=edge_list[0],xmax=edge_list[-1],colors='black',zorder=2)

    kappa_labels = [ str(param) for param in coupling_parameters ]
    title  = hist_title+' for '
    title += '$\kappa_{2V}='+kappa_labels[0]+'$, '
    title += '$\kappa_{\lambda}='+kappa_labels[1]+'$, '
    title += '$\kappa_{V}='+kappa_labels[2]+'$'
    fig.suptitle(title)

    ax_main.set(ylabel='Bin Weight')
    ax_main.legend(prop={'size':7})
    ax_main.grid()
    ax_ratio.grid()

    dpi=500
    kappa_string_list = [ label.replace('.','p') for label in kappa_labels ]
    kappa_string = 'cvv'+kappa_string_list[0]+'cl'+kappa_string_list[1]+'cv'+kappa_string_list[2]
    fig.tight_layout()
    fig.savefig('plots/validation/'+hist_name+'_'+kappa_string+'.png', dpi=dpi)
    plt.close()




def validate_truth_combinations(basis_parameters, basis_files, verification_parameters, verification_files):
    edge_list = numpy.arange(0, 2050, 50)
    basis_weight_list, basis_error_list = reweight_utils.extract_lhe_truth_data(basis_files, edge_list)
    #basis_weight_list, basis_error_list, edge_list = reweight_utils.extract_truth_data(basis_files)
    amplitude_function = combination_utils.get_amplitude_function(basis_parameters, base_equations=_scan_terms)
    vector_function = combination_utils.get_amplitude_function(basis_parameters, base_equations=_scan_terms, as_scalar=False)
    combination_function = lambda params: amplitude_function(*params, *basis_weight_list)
    coefficient_function = lambda params: vector_function(*params)[0]

    verification_weight_list, verification_error_list = reweight_utils.extract_lhe_truth_data(verification_files, edge_list)
    for index, coupling_parameters in enumerate(verification_parameters):
        combined_weights, combined_errors = reweight_utils.reco_reweight(vector_function, coupling_parameters, basis_weight_list, basis_error_list)

        plot_histogram('truth_comb', 'Full-Truth Linear Combination:\n$m_{HH}$', edge_list, coupling_parameters,
            combined_weights, combined_errors,
            verification_weight_list[index], verification_error_list[index])





def validate_reco_method(basis_parameters, verification_parameters):
    reweight_vector = get_amplitude_function(basis_parameters, as_scalar=False)
    var_edges = numpy.arange(0, 2050, 50)

    data_files = read_coupling_file('basis_files/nnt_coupling_file.dat')
    base_events_list = get_events(basis_parameters, data_files)
    verification_events_list = get_events(verification_parameters, data_files)

    base_histograms = [ retrieve_reco_weights(var_edges, base_events) for base_events in base_events_list ]
    base_weights, base_errors = numpy.array(list(zip(*base_histograms)))

    for verification_events, coupling_parameters in zip(verification_events_list, verification_parameters):
        verification_weights, verification_errors = retrieve_reco_weights(var_edges, verification_events)
        combined_weights, combined_errors = reco_reweight(reweight_vector, coupling_parameters, base_weights, base_errors)

        plot_histogram('reco_mHH', 'NNT-Based Linear Combination:\n$m_{HH}$', var_edges, coupling_parameters,
                 combined_weights, combined_errors,
                 verification_weights, verification_errors,
                 xlabel='Reconstructed $m_{HH}$ (GeV)'
        )



def main():
    # Sort out command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( "--mode", required = False, default = 'reco', type=str,
        help = "Options are: 'truth' or 'reco'",) 

    args = parser.parse_args()

    verification_parameters = [ #k2v, kl, kv
        ( 1    ,  1   , 1 ),
        ( 2    ,  1   , 1 ),
        ( 0    ,  1   , 1 ),
        ( 0.5  ,  1   , 1 ),
        ( 4    ,  1   , 1 ),
        ( 1    ,  2   , 1 ),
        ( 1    ,  10  , 1 )
    ]


    #pdb.set_trace()
    if args.mode == 'truth':
        validate_truth_combinations(basis_parameters, basis_files, verification_parameters, verification_files)
    elif args.mode == 'reco':
        basis_parameters = [ (1, 1, 1), (2, 1, 1), (0.5, 1, 1), (0, 1, 0.5), (1, 0, 1), (1, 10, 1) ]
        validate_reco_method(basis_parameters, verification_parameters)
    else:
        print('Mode - '+str(args.mode)+' - is not valid.')
        print('Please choose from:\ntruth\nrwgt_truth\nreweight\nreco\n')
        print('Aborting')
        exit(1)


if __name__ == '__main__': main()
