import sys
import argparse
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#import pdb

import fileio_utils
from combination_utils import get_amplitude_function
from combination_utils import basis_full3D_2021May_minN as _reco_basis 
#from combination_utils import basis_full3D_old_minN as _reco_basis 
#from combination_utils import basis_full3D_max as _reco_basis 
from reweight_utils import reco_reweight


def plot_histogram(hist_name, hist_title, edge_list, coupling_parameters,
        linearly_combined_weights, linearly_combined_errors,
        alt_linearly_combined_weights=None, alt_linearly_combined_errors=None, alt_label='Alt',
        range_specs=None, xlabel='Truth $m_{HH}$ (GeV)', normalize=False,
        generated_label='Linear Combination', generated_color='blue'):

    print('Plotting '+hist_name+' for ' + str(coupling_parameters))
    fig, ax = plt.subplots()
    draw_alt_comb = type(alt_linearly_combined_weights) != type(None) and type(alt_linearly_combined_errors) != type(None)

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
    counts, bins, points = ax.errorbar( xpositions, linearly_combined_weights,
        yerr=linearly_combined_errors, label=generated_label,
        marker='+', markersize=2, capsize=2, color=generated_color, linestyle='none', linewidth=1, zorder=3)

    if draw_alt_comb:
        alt_counts, alt_bins, alt_points = ax.errorbar( xpositions, alt_linearly_combined_weights,
            yerr=alt_linearly_combined_errors, label=alt_label,
            marker='x', markersize=2, capsize=2, color='red', linestyle='none', linewidth=1, zorder=3)


    kappa_labels = [ f'{param:.2f}' for param in coupling_parameters ]
    title  = hist_title+' for '
    title += '$\kappa_{2V}='+kappa_labels[0]+'$, '
    title += '$\kappa_{\lambda}='+kappa_labels[1]+'$, '
    title += '$\kappa_{V}='+kappa_labels[2]+'$'
    fig.suptitle(title)

    ax.set(xlabel=xlabel)
    ax.set(ylabel='Bin Weight')
    ax.legend(prop={'size':7})
    ax.grid()

    dpi=500
    kappa_string_list = [ label.replace('.','p') for label in kappa_labels ]
    kappa_string = 'cvv'+kappa_string_list[0]+'cl'+kappa_string_list[1]+'cv'+kappa_string_list[2]
    fig.tight_layout()
    figname = hist_name+'_'+kappa_string
    #fig.savefig('plots/previews/'+figname+'.png', dpi=dpi)
    fig.savefig('plots/previews/'+figname+'.pdf', dpi=dpi)
    plt.close()



def view_reco_method(basis_parameters, view_params):
    reweight_vector = get_amplitude_function(basis_parameters, as_scalar=False)
    var_edges = numpy.linspace(200, 1200, 31)

    data_files = fileio_utils.read_coupling_file()
    base_events_list = fileio_utils.get_events(basis_parameters, data_files)
    base_histograms = [ fileio_utils.retrieve_reco_weights(var_edges, base_events) for base_events in base_events_list ]
    base_weights, base_errors = numpy.array(list(zip(*base_histograms)))

    for coupling_parameters in view_params:
        print(coupling_parameters)
        combined_weights, combined_errors = reco_reweight(reweight_vector, coupling_parameters, base_weights, base_errors)

        plot_histogram('preview_reco_mHH_new', 'NNT-Based Linear Combination:\n$m_{HH}$', var_edges, coupling_parameters,
                 combined_weights, combined_errors,
                 xlabel='Reconstructed $m_{HH}$ (GeV)'
        )



def main():
    # Sort out command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( "--mode", required = False, default = 'reco', type=str,
        help = "Options are: 'truth' or 'reco'",) 

    args = parser.parse_args()

    #view_parameters = [ #k2v, kl, kv
    #    ( 1    ,  -7   , 1 ),
    #]

    k2v_vals = [-1.5, 0.5, 2, 3.5]
    kl_vals = [-9, -3, 5, 14]
    view_parameters = []
    for k2v in k2v_vals:
        for kl in kl_vals:
            view_parameters.append( (k2v, kl, 1) )
    view_parameters += [
        (1,-7,1),
        (4,1,1)
    ]


    #pdb.set_trace()
    if args.mode == 'reco':
        view_reco_method(_reco_basis, view_parameters)
    else:
        print('Mode - '+str(args.mode)+' - is not valid.')
        print('Please choose from:\ntruth\nrwgt_truth\nreweight\nreco\n')
        print('Aborting')
        exit(1)


if __name__ == '__main__': main()
