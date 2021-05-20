import sys
import argparse
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#import pdb

from fileio_utils import read_coupling_file, get_events, retrieve_reco_weights
from combination_utils import get_amplitude_function
from combination_utils import basis_full3D_2021May_minN as _reco_basis 
#from combination_utils import basis_full3D_old_minN as _reco_basis 
#from combination_utils import basis_full3D_max as _reco_basis 
from reweight_utils import reco_reweight


def plot_histogram(hist_name, hist_title, edge_list, coupling_parameters,
        linearly_combined_weights, linearly_combined_errors,
        range_specs=None, xlabel='Truth $m_{HH}$ (GeV)', normalize=False,
        generated_label='Linear Combination', generated_color='blue'):

    print('Plotting '+hist_name+' for ' + str(coupling_parameters))
    fig, ax = plt.subplots()

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

    kappa_labels = [ str(param) for param in coupling_parameters ]
    title  = hist_title+' for '
    title += '$\kappa_{2V}='+kappa_labels[0]+'$, '
    title += '$\kappa_{\lambda}='+kappa_labels[1]+'$, '
    title += '$\kappa_{V}='+kappa_labels[2]+'$'
    fig.suptitle(title)

    ax.set(xlabel=xlabel)
    ax.set(ylabel='Bin Weight')
    ax.grid()

    dpi=500
    kappa_string_list = [ label.replace('.','p') for label in kappa_labels ]
    kappa_string = 'cvv'+kappa_string_list[0]+'cl'+kappa_string_list[1]+'cv'+kappa_string_list[2]
    fig.tight_layout()
    fig.savefig('plots/previews/'+hist_name+'_'+kappa_string+'.png', dpi=dpi)
    plt.close()



def view_reco_method(basis_parameters, view_params):
    reweight_vector = get_amplitude_function(basis_parameters, as_scalar=False)
    var_edges = numpy.linspace(200, 1200, 31)

    data_files = read_coupling_file('basis_files/nnt_coupling_file_2021May.dat')
    base_events_list = get_events(basis_parameters, data_files)
    base_histograms = [ retrieve_reco_weights(var_edges, base_events) for base_events in base_events_list ]
    base_weights, base_errors = numpy.array(list(zip(*base_histograms)))

    for coupling_parameters in view_params:
        print(coupling_parameters)
        combined_weights, combined_errors = reco_reweight(reweight_vector, coupling_parameters, base_weights, base_errors)

        plot_histogram('preview_reco_mHH_new', 'New NNT-Based Linear Combination:\n$m_{HH}$', var_edges, coupling_parameters,
                 combined_weights, combined_errors,
                 xlabel='Reconstructed $m_{HH}$ (GeV)'
        )



def main():
    # Sort out command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( "--mode", required = False, default = 'reco', type=str,
        help = "Options are: 'truth' or 'reco'",) 

    args = parser.parse_args()

    view_parameters = [ #k2v, kl, kv
        ( 1    ,  1   , 1 ),
        ( 2    ,  1   , 1 ),
        ( 1    ,  2   , 1 ),
        ( 1    ,  3   , 1 ),
        ( 0    ,  1   , 1 ),
        ( 0.5  ,  1   , 1 ),
        ( 4    ,  1   , 1 ),
        ( 1    ,  2   , 1 ),
        ( 1    ,  10  , 1 ),
        ( 2    ,  10  , 1 ),
        ( 0.5  ,  13  , 1 ),
        ( 1.5  ,  -9  , 1 ),
        ( 0    ,  -9  , 1 )
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
