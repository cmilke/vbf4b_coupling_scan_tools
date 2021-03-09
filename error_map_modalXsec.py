import sys
import math
import argparse
import numpy
import sympy
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt

#import pdb

import combination_utils
import reweight_utils

_scan_terms = combination_utils.full_scan_terms

def nice_array(arr):
    string = ''
    for a in arr: string += f'{int(a): 3d} '
    return string


def generate_error_maps(basis_parameters, basis_files):
    reweight_vector = combination_utils.get_amplitude_function(basis_parameters, as_scalar=False, base_equations=_scan_terms)
    var_edges = numpy.linspace(250, 2000, 101)

    base_events_list = [ reweight_utils.extract_ntuple_events(b,key='m_hh',filter_vbf=False) for b in basis_files ]
    base_histograms = [ reweight_utils.retrieve_reco_weights(var_edges, base_events) for base_events in base_events_list ]
    base_weights, base_errors = numpy.array(list(zip(*base_histograms)))

    kv_val = 1.0
    k2v_val_range = numpy.linspace(-2,4,51)
    kl_val_range = numpy.linspace(-15,15,51)
    distribution_sum_grid = numpy.zeros( (len(k2v_val_range),len(kl_val_range)) )
    distribution_mode_grid = numpy.zeros( (len(k2v_val_range),len(kl_val_range)) )
    relative_error_grid = numpy.zeros( (len(k2v_val_range),len(kl_val_range)) )
    averaging_window = int( 0.1*(len(var_edges)-1) )
    for k2v_i, k2v_val in enumerate(k2v_val_range):
        for kl_j, kl_val in enumerate(kl_val_range):
            coupling_parameters = (k2v_val, kl_val, kv_val)
            combined_weights, combined_errors = reweight_utils.reco_reweight(reweight_vector, coupling_parameters, base_weights, base_errors)
            rolling_average_weights = numpy.convolve( combined_weights, numpy.ones(averaging_window)/averaging_window, mode='valid' )
            distribution_mode_index = numpy.argmax(rolling_average_weights) + int(averaging_window/2) + 1
            distribution_mode_grid[k2v_i][kl_j] = var_edges[distribution_mode_index]
            distribution_sum_grid[k2v_i][kl_j] = numpy.sum(combined_weights)

            # Calculate relative error, with safety checks to account for empty bins
            combined_weights[ combined_weights == 0 ] = float('inf')
            relative_error_list = abs(combined_errors / combined_weights)
            relative_error_list[ combined_weights == float('inf') ] = 100
            error_start_index = distribution_mode_index-int(averaging_window/2)
            error_stop_index  = distribution_mode_index+int(averaging_window/2)+1
            error_slice = slice(error_start_index,error_stop_index)
            relative_error_grid[k2v_i][kl_j] = numpy.average(relative_error_list[error_slice])

    fig, ax_array = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
        #gridspec_kw={'height_ratios':grid_ratios,'width_ratios':grid_ratios})

    ranges = k2v_val_range[0], k2v_val_range[-1], kl_val_range[0], kl_val_range[-1]
    vartitle = '$m_{HH}$'
    plottable_couplings = [ [[],[],m] for m in ['v','o','^'] ]
    for couplings in basis_parameters:
        k2v, kl, kv = [float(sympy.Rational(c)) for c in couplings]
        index = 1 if kv == 1. else (0 if kv < 1. else 2)
        plottable_couplings[index][0].append(k2v)
        plottable_couplings[index][1].append(kl)

    ax_array[0][1].set_axis_off()

    im_sum = ax_array[0][0].imshow(distribution_sum_grid.transpose(), extent=ranges, origin='lower', cmap='inferno', norm=matplotlib.colors.LogNorm(0.01,3))
    ax_array[0][0].set_aspect('auto','box')
    ax_array[0][0].set_title('X-Sec at Coupling Point', fontsize='small' )
    ax_array[0][0].grid()
    ax_array[0][0].set_xticks(ticks = range(-2,5))
    ax_array[0][0].set_yticks(ticks = list(range(-14,21,5)))

    im_mode = ax_array[1][0].imshow(distribution_mode_grid.transpose(), extent=ranges, origin='lower', cmap='viridis')
    ax_array[1][0].set_aspect('auto','box')
    ax_array[1][0].set_title('Location of Distribution Mode', fontsize='small' )
    ax_array[1][0].grid()
    #for (x,y,m) in plottable_couplings: ax_array[1][0].scatter(x,y, marker=m, color='red', s=9)

    im_err = ax_array[1][1].imshow(relative_error_grid.transpose(), vmin=0, vmax=1, extent=ranges, origin='lower', cmap='plasma')
    ax_array[1][1].set_aspect('auto','box')
    ax_array[1][1].set_title('Relative Error at Distribution Mode', fontsize='small' )
    ax_array[1][1].grid()
    for (x,y,m) in plottable_couplings: ax_array[1][1].scatter(x,y, marker=m, color='cyan', s=9)

    # Create heatmap references
    #fig.subplots_adjust(top=0.70)

    #cbar_sum_ax = fig.add_axes([0.55, 0.85, 0.35, 0.03])
    cbar_sum_ax = fig.add_axes([0.55, 0.52, 0.02, 0.35])
    fig.colorbar(im_sum, cax=cbar_sum_ax, label='$\sigma$ (fb$^-1$)')

    #cbar_mode_ax = fig.add_axes([0.55, 0.7, 0.35, 0.03])
    cbar_mode_ax = fig.add_axes([0.70, 0.52, 0.02, 0.35])
    fig.colorbar(im_mode, cax=cbar_mode_ax, label='Distribution Mode (GeV)')

    #cbar_error_ax = fig.add_axes([0.55, 0.6, 0.35, 0.03])
    cbar_error_ax = fig.add_axes([0.85, 0.52, 0.02, 0.35])
    fig.colorbar(im_err, cax=cbar_error_ax, label='Average Relative Error')

    fig.text(0.5, 0.04, '$\kappa_{2V}$', ha='center')
    fig.text(0.04, 0.5, '$\kappa_{\lambda}$', va='center', rotation='vertical')
    fig.suptitle('Linearly Combined $M_{HH}$ Cross-sections\nWith Distribution Modes and Associated Relative Error', fontsize=10, fontweight='bold')
    #plt.show()
    dpi = 500
    plt.savefig('plots/error_maps/basic_modal_w-xsec.png',dpi=dpi)
    #output.savefig()
    #plt.close()




def main():
    # Sort out command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( "--basis", required = False, default = 'basis_files/nnt_basis.dat', type=str,
        help = "File to provide basis states",)

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

    #pdb.set_trace()
    generate_error_maps(basis_parameters, basis_files)


if __name__ == '__main__': main()
