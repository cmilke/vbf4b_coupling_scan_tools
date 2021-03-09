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
    distribution_mode_grid = numpy.zeros( (len(k2v_val_range),len(kl_val_range)) )
    relative_error_grid = numpy.zeros( (len(k2v_val_range),len(kl_val_range)) )
    averaging_window = int( 0.1*(len(var_edges)-1) )
    for k2v_i, k2v_val in enumerate(k2v_val_range):
        for kl_j, kl_val in enumerate(kl_val_range):
            coupling_parameters = (k2v_val, kl_val, kv_val)
            combined_weights, combined_errors = reweight_utils.reco_reweight(reweight_vector, coupling_parameters, base_weights, base_errors)
            rolling_average_weights = numpy.convolve( combined_weights, numpy.ones(averaging_window)/averaging_window, mode='valid' )
            distribution_mode_index = numpy.argmax(rolling_average_weights) + int(averaging_window/2) + 1
            distribution_mode_grid[k2v_i][kl_j] = numpy.average(var_edges[distribution_mode_index])
            #print(nice_array(combined_weights*100))
            #print(nice_array([.0001]*int(averaging_window/2+1) + list(rolling_average_weights*100)))
            #tmp = numpy.zeros(len(combined_weights))
            #tmp[distribution_mode_index] = 1
            #print(nice_array(tmp))
            #print()
            #continue
            combined_weights[ combined_weights == 0 ] = float('inf')
            relative_error_list = abs(combined_errors / combined_weights)
            relative_error_list[ combined_weights == float('inf') ] = 100
            error_start_index = distribution_mode_index-int(averaging_window/2)
            error_stop_index  = distribution_mode_index+int(averaging_window/2)+1
            error_slice = slice(error_start_index,error_stop_index)
            relative_error_grid[k2v_i][kl_j] = numpy.average(relative_error_list[error_slice])

    fig, ax_array = plt.subplots(ncols=2, sharex=True, sharey=True)
        #gridspec_kw={'height_ratios':grid_ratios,'width_ratios':grid_ratios})

    ranges = k2v_val_range[0], k2v_val_range[-1], kl_val_range[0], kl_val_range[-1]
    vartitle = '$m_{HH}$'
    plottable_couplings = [ [[],[],m] for m in ['v','o','^'] ]
    for couplings in basis_parameters:
        k2v, kl, kv = [float(sympy.Rational(c)) for c in couplings]
        index = 1 if kv == 1. else (0 if kv < 1. else 2)
        plottable_couplings[index][0].append(k2v)
        plottable_couplings[index][1].append(kl)

    #im0 = ax_array[0].imshow(distribution_mode_grid.transpose(), vmin=250, vmax=2000, extent=ranges, origin='lower')
    im0 = ax_array[0].imshow(distribution_mode_grid.transpose(), extent=ranges, origin='lower')
    ax_array[0].set_aspect('auto','box')
    for (x,y,m) in plottable_couplings: ax_array[0].scatter(x,y, marker=m, color='red', s=9)

    im1 = ax_array[1].imshow(relative_error_grid.transpose(), vmin=0, vmax=1, extent=ranges, origin='lower', cmap='plasma')
    ax_array[1].set_aspect('auto','box')
    for (x,y,m) in plottable_couplings: ax_array[1].scatter(x,y, marker=m, color='cyan', s=9)

    # Create heatmap references
    fig.subplots_adjust(top=0.70)

    cbar_mode_ax = fig.add_axes([0.125, 0.8, 0.35, 0.05])
    fig.colorbar(im0, cax=cbar_mode_ax, orientation='horizontal', label='X-Sec Distribution Mode (GeV)')

    cbar_error_ax = fig.add_axes([0.55, 0.8, 0.35, 0.05])
    fig.colorbar(im1, cax=cbar_error_ax, orientation='horizontal', label='Average Relative Error at Mode')

    fig.text(0.5, 0.04, '$\kappa_{2V}$', ha='center')
    fig.text(0.04, 0.5, '$\kappa_{\lambda}$', va='center', rotation='vertical')
    fig.suptitle('Linearly Combined $M_{HH}$ Distribution Modes\nand Associated Relative Error', fontsize=10, fontweight='bold')
    #plt.show()
    dpi = 500
    plt.savefig('plots/error_maps/basic_modal.png',dpi=dpi)
    #output.savefig()
    #plt.close()



    #relative_error_points = [[], [], [], []]
    #for k2v_i, k2v_val in enumerate(k2v_val_range):
    #    for kl_j, kl_val in enumerate(kl_val_range):
    #        coupling_parameters = (k2v_val, kl_val, kv_val)
    #        combined_weights, combined_errors = reweight_utils.reco_reweight(reweight_vector, coupling_parameters, base_weights, base_errors)
    #        combined_weights[ combined_weights == 0 ] = float('inf')
    #        relative_error_list = combined_errors / combined_weights
    #        #relative_error_grid[k2v_i][kl_j] = relative_errors
    #        for bin_k, relative_error in enumerate(relative_error_list):
    #            print(k2v_val,kl_val,var_edges[bin_k],combined_weights[bin_k], combined_errors[bin_k], relative_error)
    #            relative_error_points[0].append(k2v_val)
    #            relative_error_points[1].append(kl_val)
    #            relative_error_points[2].append(var_edges[bin_k])
    #            relative_error_points[3].append(abs(relative_error))

    #for vals in zip(*relative_error_points): print(vals)
    #k2v,kl,mhh,error = relative_error_points
    #fig = plt.figure()
    #ax = fig.add_subplot(111,projection='3d')
    #plot = ax.scatter(k2v,kl,mhh,c=error, marker='.', vmin=0, vmax=1, cmap='Blues', alpha=0.5)
    #fig.colorbar(plot, ax=ax)
    #plt.show()





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
