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


def generate_error_maps(basis_parameters, basis_files):
    reweight_vector = combination_utils.get_amplitude_function(basis_parameters, as_scalar=False, base_equations=_scan_terms)
    var_edges = numpy.linspace(250, 2000, 101)

    base_events_list = [ reweight_utils.extract_ntuple_events(b,key='m_hh',filter_vbf=False) for b in basis_files ]
    base_histograms = [ reweight_utils.retrieve_reco_weights(var_edges, base_events) for base_events in base_events_list ]
    base_weights, base_errors = numpy.array(list(zip(*base_histograms)))

    kv_val = 1.0
    k2v_val_range = numpy.linspace(-2,4,51)
    kl_val_range = numpy.linspace(-15,15,51)
    varlen = len(var_edges)
    var_splits = [0, 0.1, 0.2, 0.5, 1]
    var_slices = [ slice(int(start*varlen),int(end*varlen)+1) for start,end in zip(var_splits,var_splits[1:])]
    relative_error_grids = [ numpy.zeros( (len(k2v_val_range),len(kl_val_range)) ) for v in var_slices ]
    for k2v_i, k2v_val in enumerate(k2v_val_range):
        for kl_j, kl_val in enumerate(kl_val_range):
            coupling_parameters = (k2v_val, kl_val, kv_val)
            combined_weights, combined_errors = reweight_utils.reco_reweight(reweight_vector, coupling_parameters, base_weights, base_errors)
            combined_weights[ combined_weights == 0 ] = float('inf')
            relative_error_list = abs(combined_errors / combined_weights)
            relative_error_list[ combined_weights == float('inf') ] = 100
            for i,s in enumerate(var_slices):
                relative_error_grids[i][k2v_i][kl_j] = numpy.average(relative_error_list[s])
                #relative_error_grids[i][k2v_i][kl_j] = numpy.average(combined_weights)

    # Work out how to arrange mini-plots
    num_plots = len(var_slices)
    num_columns = math.ceil( math.sqrt(num_plots) )
    num_rows = math.ceil(num_plots/num_columns)
    fig, ax_array = plt.subplots(nrows=num_rows, ncols=num_columns, sharex=True, sharey=True)
        #gridspec_kw={'height_ratios':grid_ratios,'width_ratios':grid_ratios})

    # Get rid of excess plots (e.g. if you have 7 slices,
    # you need a 3x3 grid, which means 9 plots, 2 of which are not used)
    num_axes_to_turn_off = num_rows*num_columns - num_plots
    for index in range(num_axes_to_turn_off):
        plot_num = num_rows*num_columns - index - 1
        i,j = numpy.unravel_index( plot_num, (num_rows, num_columns) )
        if num_rows == 1: sub_ax = ax_array[j]
        else: sub_ax = ax_array[i][j]
        sub_ax.set_axis_off()

    ranges = k2v_val_range[0], k2v_val_range[-1], kl_val_range[0], kl_val_range[-1]
    vartitle = '$m_{HH}$'
    plottable_couplings = [ [[],[],m] for m in ['v','o','^'] ]
    for couplings in basis_parameters:
        k2v, kl, kv = [float(sympy.Rational(c)) for c in couplings]
        index = 1 if kv == 1. else (0 if kv < 1. else 2)
        plottable_couplings[index][0].append(k2v)
        plottable_couplings[index][1].append(kl)

    for var_i, grid in enumerate(relative_error_grids):
        i,j = numpy.unravel_index( var_i, (num_rows, num_columns) )
        sub_ax = ax_array[i][j]
        #im = sub_ax.imshow(relative_error_grids[var_i].transpose(), vmin=0, vmax=10, extent=ranges, origin='lower')
        im = sub_ax.imshow(relative_error_grids[var_i].transpose(), norm=matplotlib.colors.LogNorm(0.01,100), extent=ranges, origin='lower')
        #im = sub_ax.imshow(relative_error_grids[var_i].transpose(), vmin=-0.01, vmax=0.04, extent=ranges, origin='lower')
        for (x,y,m) in plottable_couplings: sub_ax.scatter(x,y, marker=m, color='red', s=9)
        sub_ax.set_aspect('auto','box')
        var_range = var_edges[var_slices[var_i]]
        sub_ax.set_title(vartitle+f' = {var_range[0]} - {var_range[-1]}', fontsize='small' )

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Average |Relative Error|')
    fig.text(0.5, 0.04, '$\kappa_{2V}$', ha='center')
    fig.text(0.04, 0.5, '$\kappa_{\lambda}$', va='center', rotation='vertical')
    fig.suptitle('Relative Error of Linearly Combined Distributions,\nAveraged Over Various Regions', fontsize=10, fontweight='bold')
    #plt.show()
    dpi = 500
    plt.savefig('plots/error_maps/basic.png',dpi=dpi)
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
