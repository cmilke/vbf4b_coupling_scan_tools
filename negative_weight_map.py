import sys
import math
import argparse
import numpy
import sympy
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt

#import pdb

import fileio_utils
import combination_utils
from combination_utils import basis_full3D_2021May_minN as _reco_basis 
from reweight_utils import reco_reweight


def get_Nweight_sum1D(couplings, weights, k2v_vals, kl_vals, kv_vals, vector=False, base_equations=None, which_coupling=None):
    #numpy.set_printoptions(threshold=sys.maxsize, linewidth=230, precision=0, floatmode='fixed', suppress=True)
    reweight_vector_function = combination_utils.get_amplitude_function(couplings, as_scalar=False, base_equations=base_equations)
    multiplier_array_vector = reweight_vector_function(k2v_vals, kl_vals, kv_vals)[0]
    combined_weights = sum([ multiplier_array[...,None] * w for multiplier_array, w in zip(multiplier_array_vector, weights) ])
    negative_boolean_vector = combined_weights < 0
    if vector:
        print(negative_boolean_vector)
        nWeight_totals = negative_boolean_vector.sum(axis=1)
        print(nWeight_totals)
        exit()
        return nWeight_totals
    else:
        if which_coupling == 'k2v':
            delta_variation = k2v_vals[1] - k2v_vals[0]
        else:
            delta_variation  = kl_vals[1] - kl_vals[0]
        nWeight_integral = negative_boolean_vector.sum() * delta_variation
        return nWeight_integral


def get_Nweight_sum(couplings, weights, kv_val, k2v_val_range, kl_val_range, grid=False, mask=None):
    #numpy.set_printoptions(threshold=sys.maxsize, linewidth=230, precision=0, floatmode='fixed', suppress=True)
    reweight_vector_function = combination_utils.get_amplitude_function(couplings, as_scalar=False, base_equations=combination_utils.full_scan_terms)
    k2v_grid, kl_grid = numpy.meshgrid(k2v_val_range, kl_val_range)
    multiplier_grid_vector = reweight_vector_function(k2v_grid, kl_grid, kv_val)[0]
    combined_weights = sum([ multiplier_grid[...,None] * w for multiplier_grid, w in zip(multiplier_grid_vector, weights) ])
    negative_boolean_grid = combined_weights < 0
    if type(mask) != type(None):
        mask_grid = mask(k2v_grid, kl_grid)
        negative_boolean_grid = negative_boolean_grid * mask_grid[...,None]

    if grid:
        nWeight_totals = negative_boolean_grid.sum(axis=2)
        return nWeight_totals
    else:
        grid_pixel_area = (k2v_val_range[1] - k2v_val_range[0]) * (kl_val_range[1] - kl_val_range[0])
        nWeight_integral = negative_boolean_grid.sum() * grid_pixel_area
        return nWeight_integral


def draw_error_map(basis_parameters, var_edges, kv_val, k2v_val_range, kl_val_range, negative_weight_grid,
            name_suffix='', title_suffix=None, vmin=0, vmax=None):
    num_bins = len(var_edges) - 1
    ranges = k2v_val_range[0], k2v_val_range[-1], kl_val_range[0], kl_val_range[-1]
    vartitle = '$m_{HH}$'
    plottable_couplings = [ [[],[],m] for m in ['v','o','^'] ]
    for couplings in basis_parameters:
        k2v, kl, kv = [float(sympy.Rational(c)) for c in couplings]
        index = 1 if kv == 1. else (0 if kv < 1. else 2)
        plottable_couplings[index][0].append(k2v)
        plottable_couplings[index][1].append(kl)

    fig, ax = plt.subplots()
    #im = ax.imshow(negative_weight_grid.transpose(), vmin=vmin, vmax=vmax, extent=ranges, origin='lower', cmap='plasma')
    im = ax.imshow(negative_weight_grid, vmin=vmin, vmax=vmax, extent=ranges, origin='lower', cmap='plasma')
    ax.set_xticks(ticks = numpy.arange(ranges[0],ranges[1]+1,1))
    ax.set_yticks(ticks = numpy.linspace(ranges[2],ranges[3],7))
    ax.set_xlabel('$\kappa_{2V}$')
    ax.set_ylabel('$\kappa_{\lambda}$')
    ax.grid()
    for (x,y,m) in plottable_couplings: ax.scatter(x,y, marker=m, color='cyan', s=9)
    ax.set_aspect('auto','box')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.11, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Num Bins (Out of '+str(num_bins)+' Total)')
    #fig.text(0.5, 0.04, '$\kappa_{2V}$', ha='center')
    #fig.text(0.04, 0.5, '$\kappa_{\lambda}$', va='center', rotation='vertical')

    basis_table = '$\kappa_{2V}$  ,  $\kappa_{\lambda}$  ,  $\kappa_{V}$   '
    for coupling in basis_parameters: basis_table += '\n'+combination_utils.nice_coupling_string(coupling)
    fig.text(.99, 1, basis_table, ha='right', va='top', fontsize='xx-small', family='monospace')

    range_title = f'{var_edges[0]:.0f}-{var_edges[-1]:.0f} GeV in Bins of {var_edges[1]-var_edges[0]:.2f} GeV'
    title  = 'Number of Bins in $m_{HH}$ with Negative Weights,\nOver $m_{HH}$ Range '+range_title
    if type(title_suffix) != type(None): title += '\n'+title_suffix
    fig.suptitle(title, fontsize=10, fontweight='bold')
    #plt.show()
    dpi = 500
    figname = 'negative_weights'+name_suffix
    #plt.savefig('plots/error_maps/'+figname+'.png',dpi=dpi)
    plt.savefig('plots/error_maps/'+figname+'.pdf',dpi=dpi)
    #output.savefig()
    #plt.close()




def single_negative_weight_map(basis_parameters, name_suffix='_base', truth_level=False):
    var_edges = numpy.linspace(200, 1200, 31)
    kv_val = 1.0
    num_kappa_bins = 100
    k2v_val_range = numpy.linspace(-2,4,num_kappa_bins+1)
    kl_val_range = numpy.linspace(-14,16,num_kappa_bins+1)

    if truth_level:
        truth_data_files = fileio_utils.read_coupling_file(coupling_file='basis_files/truth_LHE_couplings.dat')
        basis_files = [ truth_data_files[coupling] for coupling in basis_parameters ]
        base_weights, base_errors = fileio_utils.extract_lhe_truth_data(basis_files, var_edges)
    else:
        data_files = fileio_utils.read_coupling_file()
        base_events_list = fileio_utils.get_events(basis_parameters, data_files)
        base_histograms = [ fileio_utils.retrieve_reco_weights(var_edges, base_events) for base_events in base_events_list ]
        base_weights, base_errors = numpy.array(list(zip(*base_histograms)))

    integral = get_Nweight_sum(basis_parameters, base_weights, kv_val, k2v_val_range, kl_val_range, grid=False)
    negative_weight_grid = get_Nweight_sum(basis_parameters, base_weights, kv_val, k2v_val_range, kl_val_range, grid=True)
    draw_error_map(basis_parameters, var_edges, kv_val, k2v_val_range, kl_val_range, negative_weight_grid, name_suffix=name_suffix,
                title_suffix=f'Integral={int(integral)}')


def multislice_negative_weight_map(basis_parameters, name_suffix='_base', truth_level=False):
    var_edges = numpy.linspace(200, 1200, 31)
    num_kappa_bins = 100
    k2v_val_range = numpy.linspace(-2,4,num_kappa_bins+1)
    kl_val_range = numpy.linspace(-14,16,num_kappa_bins+1)

    if truth_level:
        truth_data_files = fileio_utils.read_coupling_file(coupling_file='basis_files/truth_LHE_couplings.dat')
        basis_files = [ truth_data_files[coupling] for coupling in basis_parameters ]
        base_weights, base_errors = fileio_utils.extract_lhe_truth_data(basis_files, var_edges)
    else:
        data_files = fileio_utils.read_coupling_file('basis_files/nnt_coupling_file_2021Aug_test.dat')
        base_events_list = fileio_utils.get_events(basis_parameters, data_files)
        base_histograms = [ fileio_utils.retrieve_reco_weights(var_edges, base_events) for base_events in base_events_list ]
        base_weights, base_errors = numpy.array(list(zip(*base_histograms)))

    kv_val_range = numpy.arange(0.9, 1.125, 0.025)

    # Work out how to arrange mini-plots for slice plot
    num_slices = len(kv_val_range)
    num_columns = math.ceil(math.sqrt(num_slices))
    num_rows = math.ceil(num_slices / num_columns)
    fig, ax_array = plt.subplots( nrows=num_rows, ncols=num_columns, sharex=True, sharey=True)

    # Get rid of excess plots (e.g. if you have 7 slices,
    # you need a 3x3 grid, which means 9 plots, 2 of which are not used)
    num_axes_to_turn_off = num_rows * num_columns - num_slices
    for index in range(num_axes_to_turn_off):
        plot_num = num_rows * num_columns - index - 1
        i, j = numpy.unravel_index(plot_num, (num_rows, num_columns))
        if num_rows == 1: sub_ax = ax_array[j]
        else: sub_ax = ax_array[i][j]
        sub_ax.set_axis_off()

    # Make all 2D plots of mod1-mod2 across the range of the 3rd coupling (slicemod)
    # Storing mini-versions of the 2D plots for use in the final slice plot
    for slice_index, kv_val in enumerate(kv_val_range[::-1]):
        i, j = numpy.unravel_index(slice_index, (num_rows, num_columns))
        if num_slices == 1: sub_ax = None
        elif num_rows == 1: sub_ax = ax_array[j]
        else: sub_ax = ax_array[i][j]

        integral = get_Nweight_sum(basis_parameters, base_weights, kv_val, k2v_val_range, kl_val_range, grid=False)
        negative_weight_grid = get_Nweight_sum(basis_parameters, base_weights, kv_val, k2v_val_range, kl_val_range, grid=True)

        num_bins = len(var_edges) - 1
        ranges = k2v_val_range[0], k2v_val_range[-1], kl_val_range[0], kl_val_range[-1]
        vartitle = '$m_{HH}$'
        plottable_couplings = [ [[],[],m] for m in ['v','o','^'] ]
        for couplings in basis_parameters:
            k2v, kl, kv = [float(sympy.Rational(c)) for c in couplings]
            index = 1 if kv == kv_val else (0 if kv < kv_val else 2)
            plottable_couplings[index][0].append(k2v)
            plottable_couplings[index][1].append(kl)

        vmin, vmax = 0, 7
        im = sub_ax.imshow(negative_weight_grid, vmin=vmin, vmax=vmax, extent=ranges, origin='lower', cmap='plasma')

        if slice_index == 0:
            sub_ax.set_xticks(ticks = numpy.arange(ranges[0],ranges[1]+1,1))
            sub_ax.set_yticks(ticks = numpy.linspace(ranges[2],ranges[3],7))

        sub_ax.grid()
        for (x,y,m) in plottable_couplings: sub_ax.scatter(x,y, marker=m, color='cyan', s=9)
        sub_ax.set_aspect('auto','box')
        fig.text(0.5, 0.04, '$\kappa_{2V}$', ha='center')
        fig.text(0.04, 0.5, '$\kappa_{\lambda}$', va='center', rotation='vertical')

        if num_slices == 1: continue
        sub_ax.set_title( "$\kappa_{V}$ = "f"{kv_val:.2f}", fontsize="x-small")

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.11, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Num Bins (Out of '+str(num_bins)+' Total)')

    basis_table = '$\kappa_{2V}$  ,  $\kappa_{\lambda}$  ,  $\kappa_{V}$   '
    for coupling in basis_parameters: basis_table += '\n'+combination_utils.nice_coupling_string(coupling)
    fig.text(.99, 1, basis_table, ha='right', va='top', fontsize='xx-small', family='monospace')

    range_title = f'{var_edges[0]:.0f}-{var_edges[-1]:.0f} GeV in Bins of {var_edges[1]-var_edges[0]:.2f} GeV'
    title  = 'Number of Bins in $m_{HH}$ with Negative Weights,\nOver $m_{HH}$ Range '+range_title
    fig.suptitle(title, fontsize=10, fontweight='bold')

    dpi=500
    figname = 'negative_weights'+name_suffix
    fig.savefig('plots/error_maps/multislice_'+figname+'.pdf', dpi=dpi)




def main():
    # Sort out command-line arguments
#    parser = argparse.ArgumentParser()
#    parser.add_argument( "--basis", required = False, default = 'basis_files/nnt_basis.dat', type=str,
#        help = "File to provide basis states",)
#
#    args = parser.parse_args()

    multislice_negative_weight_map( combination_utils.basis_full3D_2021May_minN, truth_level=False, name_suffix='nominal' )
    multislice_negative_weight_map( combination_utils.basis_full3D_2021Aug_Neo, truth_level=False, name_suffix='neo' )


    #pdb.set_trace()
    #single_negative_weight_map(_reco_basis)
    #single_negative_weight_map( 
    #    [(1.0, 1.0, 1.0), (0.5, 1.0, 1.0), (3.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 10.0, 1.0), (0.0, 0.0, 1.0)],
    #    truth_level=True, name_suffix='truthRtop0' )
    #    #truth_level=True, name_suffix='uncappedtruthRtop0' )

    #single_negative_weight_map(
    #    [(1.0, 1.0, 1.0), (0.5, 1.0, 1.0), (3.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 10.0, 1.0), (0.0, 0.0, 1.0)],
    #    truth_level=True, name_suffix='truthRtop1' )
    #    #truth_level=True, name_suffix='uncappedtruthRtop1' )


if __name__ == '__main__': main()
