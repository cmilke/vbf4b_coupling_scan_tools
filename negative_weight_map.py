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


def get_Nweight_sum(couplings, weights, kv_val, k2v_val_range, kl_val_range, grid=False):
    #numpy.set_printoptions(threshold=sys.maxsize, linewidth=230, precision=0, floatmode='fixed', suppress=True)
    reweight_vector_function = combination_utils.get_amplitude_function(couplings, as_scalar=False)
    k2v_grid, kl_grid = numpy.meshgrid(k2v_val_range, kl_val_range)
    multiplier_grid_vector = reweight_vector_function(k2v_grid, kl_grid, kv_val)[0]
    combined_weights = sum([ multiplier_grid[...,None] * w for multiplier_grid, w in zip(multiplier_grid_vector, weights) ])
    negative_boolean_grid = combined_weights < 0
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

    range_title = f'{var_edges[0]:.0f}-{var_edges[-1]:.0f} in Bins of {var_edges[1]-var_edges[0]:.2f} GeV'
    title  = 'Number of Bins in $M_{HH}$ with Negative Weights,\nOver Range '+range_title
    if type(title_suffix) != type(None): title += '\n'+title_suffix
    fig.suptitle(title, fontsize=10, fontweight='bold')
    #plt.show()
    dpi = 500
    figname = 'negative_weights'+name_suffix
    plt.savefig('plots/error_maps/'+figname+'.png',dpi=dpi)
    plt.savefig('plots/.error_maps/'+figname+'.pdf',dpi=dpi)
    #output.savefig()
    #plt.close()




def single_reco_negative_weight_map(basis_parameters):
    #var_edges = numpy.linspace(250, 2000, 101)
    #var_edges = range(200, 1200, 33)
    var_edges = numpy.linspace(200, 1200, 31)
    kv_val = 1.0
    #k2v_val_range = [0,1,2] 
    #kl_val_range = [-8,-9,-10]
    k2v_val_range = numpy.linspace(-1,3,101)
    kl_val_range = numpy.linspace(-14,16,101)

    data_files = fileio_utils.read_coupling_file('basis_files/nnt_coupling_file_2021May.dat')
    base_events_list = fileio_utils.get_events(basis_parameters, data_files)
    base_histograms = [ fileio_utils.retrieve_reco_weights(var_edges, base_events) for base_events in base_events_list ]
    base_weights, base_errors = numpy.array(list(zip(*base_histograms)))
    #print(base_weights)

    negative_weight_grid = get_Nweight_sum(basis_parameters, base_weights, kv_val, k2v_val_range, kl_val_range, grid=True)

    draw_error_map(basis_parameters, var_edges, kv_val, k2v_val_range, kl_val_range, negative_weight_grid, 
                name_suffix='_mc16ade', title_suffix='MC16a/d/e')
                #name_suffix='_mc16d_old', vmax=12, title_suffix='MC16d Only')
                #name_suffix='_mc16ad_old', vmax=12, title_suffix='MC16a and MC16d')





def main():
    # Sort out command-line arguments
#    parser = argparse.ArgumentParser()
#    parser.add_argument( "--basis", required = False, default = 'basis_files/nnt_basis.dat', type=str,
#        help = "File to provide basis states",)
#
#    args = parser.parse_args()

    #pdb.set_trace()
    single_reco_negative_weight_map(_reco_basis)


if __name__ == '__main__': main()
