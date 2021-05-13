import sys
import math
import argparse
import numpy
import sympy
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt

#import pdb

from fileio_utils import read_coupling_file, get_events, retrieve_reco_weights
from combination_utils import get_amplitude_function
from combination_utils import basis_full3D_max as _reco_basis 
from reweight_utils import reco_reweight


def get_Nweight_sum(couplings, weights, kv_val, k2v_val_range, kl_val_range, grid=False):
    #numpy.set_printoptions(threshold=sys.maxsize, linewidth=230, precision=0, floatmode='fixed', suppress=True)
    reweight_vector_function = get_amplitude_function(couplings, as_scalar=False)
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


# Depracated, use above
def get_negative_weight_grid(couplings, weights, errors, kv_val, k2v_val_range, kl_val_range):
    reweight_vector = get_amplitude_function(couplings, as_scalar=False)
    negative_weight_grid = numpy.zeros( (len(k2v_val_range),len(kl_val_range)) )
    for k2v_i, k2v_val in enumerate(k2v_val_range):
        for kl_j, kl_val in enumerate(kl_val_range):
            coupling_parameters = (k2v_val, kl_val, kv_val)
            combined_weights, combined_errors = reco_reweight(reweight_vector, coupling_parameters, weights, errors)
            num_negative = 0
            #print(k2v_val, kl_val)
            #print(combined_weights)
            #print()
            for i,w in enumerate(combined_weights):
                if w < 0:
                    num_negative += 1
                    #print(k2v_val, kl_val, i)
            negative_weight_grid[k2v_i][kl_j] = num_negative
    return negative_weight_grid


def nice_coupling_string(coupling):
    str_list = []
    for kappa in coupling:
        if type(kappa) == int or kappa.is_integer():
            str_list.append( f'{int(kappa): 3d}  ' )
        else:
            str_list.append( f'{kappa: 5.1f}' )
    coupling_string = f'{str_list[0]}, {str_list[1]}, {str_list[2]}'
    return coupling_string



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
    for coupling in basis_parameters: basis_table += '\n'+nice_coupling_string(coupling)
    fig.text(.99, 1, basis_table, ha='right', va='top', fontsize='xx-small', family='monospace')

    range_title = f'{var_edges[0]:.0f}-{var_edges[-1]:.0f} in Bins of {var_edges[1]-var_edges[0]:.2f} GeV'
    title  = 'Number of Bins in $M_{HH}$ with Negative Weights,\nOver Range '+range_title
    if type(title_suffix) != type(None): title += '\n'+title_suffix
    fig.suptitle(title, fontsize=10, fontweight='bold')
    #plt.show()
    dpi = 500
    plt.savefig('plots/error_maps/negative_weights'+name_suffix+'.png',dpi=dpi)
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

    data_files = read_coupling_file('basis_files/nnt_coupling_file.dat')
    base_events_list = get_events(basis_parameters, data_files)
    base_histograms = [ retrieve_reco_weights(var_edges, base_events) for base_events in base_events_list ]
    base_weights, base_errors = numpy.array(list(zip(*base_histograms)))
    #print(base_weights)

    negative_weight_grid = get_Nweight_sum(basis_parameters, base_weights, base_errors, kv_val, k2v_val_range, kl_val_range, grid=True)

    draw_error_map(basis_parameters, var_edges, kv_val, k2v_val_range, kl_val_range, negative_weight_grid, 
                name_suffix='_mc16ade_old', title_suffix='MC16a/d/e')
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
