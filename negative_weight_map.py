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


def generate_error_maps(basis_parameters):
    reweight_vector = get_amplitude_function(basis_parameters, as_scalar=False)
    #var_edges = numpy.linspace(250, 2000, 101)
    #var_edges = range(200, 1200, 33)
    var_edges = numpy.linspace(200, 1200, 31)
    num_bins = len(var_edges) - 1

    data_files = read_coupling_file('basis_files/nnt_coupling_file.dat')
    base_events_list = get_events(basis_parameters, data_files)
    base_histograms = [ retrieve_reco_weights(var_edges, base_events) for base_events in base_events_list ]
    base_weights, base_errors = numpy.array(list(zip(*base_histograms)))
    #print(base_weights)

    kv_val = 1.0
    k2v_val_range = numpy.linspace(-1,3,51)
    kl_val_range = numpy.linspace(-14,16,51)
    #k2v_val_range = [0,1,2] 
    #kl_val_range = [-8,-9,-10]
    varlen = len(var_edges)
    negative_frac_grid = numpy.zeros( (len(k2v_val_range),len(kl_val_range)) )
    for k2v_i, k2v_val in enumerate(k2v_val_range):
        for kl_j, kl_val in enumerate(kl_val_range):
            coupling_parameters = (k2v_val, kl_val, kv_val)
            combined_weights, combined_errors = reco_reweight(reweight_vector, coupling_parameters, base_weights, base_errors)
            num_negative = 0
            #print(k2v_val, kl_val)
            #print(combined_weights)
            #print()
            for i,w in enumerate(combined_weights):
                if w < 0:
                    num_negative += 1
                    #print(k2v_val, kl_val, i)
            negative_frac_grid[k2v_i][kl_j] = num_negative

    ranges = k2v_val_range[0], k2v_val_range[-1], kl_val_range[0], kl_val_range[-1]
    vartitle = '$m_{HH}$'
    plottable_couplings = [ [[],[],m] for m in ['v','o','^'] ]
    for couplings in basis_parameters:
        k2v, kl, kv = [float(sympy.Rational(c)) for c in couplings]
        index = 1 if kv == 1. else (0 if kv < 1. else 2)
        plottable_couplings[index][0].append(k2v)
        plottable_couplings[index][1].append(kl)

    fig, ax = plt.subplots()
    im = ax.imshow(negative_frac_grid.transpose(), vmin=0, extent=ranges, origin='lower', cmap='plasma')
    ax.set_xticks(ticks = range(-1,4))
    ax.set_yticks(ticks = numpy.linspace(-14,16,7))
    ax.set_xlabel('$\kappa_{2V}$')
    ax.set_ylabel('$\kappa_{\lambda}$')
    ax.grid()
    for (x,y,m) in plottable_couplings: ax.scatter(x,y, marker=m, color='cyan', s=9)
    ax.set_aspect('auto','box')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Num Bins (Out of '+str(num_bins)+' Total)')
    #fig.text(0.5, 0.04, '$\kappa_{2V}$', ha='center')
    #fig.text(0.04, 0.5, '$\kappa_{\lambda}$', va='center', rotation='vertical')
    range_title = f'{var_edges[0]:.0f}-{var_edges[-1]:.0f} in Bins of {var_edges[1]-var_edges[0]:.2f} GeV'
    fig.suptitle('Number of Bins in $M_{HH}$ with Negative Weights,\nOver Range '+range_title, fontsize=10, fontweight='bold')
    #plt.show()
    dpi = 500
    plt.savefig('plots/error_maps/negative_weights.png',dpi=dpi)
    #output.savefig()
    #plt.close()




def main():
    # Sort out command-line arguments
#    parser = argparse.ArgumentParser()
#    parser.add_argument( "--basis", required = False, default = 'basis_files/nnt_basis.dat', type=str,
#        help = "File to provide basis states",)
#
#    args = parser.parse_args()

    #pdb.set_trace()
    generate_error_maps(_reco_basis)


if __name__ == '__main__': main()
