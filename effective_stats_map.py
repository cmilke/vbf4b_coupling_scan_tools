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
#from combination_utils import basis_full3D_max as _reco_basis 
from combination_utils import basis_full3D_old_minN as _reco_basis 
from reweight_utils import reco_reweight


def get_effective_stats_grid(base_couplings, base_events_list, kv_val, k2v_val_range, kl_val_range):
    reweight_vector_function = get_amplitude_function(base_couplings, as_scalar=False)
    base_event_weights = [ events[1] for events in base_events_list ]
    base_events_sum_square = sum([ events.sum() for events in base_event_weights ])**2
    base_square_events_sum = sum([ (events**2).sum() for events in base_event_weights ])
    base_effective_stats = base_events_sum_square / base_square_events_sum

    effective_stats_grid = numpy.zeros( (len(k2v_val_range),len(kl_val_range)) )
    for k2v_i, k2v_val in enumerate(k2v_val_range):
        for kl_j, kl_val in enumerate(kl_val_range):
            weight_vector = reweight_vector_function(k2v_val, kl_val, kv_val)[0]
            weighted_events = [ events*w for events,w in zip(base_event_weights, weight_vector) ]
            events_sum_square = sum([ events.sum() for events in weighted_events ])**2
            square_events_sum = sum([ (events**2).sum() for events in weighted_events ])
            effective_stats = events_sum_square / square_events_sum
            #normalized_stats = effective_stats / base_effective_stats
            #effective_stats_grid[k2v_i][kl_j] = normalized_stats
            effective_stats_grid[k2v_i][kl_j] = effective_stats
    return effective_stats_grid


def nice_coupling_string(coupling):
    str_list = []
    for kappa in coupling:
        if type(kappa) == int or kappa.is_integer():
            str_list.append( f'{int(kappa): 3d}  ' )
        else:
            str_list.append( f'{kappa: 5.1f}' )
    coupling_string = f'{str_list[0]}, {str_list[1]}, {str_list[2]}'
    return coupling_string



def draw_stats_map(basis_parameters, var_edges, kv_val, k2v_val_range, kl_val_range, effective_stats_grid,
            name_suffix='', title_suffix=None, vmin=None, vmax=None):

    num_bins = len(var_edges) - 1
    ranges = k2v_val_range[0], k2v_val_range[-1], kl_val_range[0], kl_val_range[-1]
    vartitle = '$m_{HH}$'
    plottable_couplings = [ [[],[],m] for m in ['v','o','^'] ]
    for couplings in basis_parameters:
        k2v, kl, kv = [float(sympy.Rational(c)) for c in couplings]
        index = 1 if kv == 1. else (0 if kv < 1. else 2)
        plottable_couplings[index][0].append(k2v)
        plottable_couplings[index][1].append(kl)

    #cmap = 'viridis'
    #cmap = 'Paired'
    cmap = 'viridis'
    #cmap = 'cool'
    #cmap = 'RdYlBu'
    fig, ax = plt.subplots()
    im = ax.imshow(effective_stats_grid.transpose(), vmin=vmin, vmax=vmax, extent=ranges, origin='lower', cmap=cmap)
    #im = ax.imshow(effective_stats_grid.transpose(), extent=ranges, origin='lower', cmap=cmap, norm=matplotlib.colors.LogNorm(vmin,vmax) )
    ax.set_xticks(ticks = numpy.arange(ranges[0],ranges[1]+1,1))
    ax.set_yticks(ticks = numpy.linspace(ranges[2],ranges[3],7))
    ax.set_xlabel('$\kappa_{2V}$')
    ax.set_ylabel('$\kappa_{\lambda}$')
    ax.grid()
    for (x,y,m) in plottable_couplings: ax.scatter(x,y, marker=m, color='cyan', s=9)
    ax.set_aspect('auto','box')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.11, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='')
    #fig.text(0.5, 0.04, '$\kappa_{2V}$', ha='center')
    #fig.text(0.04, 0.5, '$\kappa_{\lambda}$', va='center', rotation='vertical')

    basis_table = '$\kappa_{2V}$  ,  $\kappa_{\lambda}$  ,  $\kappa_{V}$   '
    for coupling in basis_parameters: basis_table += '\n'+nice_coupling_string(coupling)
    fig.text(.99, 1, basis_table, ha='right', va='top', fontsize='xx-small', family='monospace')

    range_title = f'{var_edges[0]:.0f}-{var_edges[-1]:.0f} in Bins of {var_edges[1]-var_edges[0]:.2f} GeV'
    title  = 'Effective Stats of Linear Combination,\nOver Range '+range_title
    if type(title_suffix) != type(None): title += '\n'+title_suffix
    fig.suptitle(title, fontsize=10, fontweight='bold')
    #plt.show()
    dpi = 500
    figname = 'effective_stats'+name_suffix
    plt.savefig('plots/error_maps/'+figname+'.png',dpi=dpi)
    plt.savefig('plots/.error_maps/'+figname+'.pdf',dpi=dpi)
    #output.savefig()
    #plt.close()




def single_reco_negative_weight_map(basis_parameters):
    var_edges = numpy.linspace(200, 1200, 31)
    kv_val = 1.0
    num_bins = 101
    k2v_val_range = numpy.linspace(-2,4,num_bins)
    kl_val_range = numpy.linspace(-14,16,num_bins)

    data_files = read_coupling_file('basis_files/nnt_coupling_file.dat')
    base_events_list = get_events(basis_parameters, data_files)

    effective_stats_grid = get_effective_stats_grid(basis_parameters, base_events_list, kv_val, k2v_val_range, kl_val_range)

    draw_stats_map(basis_parameters, var_edges, kv_val, k2v_val_range, kl_val_range, 1/effective_stats_grid)
            #vmax=1)





def main():
    # Sort out command-line arguments
#    parser = argparse.ArgumentParser()
#    parser.add_argument( "--basis", required = False, default = 'basis_files/nnt_basis.dat', type=str,
#        help = "File to provide basis states",)
#
#    args = parser.parse_args()

    #pdb.set_trace()
    #numpy.set_printoptions(precision=None, linewidth=400, threshold=10000, sign=' ', formatter={'float':lambda n: f'{n: 4.1f}'}, floatmode='fixed')
    single_reco_negative_weight_map(_reco_basis)


if __name__ == '__main__': main()
