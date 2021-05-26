import sys
import math
import argparse
import numpy
import statistics
import sympy
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt

#import pdb

from combination_utils import get_amplitude_function, get_theory_xsec_function


def get_variance_count_map(couplings, kv_val, k2v_val_range, kl_val_range):
    reweight_vector = get_amplitude_function(couplings, as_scalar=False)
    weight_contribution_grid = numpy.zeros( (len(k2v_val_range),len(kl_val_range)) )

    for k2v_i, k2v_val in enumerate(k2v_val_range):
        for kl_j, kl_val in enumerate(kl_val_range):
            coupling_parameters = (k2v_val, kl_val, kv_val)
            vector = reweight_vector(*coupling_parameters)[0]

            mean = statistics.mean(vector)
            stdev = statistics.stdev(vector)
            contribution = 0
            for v in vector:
                if abs(v-mean) > stdev: contribution += 1

            #norm_vector = vector/vector.sum()
            #contribution = statistics.stdev(norm_vector)

            #max_contribution = max(abs(norm_vector))
            weight_contribution_grid[k2v_i][kl_j] = contribution
    return weight_contribution_grid


def get_theoretical_solidarity_map(couplings, kv_val, k2v_val_range, kl_val_range, grid=False):
    #numpy.set_printoptions(threshold=sys.maxsize, linewidth=230, precision=0, floatmode='fixed', suppress=True)
    theory_xsec_function = get_theory_xsec_function()
    xsec_list = [ theory_xsec_function(c) for c in couplings ]
    reweight_vector_function = get_amplitude_function(couplings, as_scalar=False)

    k2v_grid, kl_grid = numpy.meshgrid(k2v_val_range, kl_val_range)
    multiplier_grid_vector = reweight_vector_function(k2v_grid, kl_grid, kv_val)[0]
    scaled_xsecs = numpy.array([ multiplier_grid*xsec for multiplier_grid, xsec in zip(multiplier_grid_vector, xsec_list) ])
    abs_stdev = abs(scaled_xsecs).std(axis=0)
    combined_xsecs = scaled_xsecs.sum(axis=0)
    solidarity_grid = combined_xsecs / abs_stdev
    if grid:
        return solidarity_grid
    else:
        grid_pixel_area = (k2v_val_range[1] - k2v_val_range[0]) * (kl_val_range[1] - kl_val_range[0])
        solidarity_integral = solidarity_grid.sum() * grid_pixel_area
        return solidarity_integral


def get_reco_solidarity_map(couplings, weights, kv_val, k2v_val_range, kl_val_range, grid=False):
    xsec_list = [ w.sum() for w in weights ]
    reweight_vector_function = get_amplitude_function(couplings, as_scalar=False)

    k2v_grid, kl_grid = numpy.meshgrid(k2v_val_range, kl_val_range)
    multiplier_grid_vector = reweight_vector_function(k2v_grid, kl_grid, kv_val)[0]
    scaled_xsecs = numpy.array([ multiplier_grid*xsec for multiplier_grid, xsec in zip(multiplier_grid_vector, xsec_list) ])
    abs_stdev = abs(scaled_xsecs).std(axis=0)
    combined_xsecs = scaled_xsecs.sum(axis=0)
    solidarity_grid = combined_xsecs / abs_stdev
    if grid:
        return solidarity_grid
    else:
        grid_pixel_area = (k2v_val_range[1] - k2v_val_range[0]) * (kl_val_range[1] - kl_val_range[0])
        solidarity_integral = solidarity_grid.sum() * grid_pixel_area
        return solidarity_integral


def get_theory_effective_stats_map(couplings, kv_val, k2v_val_range, kl_val_range):
    reweight_vector = get_amplitude_function(couplings, as_scalar=False)
    theory_xsec_function = get_theory_xsec_function()
    xsec_vector = [ theory_xsec_function(c) for c in couplings ]
    weight_contribution_grid = numpy.zeros( (len(k2v_val_range),len(kl_val_range)) )

    for k2v_i, k2v_val in enumerate(k2v_val_range):
        for kl_j, kl_val in enumerate(kl_val_range):
            coupling_parameters = (k2v_val, kl_val, kv_val)
            vector = reweight_vector(*coupling_parameters)[0]
            weighted_xsec = xsec_vector * vector
            effective_stats =  sum(weighted_xsec)**2 / sum(weighted_xsec**2)
            weight_contribution_grid[k2v_i][kl_j] = effective_stats
    return weight_contribution_grid


def get_test_map(couplings, kv_val, k2v_val_range, kl_val_range):
    reweight_vector = get_amplitude_function(couplings, as_scalar=False)
    theory_xsec_function = get_theory_xsec_function()
    xsec_vector = [ theory_xsec_function(c) for c in couplings ]
    weight_contribution_grid = numpy.zeros( (len(k2v_val_range),len(kl_val_range)) )

    #numpy.set_printoptions(precision=None, linewidth=400, threshold=100, sign=' ', formatter={'float':lambda n: f'{n: 4.1f}'}, floatmode='fixed')
    for k2v_i, k2v_val in enumerate(k2v_val_range):
        for kl_j, kl_val in enumerate(kl_val_range):
            coupling_parameters = (k2v_val, kl_val, kv_val)
            vector = reweight_vector(*coupling_parameters)[0]
            weighted_xsec = xsec_vector * vector
            contribution = 0
            for wxi in weighted_xsec:
                for wxj in weighted_xsec:
                    #if wxj == wxi: ratio = 1
                    #elif wxj == 0: ratio = 100000
                    #else: ratio = abs(wxi/wxj)
                    #contribution += ratio
                    contribution += abs(wxi - wxj)
            weight_contribution_grid[k2v_i][kl_j] = contribution/weighted_xsec.sum()
    return weight_contribution_grid


def nice_coupling_string(coupling):
    str_list = []
    for kappa in coupling:
        if type(kappa) == int or kappa.is_integer():
            str_list.append( f'{int(kappa): 3d}  ' )
        else:
            str_list.append( f'{kappa: 5.1f}' )
    coupling_string = f'{str_list[0]}, {str_list[1]}, {str_list[2]}'
    return coupling_string



def draw_contribution_map(basis_parameters, kv_val, k2v_val_range, kl_val_range, weight_contribution_grid,
            name_suffix='', title_suffix=None, vmin=None, vmax=None, transpose = False,
            title = 'Contribution of Weight to Combined Distribution'):
    ranges = k2v_val_range[0], k2v_val_range[-1], kl_val_range[0], kl_val_range[-1]
    vartitle = '$m_{HH}$'
    plottable_couplings = [ [[],[],m] for m in ['v','o','^'] ]
    for couplings in basis_parameters:
        k2v, kl, kv = [float(sympy.Rational(c)) for c in couplings]
        index = 1 if kv == 1. else (0 if kv < 1. else 2)
        plottable_couplings[index][0].append(k2v)
        plottable_couplings[index][1].append(kl)

    cmap = 'viridis'
    #cmap = 'Paired'
    #cmap = 'plasma'
    #cmap = 'cool'
    #cmap = 'RdYlBu'
    fig, ax = plt.subplots()
    if transpose: weight_contribution_grid = weight_contribution_grid.transpose()
    im = ax.imshow(weight_contribution_grid, extent=ranges, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    #im = ax.imshow(weight_contribution_grid.transpose(), extent=ranges, origin='lower', cmap=cmap, norm=matplotlib.colors.LogNorm(vmin,vmax) )
    ax.set_xticks(ticks = numpy.arange(ranges[0],ranges[1]+1,1))
    ax.set_yticks(ticks = numpy.linspace(ranges[2],ranges[3],7))
    ax.set_xlabel('$\kappa_{2V}$')
    ax.set_ylabel('$\kappa_{\lambda}$')
    ax.grid()
    for (x,y,m) in plottable_couplings: ax.scatter(x,y, marker=m, color='cyan', s=9)
    ax.set_aspect('auto','box')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.11, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=title)
    #fig.text(0.5, 0.04, '$\kappa_{2V}$', ha='center')
    #fig.text(0.04, 0.5, '$\kappa_{\lambda}$', va='center', rotation='vertical')

    basis_table = '$\kappa_{2V}$  ,  $\kappa_{\lambda}$  ,  $\kappa_{V}$   '
    for coupling in basis_parameters: basis_table += '\n'+nice_coupling_string(coupling)
    fig.text(.99, 1, basis_table, ha='right', va='top', fontsize='xx-small', family='monospace')

    title+=' Heatmap'
    if type(title_suffix) != type(None): title += '\n'+title_suffix
    fig.suptitle(title, fontsize=10, fontweight='bold')
    dpi = 500
    figname = 'contribution_max'+name_suffix
    plt.savefig('plots/error_maps/'+figname+'.png',dpi=dpi)
    plt.savefig('plots/.error_maps/'+figname+'.pdf',dpi=dpi)




def single_weight_contribution_map(basis_parameters, name_suffix='', title_suffix=''):
    kv_val = 1.0
    #num_bins=31
    num_bins=101
    k2v_val_range = numpy.linspace(-2,4,num_bins)
    kl_val_range = numpy.linspace(-14,16,num_bins)

    weight_contribution_grid = get_theoretical_solidarity_map(basis_parameters, kv_val, k2v_val_range, kl_val_range, grid=True)
    #weight_contribution_grid = get_test_map(basis_parameters, kv_val, k2v_val_range, kl_val_range)

    grid_pixel_area = (k2v_val_range[1] - k2v_val_range[0]) * (kl_val_range[1] - kl_val_range[0])
    grid_integral = numpy.sum( weight_contribution_grid * grid_pixel_area )
    print(name_suffix, grid_integral)

    title = 'Solidarity Value'
    draw_contribution_map(basis_parameters, kv_val, k2v_val_range, kl_val_range, weight_contribution_grid, 
                name_suffix=name_suffix, title=title, title_suffix=f'Integral = {int(grid_integral)}')
                #name_suffix=name_suffix, title_suffix=title_suffix, vmin=.1, vmax=6)
                #name_suffix=name_suffix, title_suffix=title_suffix, vmin=0, vmax=5)
                #name_suffix=name_suffix, title_suffix=title_suffix, vmin=-5, vmax=0)





def main():
    # Sort out command-line arguments
#    parser = argparse.ArgumentParser()
#    parser.add_argument( "--basis", required = False, default = 'basis_files/nnt_basis.dat', type=str,
#        help = "File to provide basis states",)
#
#    args = parser.parse_args()

    #pdb.set_trace()
    #single_weight_contribution_map(
    #    [ (1, 1, 1), (2, 1, 1), (1.5, 1, 1), (0, 1, 0.5), (1, 0, 1), (1, 10, 1) ],
    #    name_suffix='rank28', title_suffix='Rank 28')

    #single_weight_contribution_map(
    #    [(1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (0.0, 1.0, 0.5), (1.0, 10.0, 1.0), (1.0, 2.0, 1.0), (4.0, 1.0, 1.0)],
    #    name_suffix='rank25', title_suffix='Rank 25')

    #single_weight_contribution_map(
    #    [(1.0, 1.0, 1.0), (1.5, 1.0, 1.0), (0.0, 1.0, 0.5), (1.0, 0.0, 1.0), (1.0, 2.0, 1.0), (4.0, 1.0, 1.0)],
    #    name_suffix='rank20', title_suffix='Rank 20')

    #single_weight_contribution_map(
    #    [(1.0, 1.0, 1.0), (1.5, 1.0, 1.0), (0.0, 1.0, 0.5), (1.0, 10.0, 1.0), (1.0, 2.0, 1.0), (0.0, 1.0, 1.0)],
    #    name_suffix='rank15', title_suffix='Rank 15')

    #single_weight_contribution_map(
    #    [(1.0, 1.0, 1.0), (0.0, 1.0, 0.5), (1.0, 10.0, 1.0), (1.0, 2.0, 1.0), (0.0, 1.0, 1.0), (0.5, 1.0, 1.0)],
    #    name_suffix='rank10', title_suffix='Rank 10')

    #single_weight_contribution_map(
    #    [(1.0, 1.0, 1.0), (1.5, 1.0, 1.0), (0.0, 1.0, 0.5), (1.0, 0.0, 1.0), (1.0, 10.0, 1.0), (0.5, 1.0, 1.0) ],
    #    name_suffix='rank05', title_suffix='Rank 5')

    single_weight_contribution_map(
        [ (1, 1, 1), (0, 1, 0.5), (1, 0, 1), (1, 10, 1), (0.5, 1, 1), (4, 1, 1) ],
        name_suffix='rank01', title_suffix='')

    single_weight_contribution_map(
        [ (1, 1, 1.1), (1, 1.1, 1), (1.1, 1, 1), (0.9, 1, 1.1), (1, 1.1, 0.9), (1.1, 0.9, 1) ],
        name_suffix='exp', title_suffix='Experimentation')



if __name__ == '__main__': main()
