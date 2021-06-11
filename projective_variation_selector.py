import sys
import pickle
import numpy
import itertools
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#import pdb

import combination_utils
import weight_contribution_map


_k2v_vals = numpy.arange(-1,4.5,0.5)
_kl_vals = numpy.arange(-9,11,1)
_kv_vals = [ 0.5, 1, 1.5 ]
#_k2v_vals = [2.5]
#_kl_vals = [10]
#_kv_vals = [1]
_possible_variations = list(itertools.product( _k2v_vals, _kl_vals, _kv_vals ))


def calculate_solidarity_results():
    kv_fixed = 1.0
    num_kappa_bins = 100
    k2v_val_range = numpy.linspace(-2,4,num_kappa_bins+1)
    kl_val_range = numpy.linspace(-14,16,num_kappa_bins+1)
    grid_pixel_area = (k2v_val_range[1] - k2v_val_range[0]) * (kl_val_range[1] - kl_val_range[0])
    k2v_grid, kl_grid = numpy.meshgrid(k2v_val_range, kl_val_range)
    mask = lambda k2v, kl: ((k2v-1)/1)**2 + ((kl-1)/10)**2 < 1
    mask_grid = mask(k2v_grid, kl_grid)

    pre_existing_variations = [
        #(1    ,   1  ,  1     ), # Ignore SM point b/c I hard-code it in further down to ease the combinatorics
        (0    ,   1  ,  1     ),
        (0.5  ,   1  ,  1     ),
        (1.5  ,   1  ,  1     ),
        (2    ,   1  ,  1     ),
        (3    ,   1  ,  1     ),
        (1    ,   0  ,  1     ),
        #(1    ,   2  ,  1     ),
        (1    ,   10 ,  1     ),
        (1    ,   1  ,  0.5   ),
        (1    ,   1  ,  1.5   ),
        (0    ,   0  ,  1     )
    ]

    sm_coupling = (1,1,1)
    kl2_coupling = (1,2,1)
    theory_xsec_function = combination_utils.get_theory_xsec_function()
    sm_tuple = (sm_coupling, theory_xsec_function(sm_coupling))
    kl2_tuple = (kl2_coupling, theory_xsec_function(kl2_coupling))

    solidarity_cap = 10

    variation_result_list = []
    variations_computed = 0
    total_variations = len(_possible_variations)
    coupling_xsec_tuples = [ (var, theory_xsec_function(var)) for var in pre_existing_variations ]
    for new_variation in _possible_variations:
        print('------ Calculating ' + str(new_variation))
        prospective_variations = pre_existing_variations
        new_tuple = (new_variation, theory_xsec_function(new_variation))
        solidarity_list = []

        total = 0
        for nonSM_coupling_tuples in itertools.combinations(coupling_xsec_tuples,3):
            coupling_base_tuple = [ sm_tuple, kl2_tuple, *nonSM_coupling_tuples, new_tuple ]
            coupling_base, xsec_list = list(zip(*coupling_base_tuple))
            xsec_array = numpy.array(xsec_list)
            reweight_vector_function = combination_utils.get_amplitude_function(coupling_base, as_scalar=False)
            if type(reweight_vector_function) == type(None): continue

            multiplier_grid_vector = reweight_vector_function(k2v_grid, kl_grid, kv_fixed)[0]
            scaled_xsecs =  multiplier_grid_vector * xsec_array[:,None,None]

            abs_stdev = abs(scaled_xsecs).std(axis=0)
            combined_xsecs = scaled_xsecs.sum(axis=0)
            solidarity_grid = combined_xsecs / abs_stdev
            metric_integral = solidarity_grid.sum() * grid_pixel_area

            #effective_stats =  scaled_xsecs.sum(axis=0)**2 / (scaled_xsecs**2).sum(axis=0)
            #effective_stats_grid = solidarity_grid * mask_grid
            #effective_stats_grid[solidarity_grid > solidarity_cap] = solidarity_cap
            #metric_integral = effective_stats.sum() * grid_pixel_area

            solidarity_list.append(metric_integral)

            total += 1
            if total % 100 == 0: print(total)
        variation_result_list.append( (new_variation, solidarity_list) )
        variations_computed += 1
        print(f'Completed variation {variations_computed} / {total_variations}\n')
    print(f'\n\nAll possible variations checked!')
    return variation_result_list



def generate_heatmap_overview(variation_result_list):
    k2v_grid_vals = numpy.arange(-1,5,0.5)
    kl_grid_vals = numpy.arange(-9,12,1)
    performance_grid = numpy.zeros( (len(kl_grid_vals)-1, len(k2v_grid_vals)-1) )
    kv_fixed = 1
    kv_range = [0.5, 1, 1.5]
    #kv_range = [1]
    for kv_fixed in kv_range:
        min_performance = float('inf')
        for variation, result_list in variation_result_list:
            k2v, kl, kv = variation
            if kv != kv_fixed: continue
            k2v_index = numpy.where(k2v_grid_vals == k2v)[0][0]
            kl_index = numpy.where(kl_grid_vals == kl)[0][0]
            #performance = sum([ r for r in result_list if r > 50 ]) / 1000
            performance = sum(result_list) / 1000
            performance_grid[kl_index][k2v_index] = performance
            if performance < min_performance: min_performance = performance
        print(performance_grid[::-1])
        print()

        fig, ax = plt.subplots()
        #im = ax.imshow(negative_weight_grid.transpose(), vmin=vmin, vmax=vmax, extent=ranges, origin='lower', cmap='plasma')
        im = ax.imshow(performance_grid, vmin=min_performance, vmax=None,
                extent=(k2v_grid_vals[0], k2v_grid_vals[-1], kl_grid_vals[0], kl_grid_vals[-1]), origin='lower', cmap='viridis')

        ax.set_xticks(ticks = k2v_grid_vals)
        ax.set_yticks(ticks = kl_grid_vals)
        ax.set_xlabel('$\kappa_{2V}$')
        ax.set_ylabel('$\kappa_{\lambda}$')

        ax.set_xticks(k2v_grid_vals[:-1]+(k2v_grid_vals[1]-k2v_grid_vals[0])/2, minor=True)
        ax.set_xticklabels('') # Clear major tick labels
        ax.set_xticklabels(k2v_grid_vals[:-1], minor=True, fontsize=8)

        ax.set_yticks(kl_grid_vals[:-1]+(kl_grid_vals[1]-kl_grid_vals[0])/2, minor=True)
        ax.set_yticklabels('') # Clear major tick labels
        ax.set_yticklabels(kl_grid_vals[:-1], minor=True, fontsize=6)

        ax.grid()
        ax.set_aspect('auto','box')

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.11, 0.03, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Performance (x$10^{5}$)')

        title = 'Projected Performance Effected by New Variation for 'r'$\kappa_{V} = 'f'{kv_fixed}$'
        #title += '\n(Measured as Sum Total of All Solidarity Integrals)'
        title += '\n(Measured as Sum Total of All Effective Stats Integrals)'
        fig.suptitle(title, fontsize=10, fontweight='bold')
        dpi = 500
        figname = f'solidarity_performance_kv{kv_fixed}'
        plt.savefig('plots/dump/'+figname+'.pdf',dpi=dpi)



def generate_2D_histogram(variation_result_list):
    bin_edges = numpy.linspace(0,1000,20)
    #bin_edges = numpy.linspace(0,200,20)
    #bin_edges = numpy.linspace(60,200,20)

    solidarity_results = []
    for variation, result_list in variation_result_list:
        result_array = numpy.histogram(result_list, bins=bin_edges)[0]
        result_count = sum(result_array)
        result_sum = sum(bin_edges[:-1]*result_array)
        #result_sum = result_count
        solidarity_results.append( (result_sum, result_count, variation, result_array) )
    solidarity_results.sort(reverse=True)


    #num_to_plot=20
    num_to_plot=len(solidarity_results)
    top_results = []
    top_variations = []
    for result_sum, result_count, variation, result_array in solidarity_results[:num_to_plot]:
        print(f'{int(result_sum)}', variation)
        top_results.append(result_array)
        #top_variations.append(variation)
        top_variations.append(str(variation)+'\n'+str(result_count))
    result_grid = numpy.array(top_results)

    fig, ax = plt.subplots()
    im = ax.imshow(result_grid, cmap='viridis', extent=(bin_edges[0],bin_edges[-2], 0,num_to_plot) )

    #ax.set_xlabel('Theoretical Solidarity Value')
    ax.set_xlabel('Effective Stats')
    ax.set_ylabel('Added Variation 'r'$(\kappa_{2V},\kappa_{\lambda},\kappa_{V})$')

    y_param_labels = [ var for var in top_variations[::-1] ]
    y_param_ticks = numpy.array(range(len(top_variations)))
    ax.set_yticks(y_param_ticks)
    ax.set_yticks(y_param_ticks+0.5, minor=True)
    ax.set_yticklabels('') # Clear major tick labels
    ax.set_yticklabels(y_param_labels, minor=True, fontsize=4)

    ##ax.grid()
    ax.set_aspect('auto','box')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.86, 0.11, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Number of Valid Combinations')

    #title  = 'Projected Solidarity Distributions\nfor Different Additional Variations'
    title  = 'Projected Effective Stat Integrals\nfor Different Additional Variations'
    fig.suptitle(title, fontsize=10, fontweight='bold')

    figname = 'projective_solidarity_dump'
    #plt.savefig('plots/dump/'+figname+'.png',dpi=dpi)
    plt.savefig('plots/dump/'+figname+'.pdf')
        


def main():
    numpy.set_printoptions(threshold=sys.maxsize, linewidth=230, precision=0, floatmode='fixed', suppress=True)

    use_cached = len(sys.argv) == 1
    variation_result_list = None
    cache_file = '.cached_projective_solidarity_lists.p'
    if use_cached:
        #cache_file = '.cached_projective_solidarity_lists.elliptic.p'
        #cache_file = '.cached_projective_solidarity_lists.full.p'
        #cache_file = '.cached_projective_solidarity_lists.capped.p'
        #cache_file = '.cached_projective_solidarity_lists.stats.p'
        variation_result_list = pickle.load(open(cache_file,'rb'))
    else:
        variation_result_list = calculate_solidarity_results()
        pickle.dump(variation_result_list, open(cache_file,'wb'))

    generate_heatmap_overview(variation_result_list)
    generate_2D_histogram(variation_result_list)
        


if __name__ == '__main__': main()
