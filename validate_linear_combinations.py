import sys
import argparse
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#import pdb

import fileio_utils
import combination_utils
from combination_utils import get_amplitude_function
#from combination_utils import basis_full3D_old_minN as _reco_basis 
from combination_utils import basis_full3D_2021May_minN as _reco_basis 
from reweight_utils import reco_reweight
import negative_weight_map
import view_linear_combination



def draw_1D_mhh_heatmap(couplings, weights, var_edges, k2v_vals, kl_vals, kv_vals,
        base_equations=None, which_coupling=None, filename='test', title_suffix=None, vrange=None):

    #numpy.set_printoptions(threshold=sys.maxsize, linewidth=230, precision=0, floatmode='fixed', suppress=True)
    reweight_vector_function = combination_utils.get_amplitude_function(couplings, as_scalar=False, base_equations=base_equations)
    multiplier_array_vector = reweight_vector_function(k2v_vals, kl_vals, kv_vals)[0]
    weight_grid = sum([ multiplier_array[...,None] * w for multiplier_array, w in zip(multiplier_array_vector, weights) ])

    if which_coupling == 'k2v':
        num_bins = len(k2v_vals) - 1
        ranges = k2v_vals[0], k2v_vals[-1], var_edges[0], var_edges[-1]
        title = 'Combined $m_{HH}$ Across 'r'$\kappa_{2V}$'
        axis_title = r'$\kappa_{2V}$'
        plottable_couplings = [ c[0] for c in couplings ]
        tick_vals = numpy.arange(ranges[0],ranges[1]+1,1)
    elif which_coupling == 'kl':
        num_bins = len(kl_vals) - 1
        ranges = kl_vals[0], kl_vals[-1], var_edges[0], var_edges[-1]
        title = 'Combined $m_{HH}$ Across 'r'$\kappa_{\lambda}$'
        axis_title = r'$\kappa_{\lambda}$'
        plottable_couplings = [ c[1] for c in couplings ]
        tick_vals = numpy.linspace(ranges[0],ranges[1],7)

    fig, ax = plt.subplots()
    if type(vrange) == type(None):
        im = ax.imshow(weight_grid.transpose(), cmap='viridis', extent=ranges, origin='lower', norm=matplotlib.colors.LogNorm() )
    else:
        im = ax.imshow(weight_grid.transpose(), cmap='viridis', extent=ranges, origin='lower', norm=matplotlib.colors.LogNorm(*vrange) )
    #im = ax.imshow(weight_grid.transpose(), extent=ranges, origin='lower', cmap='viridis')
    ax.set_xticks(ticks = tick_vals)
    ax.set_xlabel(axis_title)
    ax.set_ylabel('$m_{HH}$')
    ax.grid()
    for x in plottable_couplings: ax.vlines(x, ymin=var_edges[0], ymax=var_edges[-1], color='red')
    ax.set_aspect('auto','box')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.11, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Bin Weight')

    basis_table = '$\kappa_{2V}$  ,  $\kappa_{\lambda}$  ,  $\kappa_{V}$   '
    for coupling in couplings: basis_table += '\n'+combination_utils.nice_coupling_string(coupling)
    fig.text(.99, 1, basis_table, ha='right', va='top', fontsize='xx-small', family='monospace')

    if type(title_suffix) != type(None): title += '\n'+title_suffix
    fig.suptitle(title, fontsize=10, fontweight='bold')
    dpi = 500
    figname = filename
    #plt.savefig('plots/scan_maps/'+figname+'.png',dpi=dpi)
    plt.savefig('plots/scan_maps/'+figname+'.pdf',dpi=dpi)



def plot_histogram(hist_name, hist_title, edge_list, coupling_parameters,
        linearly_combined_weights, linearly_combined_errors,
        verification_weights, verification_errors,
        alt_linearly_combined_weights=None, alt_linearly_combined_errors=None,
        alt_label='1D Combination',
        range_specs=None, xlabel='Truth $m_{HH}$ (GeV)', normalize=False,
        generated_label='Linear Combination', generated_color='blue'):

    print('Plotting '+hist_name+' for ' + str(coupling_parameters))
    draw_verify = type(verification_weights) != type(None) and type(verification_errors) != type(None)
    draw_alt_comb = type(alt_linearly_combined_weights) != type(None) and type(alt_linearly_combined_errors) != type(None)

    if draw_verify:
        fig, (ax_main, ax_ratio) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios':[4,1]} )
    else:
        fig, ax_main = plt.subplots()


    xpositions = 0.5*(edge_list[1:]+edge_list[:-1])
    counts, bins, points = ax_main.errorbar( xpositions, linearly_combined_weights,
        yerr=linearly_combined_errors, label=generated_label,
        marker='+', markersize=2, capsize=2, color=generated_color, linestyle='none', linewidth=1, zorder=3)

    if draw_alt_comb:
        alt_counts, alt_bins, alt_points = ax_main.errorbar( xpositions, alt_linearly_combined_weights,
            yerr=alt_linearly_combined_errors, label=alt_label,
            marker='x', markersize=2, capsize=2, color='purple', linestyle='none', linewidth=1, zorder=3)

    if draw_verify:
        vcounts, vbins, vhists = ax_main.hist( [edge_list[:-1]]*2,
            weights=[verification_weights-verification_errors, 2*verification_errors],
            label=['Generated MC', 'MC Statistical Error'],
            bins=edge_list, fill=True, histtype='barstacked', zorder=1, alpha=0.5, color=['green','red'])
        plt.setp(vhists[1], hatch='/////')

        safe_error = verification_errors.copy()
        #safe_error = linearly_combined_errors.copy()
        safe_error[ safe_error == 0 ] = float('inf')
        safe_combination = linearly_combined_weights.copy()
        safe_combination[ safe_combination == 0 ] = float('inf')
        rcounts, rbins, rpoints = ax_ratio.errorbar( xpositions, (linearly_combined_weights-verification_weights)/safe_error,
            yerr=linearly_combined_errors/safe_error, label='MC Statistical Error check',
            marker='+', markersize=2, capsize=2, color=generated_color, linestyle='none', linewidth=1, zorder=3)

        if draw_alt_comb:
            alt_rcounts, alt_rbins, alt_rpoints = ax_ratio.errorbar( xpositions, (alt_linearly_combined_weights-verification_weights)/safe_error,
                yerr=alt_linearly_combined_errors/safe_error, label='MC Statistical Error check',
                marker='x', markersize=2, capsize=2, color='purple', linestyle='none', linewidth=1, zorder=3)

        ax_ratio.set(ylabel=r'$\frac{lin. comb. - gen.}{stat. error}$', xlabel=xlabel)
        #ax_ratio.set(ylabel=r'$\frac{lin. comb. - gen.}{comb. error}$', xlabel=xlabel)
        ax_ratio.set_ylim([-3,3])
        #ax_ratio.set_yticks(ticks=range(-3,4))
        ax_ratio.set_yticks(ticks=[-2,-1,0,1,2])
        zero_line = ax_ratio.hlines(0,xmin=edge_list[0],xmax=edge_list[-1],colors='black',zorder=2)
        ax_ratio.grid()

    kappa_labels = [ f'{param:.1f}' for param in coupling_parameters ]
    title  = hist_title+'\nfor '
    title += '$\kappa_{2V}='+kappa_labels[0]+'$, '
    title += '$\kappa_{\lambda}='+kappa_labels[1]+'$, '
    title += '$\kappa_{V}='+kappa_labels[2]+'$'
    fig.suptitle(title)

    ax_main.set(ylabel='Bin Weight')
    ax_main.legend(prop={'size':7})
    ax_main.grid()

    dpi=500
    kappa_string_list = [ label.replace('.','p') for label in kappa_labels ]
    kappa_string = 'cvv'+kappa_string_list[0]+'cl'+kappa_string_list[1]+'cv'+kappa_string_list[2]
    fig.tight_layout()
    figname = hist_name+'_'+kappa_string
    savedir = 'validation' if draw_verify else 'previews'
    #fig.savefig('plots/'+savedir+'/'+figname+'.png', dpi=dpi)
    fig.savefig('plots/'+savedir+'/'+figname+'.pdf', dpi=dpi)
    plt.close()



def generate_1D_pojection_scans(basis_parameters):
    #var_edges = numpy.linspace(200, 1200, 31)
    var_edges = numpy.linspace(200, 2000, 55)
    data_files = fileio_utils.read_coupling_file()
    base_events_list = fileio_utils.get_events(basis_parameters, data_files)
    base_histograms = [ fileio_utils.retrieve_reco_weights(var_edges, base_events) for base_events in base_events_list ]
    base_weights, base_errors = numpy.array(list(zip(*base_histograms)))

    num_kappa_bins = 100
    k2v_fixed, kl_fixed, kv_fixed = 1, 1, 1
    k2v_vals = numpy.linspace(-2,4,num_kappa_bins+1)
    kl_vals = numpy.linspace(-14,16,num_kappa_bins+1)

    k2v_nWeight_integral = negative_weight_map.get_Nweight_sum1D(basis_parameters, base_weights, k2v_vals, kl_fixed, kv_fixed,
            base_equations=combination_utils.full_scan_terms, which_coupling='k2v')
    draw_1D_mhh_heatmap(basis_parameters, base_weights, var_edges, k2v_vals, kl_fixed, kv_fixed,
        base_equations=combination_utils.full_scan_terms, which_coupling='k2v',
        filename='projectionscan_k2v', title_suffix=' 1D Projection Scan, Axis Integral = '+f'{k2v_nWeight_integral:.3f}')

    kl_nWeight_integral = negative_weight_map.get_Nweight_sum1D(basis_parameters, base_weights, k2v_fixed, kl_vals, kv_fixed,
            base_equations=combination_utils.full_scan_terms, which_coupling='kl')
    draw_1D_mhh_heatmap(basis_parameters, base_weights, var_edges, k2v_fixed, kl_vals, kv_fixed,
        base_equations=combination_utils.full_scan_terms, which_coupling='kl',
        filename='projectionscan_kl', title_suffix=' 1D Projection Scan, Axis Integral = '+f'{kl_nWeight_integral:.3f}')



def generate_1D9S_pojection_scans(k2v_9S_basis_tuple, vmin, vmax):
    var_edges = numpy.linspace(200, 1200, 31)
    #var_edges = numpy.linspace(200, 2000, 55)
    data_files = fileio_utils.read_coupling_file()

    numpy.set_printoptions(threshold=sys.maxsize, linewidth=230, precision=1, floatmode='fixed', suppress=True)

    num_kappa_bins = 100
    kl_fixed, kv_fixed = 1, 1
    k2v_vals = numpy.linspace(-2,4,num_kappa_bins+1)

    index_bounds = k2v_9S_basis_tuple[0]
    grid_bounds = [
        k2v_vals <= index_bounds[0],
        numpy.logical_and( index_bounds[0] < k2v_vals, k2v_vals <= index_bounds[1] ),
        index_bounds[1] < k2v_vals
    ]


    grid_list = []
    for k2v_list, boundary in zip(k2v_9S_basis_tuple[1], grid_bounds):
        basis_parameters = [ (k2v, 1, 1) for k2v in k2v_list ]
        base_events_list = fileio_utils.get_events(basis_parameters, data_files)
        base_histograms = [ fileio_utils.retrieve_reco_weights(var_edges, base_events) for base_events in base_events_list ]
        weights, errors = numpy.array(list(zip(*base_histograms)))
        reweight_vector_function = combination_utils.get_amplitude_function(basis_parameters, as_scalar=False, base_equations=combination_utils.k2v_scan_terms)
        multiplier_array_vector = reweight_vector_function(k2v_vals, kl_fixed, kv_fixed)[0]
        partial_weight_grid = sum([ multiplier_array[...,None] * w for multiplier_array, w in zip(multiplier_array_vector, weights) ])
        bounded_weight_grid = (partial_weight_grid.transpose() * boundary).transpose()
        grid_list.append(bounded_weight_grid)
    weight_grid = sum(grid_list)
    
    num_bins = len(k2v_vals) - 1
    ranges = k2v_vals[0], k2v_vals[-1], var_edges[0], var_edges[-1]
    title = 'Combined $m_{HH}$ Across 'r'$\kappa_{2V}$ Using Multi-Basis Combination'
    axis_title = r'$\kappa_{2V}$'
    #plottable_couplings = [ c[0] for c in couplings ]
    tick_vals = numpy.arange(ranges[0],ranges[1]+1,1)

    fig, ax = plt.subplots()
    im = ax.imshow(weight_grid.transpose(), cmap='viridis', extent=ranges, origin='lower', norm=matplotlib.colors.LogNorm(vmin,vmax) )
    #im = ax.imshow(weight_grid.transpose(), extent=ranges, origin='lower', cmap='viridis')
    ax.set_xticks(ticks = tick_vals)
    ax.set_xlabel(axis_title)
    ax.set_ylabel('$m_{HH}$')
    ax.grid()
    #for x in plottable_couplings: ax.vlines(x, ymin=var_edges[0], ymax=var_edges[-1], color='red')
    ax.set_aspect('auto','box')

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.11, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Bin Weight')

    #basis_table = '$\kappa_{2V}$  ,  $\kappa_{\lambda}$  ,  $\kappa_{V}$   '
    #for coupling in couplings: basis_table += '\n'+combination_utils.nice_coupling_string(coupling)
    #fig.text(.99, 1, basis_table, ha='right', va='top', fontsize='xx-small', family='monospace')

    fig.suptitle(title, fontsize=10, fontweight='bold')
    dpi = 500
    figname = 'c2v_9S_projection'
    #plt.savefig('plots/scan_maps/'+figname+'.png',dpi=dpi)
    plt.savefig('plots/scan_maps/'+figname+'.pdf',dpi=dpi)



def compare1D3S9S_reco_method(k2v_3S_basis_parameters, k2v_9S_basis_tuple):
    vmin, vmax = 1e-5, 5
    generate_1D9S_pojection_scans(k2v_9S_basis_tuple, vmin, vmax)

    #var_edges = numpy.linspace(200, 1200, 31)
    alt_var_edges = numpy.linspace(200, 1200, 31)
    var_edges = numpy.linspace(200, 2000, 55)
    #var_edges = numpy.arange(0, 2050, 50)
    num_kappa_bins = 10
    k2v_vals = numpy.linspace(-2,4,num_kappa_bins+1)
    k2v_vals_alt = numpy.linspace(-2,4,100+1)

    data_files = fileio_utils.read_coupling_file()
    k2v_3S_reweight_vector = get_amplitude_function(k2v_3S_basis_parameters, as_scalar=False, base_equations=combination_utils.k2v_scan_terms)
    k2v_3S_base_events_list = fileio_utils.get_events(k2v_3S_basis_parameters, data_files)
    k2v_3S_base_histograms = [ fileio_utils.retrieve_reco_weights(var_edges, base_events) for base_events in k2v_3S_base_events_list ]
    k2v_3S_base_weights, k2v_3S_base_errors = numpy.array(list(zip(*k2v_3S_base_histograms)))

    k2v_3S_base_histograms_alt = [ fileio_utils.retrieve_reco_weights(alt_var_edges, base_events) for base_events in k2v_3S_base_events_list ]
    k2v_3S_base_weights_alt, k2v_3S_base_errors_alt = numpy.array(list(zip(*k2v_3S_base_histograms_alt)))
    draw_1D_mhh_heatmap(k2v_3S_basis_parameters, k2v_3S_base_weights_alt, alt_var_edges, k2v_vals_alt, 1, 1,
        base_equations=combination_utils.k2v_scan_terms, which_coupling='k2v',
        filename='projectionscan_k2v_multicompare', title_suffix='Using Single Basis', vrange=(vmin, vmax))

    multibasis_list = []
    for k2v_list in k2v_9S_basis_tuple[1]:
        basis_parameters = [ (k2v, 1, 1) for k2v in k2v_list ]
        base_events_list = fileio_utils.get_events(basis_parameters, data_files)
        base_histograms = [ fileio_utils.retrieve_reco_weights(var_edges, base_events) for base_events in base_events_list ]
        weights, errors = numpy.array(list(zip(*base_histograms)))
        reweight_vector_function = combination_utils.get_amplitude_function(basis_parameters, as_scalar=False, base_equations=combination_utils.k2v_scan_terms)
        multibasis_list.append( (weights, errors, reweight_vector_function) )

    index_bounds = k2v_9S_basis_tuple[0]
    for k2v in k2v_vals:
        coupling_parameters = [k2v, 1, 1]
        k2v_combined_weights, k2v_combined_errors = reco_reweight(k2v_3S_reweight_vector, coupling_parameters, k2v_3S_base_weights, k2v_3S_base_errors)

        multibasis_index = None
        if k2v <= index_bounds[0]: multibasis_index = 0
        elif k2v <= index_bounds[1]: multibasis_index = 1
        else: multibasis_index = 2
        multibasis_weights, multibasis_errors, multibasis_reweight_vector_function = multibasis_list[multibasis_index]
        multicombined_weights, multicombined_errors = reco_reweight(multibasis_reweight_vector_function, coupling_parameters, multibasis_weights, multibasis_errors)

        view_linear_combination.plot_histogram('preview_reco_mHH_multibasis', 'NNT-Based Linear Combination:\n$m_{HH}$',
                var_edges, coupling_parameters,
                k2v_combined_weights, k2v_combined_errors,
                alt_linearly_combined_weights=multicombined_weights,
                alt_linearly_combined_errors=multicombined_errors,
                alt_label = '3-Basis Set',
                generated_label='1-Basis Equation',
                xlabel='Reconstructed $m_{HH}$ (GeV)',
        )



def compare12_reco_method(basis_parameters, k2v_basis_parameters, kl_basis_parameters, verification_parameters,
        base_equations=combination_utils.full_scan_terms, name_suffix='', title_suffix=''):

    reweight_vector = get_amplitude_function(basis_parameters, as_scalar=False, base_equations=base_equations)
    k2v_reweight_vector = get_amplitude_function(k2v_basis_parameters, as_scalar=False, base_equations=combination_utils.k2v_scan_terms)
    kl_reweight_vector = get_amplitude_function(kl_basis_parameters, as_scalar=False, base_equations=combination_utils.kl_scan_terms)

    #var_edges = numpy.linspace(200, 1200, 31)
    var_edges = numpy.linspace(200, 2000, 55)
    #var_edges = numpy.arange(0, 2050, 50)

    data_files = fileio_utils.read_coupling_file()
    base_events_list = fileio_utils.get_events(basis_parameters, data_files)
    k2v_base_events_list = fileio_utils.get_events(k2v_basis_parameters, data_files)
    kl_base_events_list = fileio_utils.get_events(kl_basis_parameters, data_files)
    verification_events_list = fileio_utils.get_events(verification_parameters, data_files)

    base_histograms = [ fileio_utils.retrieve_reco_weights(var_edges, base_events) for base_events in base_events_list ]
    base_weights, base_errors = numpy.array(list(zip(*base_histograms)))
    k2v_base_histograms = [ fileio_utils.retrieve_reco_weights(var_edges, base_events) for base_events in k2v_base_events_list ]
    k2v_base_weights, k2v_base_errors = numpy.array(list(zip(*k2v_base_histograms)))
    kl_base_histograms = [ fileio_utils.retrieve_reco_weights(var_edges, base_events) for base_events in kl_base_events_list ]
    kl_base_weights, kl_base_errors = numpy.array(list(zip(*kl_base_histograms)))

    for verification_events, coupling_parameters in zip(verification_events_list, verification_parameters):
        k2v, kl, kv = coupling_parameters
        if coupling_parameters == (1,1,1): continue
        if k2v != 1 and kl != 1: continue
        if kv != 1: continue
        alt_combined_weights, alt_combined_errors = None, None
        if k2v != 1 and kl == 1:
            alt_combined_weights, alt_combined_errors = reco_reweight(k2v_reweight_vector, coupling_parameters, k2v_base_weights, k2v_base_errors)

        if k2v == 1 and kl != 1:
            alt_combined_weights, alt_combined_errors = reco_reweight(kl_reweight_vector, coupling_parameters, kl_base_weights, kl_base_errors)

        verification_weights, verification_errors = fileio_utils.retrieve_reco_weights(var_edges, verification_events)
        combined_weights, combined_errors = reco_reweight(reweight_vector, coupling_parameters, base_weights, base_errors)

        plot_histogram('reco_mHH_1-2D_compare'+name_suffix, 'NNT-Based Linear Combination:\n$m_{HH}$'+title_suffix,
                var_edges, coupling_parameters,
                combined_weights, combined_errors,
                verification_weights, verification_errors,
                alt_linearly_combined_weights=alt_combined_weights,
                alt_linearly_combined_errors=alt_combined_errors,
                generated_label='3D Combination',
                xlabel='Reconstructed $m_{HH}$ (GeV)',
        )



def compare_bases_reco_method(basis_parameters_list, verification_parameters,
        base_equations=combination_utils.full_scan_terms, name_suffix='', title_suffix='',
        labels=('',''), is_verification=True, truth_level=False, truth_data_files=None, coupling_file=None):

    #var_edges = numpy.linspace(200, 1200, 31)
    #var_edges = numpy.arange(0, 2050, 50)
    var_edges = numpy.linspace(200, 2000, 55)

    basis_tuple_list = []
    for basis_parameters in basis_parameters_list:
        reweight_vector = get_amplitude_function(basis_parameters, as_scalar=False, base_equations=base_equations)
        if truth_level:
            data_files = fileio_utils.read_coupling_file(coupling_file='basis_files/truth_LHE_couplings_extended.dat')
            basis_files = [ truth_data_files[coupling] for coupling in basis_parameters ]
            truth_weights, truth_errors = fileio_utils.extract_lhe_truth_data(basis_files, var_edges)
            basis_tuple_list.append((truth_weights, truth_errors, reweight_vector))
        else:
            data_files = fileio_utils.read_coupling_file(coupling_file=coupling_file)
            base_events_list = fileio_utils.get_events(basis_parameters, data_files)
            base_histograms = [ fileio_utils.retrieve_reco_weights(var_edges, base_events) for base_events in base_events_list ]
            base_weights, base_errors = numpy.array(list(zip(*base_histograms)))
            basis_tuple_list.append((base_weights, base_errors, reweight_vector))

    testpoint_list = verification_parameters
    if is_verification:
        if truth_level:
            verification_files = [ data_files[key] for key in verification_parameters ]
            truth_verification_weights, truth_verification_errors = fileio_utils.extract_lhe_truth_data(verification_files, var_edges)
            testpoint_list = zip(verification_parameters, truth_verification_weights, truth_verification_errors)
        else:
            testpoint_list = []
            verification_events_list = fileio_utils.get_events(verification_parameters, data_files)
            for events, param in zip(verification_events_list, verification_parameters):
                verification_weights, verification_errors = fileio_utils.retrieve_reco_weights(var_edges, events)
                testpoint_list.append( (param, verification_weights, verification_errors) )

    for testpoint in testpoint_list:
        verification_weights, verification_errors = None, None
        if is_verification:
            coupling_parameters, verification_weights, verification_errors = testpoint
        else: 
            coupling_parameters = testpoint

        combined_tuples = []
        for base_weights, base_errors, reweight_vector in basis_tuple_list:
            combined_tuples.append( reco_reweight(reweight_vector, coupling_parameters, base_weights, base_errors) )

        if truth_level:
            name='truth_mHH_compare'+name_suffix
            title = 'Truth LHE-Based Linear Combination:\nTruth $m_{HH}$'+title_suffix
            xlabel='Truth $m_{HH}$ (GeV)'
        else:
            name = 'reco_mHH_compare'+name_suffix
            title = 'NNT-Based Linear Combination:\n$m_{HH}$'+title_suffix
            xlabel='Reconstructed $m_{HH}$ (GeV)'

        plot_histogram( name, title,
                var_edges, coupling_parameters,
                combined_tuples[0][0], combined_tuples[0][1],
                verification_weights, verification_errors,
                alt_linearly_combined_weights=combined_tuples[1][0],
                alt_linearly_combined_errors=combined_tuples[1][1],
                generated_label=labels[0],
                alt_label=labels[1],
                xlabel=xlabel,
        )



def validate_reco_method(basis_parameters, verification_parameters,
        base_equations=combination_utils.full_scan_terms, name_suffix='', title_suffix=''):

    reweight_vector = get_amplitude_function(basis_parameters, as_scalar=False, base_equations=base_equations)
    #var_edges = numpy.linspace(200, 1200, 31)
    var_edges = numpy.linspace(200, 2000, 55)
    #var_edges = numpy.arange(0, 2050, 50)

    data_files = fileio_utils.read_coupling_file()
    base_events_list = fileio_utils.get_events(basis_parameters, data_files)
    verification_events_list = fileio_utils.get_events(verification_parameters, data_files)

    base_histograms = [ fileio_utils.retrieve_reco_weights(var_edges, base_events) for base_events in base_events_list ]
    base_weights, base_errors = numpy.array(list(zip(*base_histograms)))

    for verification_events, coupling_parameters in zip(verification_events_list, verification_parameters):
        verification_weights, verification_errors = fileio_utils.retrieve_reco_weights(var_edges, verification_events)
        combined_weights, combined_errors = reco_reweight(reweight_vector, coupling_parameters, base_weights, base_errors)

        plot_histogram('reco_mHH'+name_suffix, 'NNT-Based Linear Combination:\n$m_{HH}$'+title_suffix,
                var_edges, coupling_parameters,
                combined_weights, combined_errors,
                verification_weights, verification_errors,
                xlabel='Reconstructed $m_{HH}$ (GeV)'
        )



def main():
    # Sort out command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( "--mode", required = False, default = 'reco', type=str,
        help = "Options are: 'reco' or 'reco'",) 

    args = parser.parse_args()

    data_files = fileio_utils.read_coupling_file()
    verification_parameters = list(data_files.keys())

    #pdb.set_trace()
    if args.mode == 'reco':
        validate_reco_method(_reco_basis, verification_parameters)
        #validate_reco_method( [(1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (3.0, 1.0, 1.0)] , verification_parameters)
    elif args.mode == '1D':
        generate_1D_pojection_scans(_reco_basis)
    elif args.mode == 'dual':
        compare12_reco_method( [(1.0, 1.0, 1.0), (0.5, 1.0, 1.0), (3.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 10.0, 1.0), (0.0, 0.0, 1.0)],
                [(1,1,1), (1.5,1,1), (3,1,1)],
                [(1,1,1), (1,2,1), (1,10,1)],
                verification_parameters)
    elif args.mode == 'multi':
        compare1D3S9S_reco_method( [(1,1,1), (1.5,1,1), (3,1,1)],
        #compare1D3S9S_reco_method( [(1.0, 1.0, 1.0), (1.5, 1.0, 1.0), (2.0, 1.0, 1.0)],
                ( [0,2], (
                    [1,0.5,0],
                    [2,1.5,1],
                    [3,2,1.5]
                ))
        )

    elif args.mode == 'compare':
        coupling_file = 'basis_files/nnt_coupling_file_2021Aug_test.dat'
        k2v_vals = [-1.5, 0.5, 1, 2, 3.5]
        kl_vals = [-9, -7, -3, 1, 5, 14]
        preview_couplings = []
        for k2v in k2v_vals:
            for kl in kl_vals:
                preview_couplings.append( (k2v, kl, 1) )
        compare_bases_reco_method(
            [ combination_utils.basis_full3D_2021May_minN, combination_utils.basis_full3D_2021Aug_Neo ],
            preview_couplings,
            name_suffix='_preview_2021newold',
            labels=(f'Old', f'New'), is_verification=False, truth_level=False, coupling_file=coupling_file)
    else:
        print('Mode - '+str(args.mode)+' - is not valid.')
        print('Aborting')
        exit(1)


if __name__ == '__main__': main()
