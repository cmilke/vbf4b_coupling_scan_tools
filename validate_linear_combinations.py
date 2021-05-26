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



def draw_1D_mhh_heatmap(couplings, weights, var_edges, k2v_vals, kl_vals, kv_vals,
        base_equations=None, which_coupling=None, filename='test', title_suffix=None):
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
    im = ax.imshow(weight_grid.transpose(), cmap='viridis', extent=ranges, origin='lower', norm=matplotlib.colors.LogNorm() )
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
    plt.savefig('plots/scan_maps/'+figname+'.png',dpi=dpi)
    plt.savefig('plots/.scan_maps/'+figname+'.pdf',dpi=dpi)



def plot_histogram(hist_name, hist_title, edge_list, coupling_parameters,
        linearly_combined_weights, linearly_combined_errors,
        verification_weights, verification_errors,
        alt_linearly_combined_weights=None, alt_linearly_combined_errors=None,
        range_specs=None, xlabel='Truth $m_{HH}$ (GeV)', normalize=False,
        generated_label='Linear Combination', generated_color='blue'):

    print('Plotting '+hist_name+' for ' + str(coupling_parameters))
    fig, (ax_main, ax_ratio) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios':[4,1]} )


    draw_alt_comb = type(alt_linearly_combined_weights) != type(None) and type(alt_linearly_combined_errors) != type(None)

    if type(range_specs) != type(None):
        linearly_combined_weights = linearly_combined_weights[range_specs[0]:range_specs[1]]
        linearly_combined_errors = linearly_combined_errors[range_specs[0]:range_specs[1]]
        verification_weights = verification_weights[range_specs[0]:range_specs[1]]
        verification_errors = verification_errors[range_specs[0]:range_specs[1]]
        edge_list = edge_list[range_specs[0]:range_specs[1]+1]

    if normalize:
        linear_normalization = linearly_combined_weights.sum()
        verification_normalization = verification_weights.sum()

        linearly_combined_weights /= linear_normalization
        linearly_combined_errors /= linear_normalization
        verification_weights /= verification_normalization
        verification_errors /= verification_normalization


    xpositions = 0.5*(edge_list[1:]+edge_list[:-1])
    counts, bins, points = ax_main.errorbar( xpositions, linearly_combined_weights,
        yerr=linearly_combined_errors, label=generated_label,
        marker='+', markersize=2, capsize=2, color=generated_color, linestyle='none', linewidth=1, zorder=3)

    if draw_alt_comb:
        alt_counts, alt_bins, alt_points = ax_main.errorbar( xpositions, alt_linearly_combined_weights,
            yerr=alt_linearly_combined_errors, label='1D Combination',
            marker='x', markersize=2, capsize=2, color='purple', linestyle='none', linewidth=1, zorder=3)

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

    #rcounts, rbins, rpoints = ax_ratio.errorbar( xpositions, linearly_combined_weights/verification_weights,
    #    yerr=linearly_combined_errors/verification_weights, label='MC Statistical Error check',
    #    marker='+', markersize=2, capsize=2, color=generated_color, linestyle='none', linewidth=1, zorder=3)
    #ax_ratio.set(ylabel=r'$\frac{generated}{lin. combination}$', xlabel=xlabel)
    #ax_ratio.set_ylim([0.5,1.5])
    #ax_ratio.set_yticks(ticks=[0.5,1,1.5])
    #zero_line = ax_ratio.hlines(1,xmin=edge_list[0],xmax=edge_list[-1],colors='black',zorder=2)

    kappa_labels = [ str(param) for param in coupling_parameters ]
    title  = hist_title+'\nfor '
    title += '$\kappa_{2V}='+kappa_labels[0]+'$, '
    title += '$\kappa_{\lambda}='+kappa_labels[1]+'$, '
    title += '$\kappa_{V}='+kappa_labels[2]+'$'
    fig.suptitle(title)

    ax_main.set(ylabel='Bin Weight')
    ax_main.legend(prop={'size':7})
    ax_main.grid()
    ax_ratio.grid()

    dpi=500
    kappa_string_list = [ label.replace('.','p') for label in kappa_labels ]
    kappa_string = 'cvv'+kappa_string_list[0]+'cl'+kappa_string_list[1]+'cv'+kappa_string_list[2]
    fig.tight_layout()
    figname = hist_name+'_'+kappa_string
    fig.savefig('plots/validation/'+figname+'.png', dpi=dpi)
    fig.savefig('plots/.validation/'+figname+'.pdf', dpi=dpi)
    plt.close()



def generate_1D_pojection_scans(basis_parameters):
    #var_edges = numpy.linspace(200, 1200, 31)
    var_edges = numpy.linspace(200, 2000, 55)
    data_files = fileio_utils.read_coupling_file(fileio_utils.coupling_file)
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



def compare12_reco_method(basis_parameters, k2v_basis_parameters, kl_basis_parameters, verification_parameters,
        base_equations=combination_utils.full_scan_terms, name_suffix='', title_suffix=''):

    reweight_vector = get_amplitude_function(basis_parameters, as_scalar=False, base_equations=base_equations)
    k2v_reweight_vector = get_amplitude_function(k2v_basis_parameters, as_scalar=False, base_equations=combination_utils.k2v_scan_terms)
    kl_reweight_vector = get_amplitude_function(kl_basis_parameters, as_scalar=False, base_equations=combination_utils.kl_scan_terms)

    #var_edges = numpy.linspace(200, 1200, 31)
    var_edges = numpy.linspace(200, 2000, 55)
    #var_edges = numpy.arange(0, 2050, 50)

    data_files = fileio_utils.read_coupling_file(fileio_utils.coupling_file)
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



def validate_reco_method(basis_parameters, verification_parameters,
        base_equations=combination_utils.full_scan_terms, name_suffix='', title_suffix=''):
    reweight_vector = get_amplitude_function(basis_parameters, as_scalar=False, base_equations=base_equations)
    #var_edges = numpy.linspace(200, 1200, 31)
    var_edges = numpy.linspace(200, 2000, 55)
    #var_edges = numpy.arange(0, 2050, 50)

    data_files = fileio_utils.read_coupling_file(fileio_utils.coupling_file)
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

    data_files = fileio_utils.read_coupling_file(fileio_utils.coupling_file)
    verification_parameters = list(data_files.keys())

    #pdb.set_trace()
    if args.mode == 'reco':
        validate_reco_method(_reco_basis, verification_parameters)
        #validate_reco_method( [(1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (3.0, 1.0, 1.0)] , verification_parameters)
    elif args.mode == '1D':
        generate_1D_pojection_scans(_reco_basis)
    elif args.mode == 'dual':
        compare12_reco_method(_reco_basis,
                [(1,1,1), (0.5,1,1), (1.5,1,1)],
                [(1,1,1), (1,2,1), (1,10,1)],
                verification_parameters)
    else:
        print('Mode - '+str(args.mode)+' - is not valid.')
        print('Aborting')
        exit(1)


if __name__ == '__main__': main()
