import sys
import math
import re
import itertools
import argparse
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import reweight_utils


def get_acceptance(tree, truth_variation_file, reco_variation_file, truth_vars, reco_vars, var_edges):
    truth_variation_weights = reweight_utils.extract_lhe_truth_dual_weight(truth_variation_file, var_edges=var_edges, kin_vars=truth_vars)
    reco_variation_weights = reweight_utils.extract_reco_dual_weight(reco_variation_file, kin_vars=reco_vars, var_edges=var_edges, tree_name=tree)
    #print(truth_variation_weights)
    #print()
    #print(reco_variation_weights)
    #print()
    truth_variation_weights[ truth_variation_weights == 0 ] = float('inf')
    acceptance = reco_variation_weights / truth_variation_weights
    #print(acceptance)
    #print('-------------\n\n')

    return acceptance


def plot_acceptance_diffs(sub_ax, base_acceptance, variation_acceptance, key, var_edges):
    variation_acceptance[ variation_acceptance == 0.0 ] = float('inf')
    fractional_difference = abs( 1 - (base_acceptance / variation_acceptance) )
    fractional_difference[ base_acceptance == 0.0 ] = float('Nan')
    fractional_difference[ variation_acceptance == float('inf') ] = float('Nan')
    ranges = var_edges[0][0], var_edges[0][-1], var_edges[1][0], var_edges[1][-1]
    im = sub_ax.imshow(fractional_difference.transpose(), vmin=0, vmax=1, extent=ranges, origin='lower')
    sub_ax.set_aspect('auto', 'box')
    return im


def make_title(coupling_parameters):
    kappa_labels = [ str(param) for param in coupling_parameters ]
    title  = '$k_{2V}='+kappa_labels[0]+'$ '
    title += '$k_{\lambda}='+kappa_labels[1]+'$ '
    title += '$k_{V}='+kappa_labels[2]+'$'
    return title


def make_multiplot(output, file_list, truth_dir, reco_dir, truth_base_file, reco_base_file, var_dict, keys, tree):
    labels, truth_vars, reco_vars, var_edges = zip(*[var_dict[k] for k in keys])


    # Work out how to arrange mini-plots
    num_plots = len(file_list)
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

    for index, (var, truth_variation_file, reco_variation_file) in enumerate(file_list):
        i,j = numpy.unravel_index( index, (num_rows, num_columns) )
        sub_ax = ax_array[j] if num_rows == 1 else ax_array[i][j]

        print(var)
        base_acceptance = get_acceptance(tree, truth_dir+truth_base_file, reco_dir+reco_base_file, truth_vars, reco_vars, var_edges)
        variation_acceptance = get_acceptance(tree, truth_dir+truth_variation_file, reco_dir+reco_variation_file, truth_vars, reco_vars, var_edges)
        im = plot_acceptance_diffs(sub_ax, base_acceptance, variation_acceptance, keys, var_edges)
        sub_ax.set_title(make_title(var), fontsize='small' )

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.text(0.5, 0.04, labels[0], ha='center')
    fig.text(0.04, 0.5, labels[1], va='center', rotation='vertical')
    fig.suptitle('$|\Delta A / A_{var}|$ for Various Couplings, Parameterized in '+labels[0]+' and '+labels[1], fontsize='medium')
    #plt.show()
    dpi = 500
    output.savefig()
    plt.close()


def main():
    tree = b'sig'

    base_n_bins = 11
    var_dict = {
        'M_hh':    [ '$M_{hh}$', 'HH_m', 'truth_mhh', numpy.linspace(250, 1250, num=base_n_bins) ],
        'Deta_hh': [ '$\Delta \eta_{hh}$', 'HH_dEta', 'dEta_hh', numpy.linspace(0, 4, num=base_n_bins) ],
        'pT_hh':   [ '$p_{T,hh}$', 'HH_pt', 'pt_hh', numpy.linspace(0, 500, num=base_n_bins) ],
        'M_jj':    [ '$M_{jj}$', 'jj_M', 'vbf_mjj', numpy.linspace(1000, 4000, num=base_n_bins) ],
        'Deta_jj': [ '$\Delta \eta_{jj}$', 'jj_Deta', 'vbf_dEtajj', numpy.linspace(3, 5, num=base_n_bins) ],
        'pT_jj':   [ '$p_{T,jj}$', 'jj_pT', 'jj_pTvecsum', numpy.linspace(0, 500, num=base_n_bins) ],
        'pT_sum':  [ '$p_{T}$ Sum, jj', 'ptsumjj', 'ptsumjj', numpy.linspace(50, 500, num=base_n_bins) ],
    }

    truth_dir = '/home/cmilke/Documents/dihiggs/lhe_histograms/output/'
    reco_dir = '/home/cmilke/Documents/dihiggs/nano_ntuples/truth_mhh/'
    #reco_dir = '/home/cmilke/Documents/dihiggs/nano_ntuples/apr20/'

    truth_base_file = 'slurm_l1cvv1cv1-tree.root'
    reco_base_file = 'vbf4b_l1cvv1cv1_r10724.root'
    #reco_base_file = 'ntuples_MC16e_VBF-HH-bbbb_cl1_cvv1_cv1.root'

    file_list = [
        #( (1, 1 ,1 ), 'slurm_l1cvv1cv1-tree.root', 'vbf4b_l1cvv1cv1_r10724.root'),
        ( (1, 0 ,1 ), 'slurm_l0cvv1cv1-tree.root', 'vbf4b_l0cvv1cv1_r10724.root'),
        #( (1, 2 ,1 ), 'slurm_l2cvv1cv1-tree.root', 'vbf4b_l2cvv1cv1_r10724.root'),
        ( (0, 1 ,1 ), 'slurm_l1cvv0cv1-tree.root', 'vbf4b_l1cvv0cv1_r10724.root'),
        ( (1, 10,1 ), 'slurm_l10cvv1cv1-tree.root', 'vbf4b_l10cvv1cv1_r10724.root'),
        ( (2, 1 ,1 ), 'slurm_l1cvv2cv1-tree.root', 'vbf4b_l1cvv2cv1_r10724.root'),

        #( (1, 10 ,1 ), 'slurm_l10cvv1cv1-tree.root', 'ntuples_MC16e_VBF-HH-bbbb_cl10_cvv1_cv1.root'),
        #( (2, 1 ,1 ), 'slurm_l1cvv2cv1-tree.root', 'ntuples_MC16e_VBF-HH-bbbb_cl1_cvv2_cv1.root')
    ]

    output = PdfPages('plots/reweight_exhaustion/all_var_combinations.pdf')
    numpy.set_printoptions(precision=2, linewidth=400, sign=' ', floatmode='fixed')
    for keys in itertools.combinations(var_dict.keys(), 2):
        print(keys)
        print('--------')
        make_multiplot(output, file_list, truth_dir, reco_dir, truth_base_file, reco_base_file, var_dict, keys, tree)
        print()
    output.close()



if __name__ == '__main__': main()
