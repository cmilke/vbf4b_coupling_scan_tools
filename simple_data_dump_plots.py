import sys
import re
import argparse
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import reweight_utils



def plot_histograms(hist_name, label, edge_list,
        truth_sm_weights, truth_variation_weights,
        reco_sm_weights, reco_variation_weights,
        range_specs=None, Rpower = 5 ):

    print('Plotting '+hist_name+'...')

    if type(range_specs) != type(None):
        range_specs = numpy.digitize(range_specs, edge_list, right=True)
        truth_sm_weights        = truth_sm_weights[range_specs[0]:range_specs[1]]
        truth_variation_weights = truth_variation_weights[range_specs[0]:range_specs[1]]
        reco_sm_weights         = reco_sm_weights[range_specs[0]:range_specs[1]]
        reco_variation_weights  = reco_variation_weights[range_specs[0]:range_specs[1]]
        edge_list = edge_list[range_specs[0]:range_specs[1]+1]

    grid_ratios = [4,1,4]
    if len(edge_list) > 15: grid_ratios = [100,1,100]
    fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True,
            gridspec_kw={'height_ratios':grid_ratios,'width_ratios':grid_ratios})


    axes[0][0].hist( edge_list[:-1], weights=truth_sm_weights, bins=edge_list, color='green')
    axes[0][2].hist( edge_list[:-1], weights=truth_variation_weights, bins=edge_list, color='blue')
    axes[2][0].hist( edge_list[:-1], weights=reco_sm_weights, bins=edge_list, color='red')
    axes[2][2].hist( edge_list[:-1], weights=reco_variation_weights, bins=edge_list, color='purple', label='Generated')


    truth_transform = truth_variation_weights / truth_sm_weights
    truth_transform_str = [ f'{t:.1f}' for t in truth_transform ]
    truth_transform_str = numpy.array(truth_transform_str)[numpy.newaxis].transpose()
    axes[0][1].table( cellText = truth_transform_str, loc='center' )
    axes[0][1].axis('tight')
    axes[0][1].axis('off')


    #print(reco_variation_weights.sum() / (reco_sm_weights*truth_transform).sum() )
    axes[2][2].hist( edge_list[:-1], weights=(reco_sm_weights*truth_transform), bins=edge_list,
            #fc=(0,0,0,0), edgecolor='cyan', ls='solid', linewidth=3, label='Reweighted')
            fc=(0,0.5,.8,0.4), label='Reweighted')

    sm_efficiency = reco_sm_weights / truth_sm_weights * 10**(Rpower)
    sm_efficiency_str = [ f'{s:.2f}' for s in sm_efficiency ]
    sm_efficiency_str = numpy.array(sm_efficiency_str)[numpy.newaxis]
    axes[1][0].table( cellText = sm_efficiency_str, loc='center' )
    axes[1][0].axis('tight')
    axes[1][0].axis('off')

    variation_efficiency = reco_variation_weights / truth_variation_weights * 10**(Rpower)
    variation_efficiency_str = [ f'{v:.2f}' for v in variation_efficiency ]
    variation_efficiency_str = numpy.array(variation_efficiency_str)[numpy.newaxis]
    axes[1][2].table( cellText = variation_efficiency_str, loc='center' )
    axes[1][2].axis('tight')
    axes[1][2].axis('off')

    #efficiency_ratio = sm_efficiency / variation_efficiency
    #efficiency_ratio_str = [ f'{v:.1f}' for v in efficiency_ratio ]
    #efficiency_ratio_str = numpy.array(efficiency_ratio_str)[numpy.newaxis]
    #axes[1][1].table( cellText = efficiency_ratio_str, loc='center' )
    #axes[1][1].axis('tight')
    #axes[1][1].axis('off')

    axes[2][0].set_xlabel('$m_{HH}$')
    axes[2][2].set_xlabel('$m_{HH}$')
    axes[0][0].set_ylabel('Cross Section')
    axes[2][0].set_ylabel('Cross Section ('r'$\times 10^{'f'{-Rpower}''}$'')')

    grid_skip = 1
    if len(edge_list) > 10: grid_skip = 2
    axes[0][0].set_xticks(edge_list[::grid_skip])
    axes[0][2].set_xticks(edge_list[::grid_skip])
    axes[2][0].set_xticks(edge_list[::grid_skip])
    axes[2][2].set_xticks(edge_list[::grid_skip])
    axes[2][0].set_xticklabels([int(l) for l in axes[2][0].get_xticks()], fontsize=4, rotation=45)
    axes[2][2].set_xticklabels([int(l) for l in axes[2][2].get_xticks()], fontsize=4, rotation=45)


    num_yticks = 8
    axes[0][0].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(max(truth_sm_weights)/num_yticks))
    axes[0][2].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(max(truth_variation_weights)/num_yticks))
    axes[2][0].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(max(reco_sm_weights)/num_yticks))
    axes[2][2].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(max(reco_variation_weights)/num_yticks))
    axes[0][0].set_yticklabels([int(l) for l in axes[0][0].get_yticks()], fontsize=6)
    axes[0][2].set_yticklabels([int(l) for l in axes[0][2].get_yticks()], fontsize=6)
    axes[2][0].set_yticklabels([int(l*10**Rpower) for l in axes[2][0].get_yticks()], fontsize=6)
    axes[2][2].set_yticklabels([int(l*10**Rpower) for l in axes[2][2].get_yticks()], fontsize=6)

    #axes[2][2].set_yscale('log')


    axes[0][0].grid()
    axes[0][2].grid()
    axes[2][0].grid()
    axes[2][2].grid()

    axes[0][0].set_title('Truth SM', fontsize='x-small')
    axes[0][2].set_title('Truth '+label, fontsize='x-small')
    axes[2][0].set_title('Post-Selection SM', fontsize='x-small')
    axes[2][2].set_title('Post-Selection '+label, fontsize='x-small')
    axes[0][1].set_title('Transform', fontsize='x-small')
    axes[1][0].set_title('SM Acceptance ('r'$\times 10^{'f'{-Rpower}''}$'')', fontsize='x-small')
    axes[1][2].set_title(label+' Acceptance ('r'$\times 10^{'f'{-Rpower}''}$'')', fontsize='x-small')

    axes[2][2].legend()
    axes[2][2].legend(prop={'size':5})

    axes[1][1].set_axis_off()
    axes[1][1].set_axis_off()
    axes[2][1].set_axis_off()
    if len(edge_list) > 15:
        axes[0][1].clear()
        axes[1][0].clear()
        axes[1][2].clear()
        axes[0][1].set_axis_off()
        axes[1][0].set_axis_off()
        axes[1][2].set_axis_off()

    fig.tight_layout()
    dpi=500
    fig.savefig('plots/dump/'+hist_name+'.png', dpi=dpi)
    plt.close()





def dump_distributions(name, truth_sm_file, truth_variation_file, reco_sm_file, reco_variation_file, ggF=False):
    #minM = 250
    #maxM = 6000
    #displayM = 1250
    #displayBins = 10
    #numBins = int((maxM/displayM)*(displayBins+1))
    #mHH_edges = numpy.linspace(minM, maxM, num=numBins)
    mHH_edges = numpy.linspace(250, 1000, num=100)
    truth_sm_weights = reweight_utils.extract_lhe_truth_weight(truth_sm_file, mHH_edges, force_cap=10000000)
    truth_variation_weights = reweight_utils.extract_lhe_truth_weight(truth_variation_file, mHH_edges, force_cap=10000000)
    #print(truth_variation_weights.sum()/truth_sm_weights.sum())
    filter_vbf = not ggF
    reco_sm_weights = reweight_utils.extract_reco_weight(reco_sm_file, mHH_edges, key='truth_mhh', unit_conversion=1/1000, filter_vbf=filter_vbf)
    reco_variation_weights = reweight_utils.extract_reco_weight(reco_variation_file, mHH_edges, key='truth_mhh', unit_conversion=1/1000, filter_vbf=filter_vbf)

    if ggF:
        search = re.search('.*l(.*)', name)
        label = 'ggF [''$\kappa_{\lambda}='+search.group(0)+'$]'
    else:
        search = re.search('.*l(.*)cvv(.*)cv([^_]*)', name)
        kappa = [ search.group(n).replace('p','.') for n in range(1,4) ]
        label  = '$\kappa_{\lambda}='+kappa[0]+'$, '
        label += '$\kappa_{2V}='+kappa[1]+'$, '
        label += '$\kappa_{V}='+kappa[2]+'$'
        label = '['+label+']'

    plot_histograms(name, label, mHH_edges,
             truth_sm_weights, truth_variation_weights,
             reco_sm_weights, reco_variation_weights,
             #range_specs=(minM,displayM)
             range_specs=None, Rpower = 6 if ggF else 5
    )



def main():
    # VBF Testing
    truth_sm_file = '/home/cmilke/Documents/dihiggs/lhe_histograms/output/slurm_l1cvv1cv1-tree.root'
    reco_sm_file = '/home/cmilke/Documents/dihiggs/nano_ntuples/truth_mhh/vbf4b_l1cvv1cv1_r10724.root'

    truth_variation_file = '/home/cmilke/Documents/dihiggs/lhe_histograms/output/slurm_l10cvv1cv1-tree.root'
    reco_variation_file = '/home/cmilke/Documents/dihiggs/nano_ntuples/intermediate_trees/vbf4b_l10cvv1cv1_r10724.root'
    dump_distributions('l10cvv1cv1', truth_sm_file, truth_variation_file, reco_sm_file, reco_variation_file)

    truth_variation_file = '/home/cmilke/Documents/dihiggs/lhe_histograms/output/slurm_l1cvv2cv1-tree.root'
    reco_variation_file = '/home/cmilke/Documents/dihiggs/nano_ntuples/truth_mhh/vbf4b_l1cvv2cv1_r10724.root'
    dump_distributions('l1cvv2cv1', truth_sm_file, truth_variation_file, reco_sm_file, reco_variation_file)

    truth_variation_file = '/home/cmilke/Documents/dihiggs/lhe_histograms/output/slurm_l1cvv0cv1-tree.root'
    reco_variation_file = '/home/cmilke/Documents/dihiggs/nano_ntuples/truth_mhh/vbf4b_l1cvv0cv1_r10724.root'
    dump_distributions('l1cvv0cv1', truth_sm_file, truth_variation_file, reco_sm_file, reco_variation_file)

    truth_variation_file = '/home/cmilke/Documents/dihiggs/lhe_histograms/output/slurm_l0cvv1cv1-tree.root'
    reco_variation_file = '/home/cmilke/Documents/dihiggs/nano_ntuples/truth_mhh/vbf4b_l0cvv1cv1_r10724.root'
    dump_distributions('l0cvv1cv1', truth_sm_file, truth_variation_file, reco_sm_file, reco_variation_file)

    truth_variation_file = '/home/cmilke/Documents/dihiggs/lhe_histograms/output/slurm_l2cvv1cv1-tree.root'
    reco_variation_file = '/home/cmilke/Documents/dihiggs/nano_ntuples/truth_mhh/vbf4b_l2cvv1cv1_r10724.root'
    dump_distributions('l2cvv1cv1', truth_sm_file, truth_variation_file, reco_sm_file, reco_variation_file)


    # ggF Testing
    #truth_sm_file = '/home/cmilke/Documents/dihiggs/lhe_histograms/output/ggF_lhe_kappa_1_collection-tree.root'
    #reco_sm_file = '/home/cmilke/Documents/dihiggs/nano_ntuples/ggF_variations/2018_kL1_nonres.root'

    #truth_variation_file = '/home/cmilke/Documents/dihiggs/lhe_histograms/output/ggF_lhe_kappa_10_collection-tree.root'
    #reco_variation_file = '/home/cmilke/Documents/dihiggs/nano_ntuples/ggF_variations/2018_kL10_nonres.root'
    #dump_distributions('ggF_l10', truth_sm_file, truth_variation_file, reco_sm_file, reco_variation_file, ggF=True)

    #truth_variation_file = '/home/cmilke/Documents/dihiggs/lhe_histograms/output/ggF_lhe_kappa_20_collection-tree.root'
    #reco_variation_file = '/home/cmilke/Documents/dihiggs/nano_ntuples/ggF_variations/2018_kL20_nonres.root'
    #dump_distributions('ggF_l20', truth_sm_file, truth_variation_file, reco_sm_file, reco_variation_file, ggF=True)

    #truth_variation_file = '/home/cmilke/Documents/dihiggs/lhe_histograms/output/ggF_lhe_kappa_0_collection-tree.root'
    #reco_variation_file = '/home/cmilke/Documents/dihiggs/nano_ntuples/ggF_variations/2018_kL0_nonres.root'
    #dump_distributions('ggF_l0', truth_sm_file, truth_variation_file, reco_sm_file, reco_variation_file, ggF=True)


if __name__ == '__main__': main()
