import sys
import re
import argparse
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import reweight_utils



def dump_distributions(name, mHH_edges, truth_sm_file, truth_variation_file, reco_sm_file, reco_variation_file, reco_tree):
    truth_sm_weights = reweight_utils.extract_lhe_truth_weight(truth_sm_file, mHH_edges)
    truth_variation_weights = reweight_utils.extract_lhe_truth_weight(truth_variation_file, mHH_edges)
    reco_sm_weights = reweight_utils.extract_reco_weight(reco_sm_file, mHH_edges, mhh_key='truth_mhh', unit_conversion=1/1000, tree_name=reco_tree)
    reco_variation_weights = reweight_utils.extract_reco_weight(reco_variation_file, mHH_edges, mhh_key='truth_mhh', unit_conversion=1/1000, tree_name=reco_tree)

    sm_efficiency = (reco_sm_weights / truth_sm_weights)
    variation_efficiency = (reco_variation_weights / truth_variation_weights)
    variation_efficiency[ variation_efficiency == 0 ] = float('inf')
    fractional_difference = 1 - (sm_efficiency / variation_efficiency)
    return fractional_difference



def make_title(coupling_parameters):
    kappa_labels = [ str(param) for param in coupling_parameters ]
    title  = 'k2V='+kappa_labels[0]+' '
    title += 'kLambda='+kappa_labels[1]+' '
    title += 'kV='+kappa_labels[2]
    return title


def stringify_edges(mHH_edges):
    string = str(int(mHH_edges[0])) + '-'
    for edge in mHH_edges[1:-1]:
        string += str(int(edge)) +', '
        string += str(int(edge)) +'-'
    string += str(int(mHH_edges[-1]))
    return string




def main():
    mHH_edges = numpy.linspace(250, 1250, num=10)

    truth_dir = '/home/cmilke/Documents/dihiggs/lhe_histograms/output/'
    reco_dir = '/home/cmilke/Documents/dihiggs/nano_ntuples/intermediate_trees/'
    truth_sm_file = truth_dir+'slurm_l1cvv1cv1-tree.root'
    reco_sm_file = reco_dir+'vbf4b_l1cvv1cv1_r10724.root'

    #tree_list = [b'mc_weight', b'trigger_cut', b'four_jets', b'valid_cut', b'trig_bkts_frame', b'trig_sf_frame', b'blinded', b'result', b'delta_eta_cut', b'main_result', b'sig']
    tree_list = [b'trigger_cut']
    file_list = [
        #( (1, 0 ,1 ), 'slurm_l0cvv1cv1-tree.root', 'vbf4b_l0cvv1cv1_r10724.root'),
        #( (1, 2 ,1 ), 'slurm_l2cvv1cv1-tree.root', 'vbf4b_l2cvv1cv1_r10724.root'),
        #( (1, 10,1 ), 'slurm_l10cvv1cv1-tree.root', 'vbf4b_l10cvv1cv1_r10724.root'),
        #( (0, 1 ,1 ), 'slurm_l1cvv0cv1-tree.root', 'vbf4b_l1cvv0cv1_r10724.root'),
        ( (2, 1 ,1 ), 'slurm_l1cvv2cv1-tree.root', 'vbf4b_l1cvv2cv1_r10724.root'),
    ]

    #numpy.set_printoptions(precision=2, linewidth=100, sign=' ')#, floatmode='fixed')
    for var, truth_variation_file, reco_variation_file in file_list:
        print(make_title(var)+',')
        print('mHH bin, ' + stringify_edges(mHH_edges))
        prior_array = numpy.zeros(len(mHH_edges)-1)
        for tree in tree_list:
            new_array = dump_distributions(tree.decode(), mHH_edges, truth_sm_file, truth_dir+truth_variation_file, reco_sm_file, reco_dir+reco_variation_file, tree)
            difference = new_array - prior_array
            valueString = ''
            #for d in difference: valueString += f'{d:4.4f}, '
            for d in new_array: valueString += f'{d:4.4f}, '
            print(tree.decode() + ', ' + valueString)
            #print()
            prior_array = new_array
        print()
        #print('\n\n-------------\n\n')





if __name__ == '__main__': main()
