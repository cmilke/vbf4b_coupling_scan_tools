import sys
import re
import argparse
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import reweight_utils



def get_acceptance(tree, var_edges, truth_variation_file, reco_variation_file, truth_kin_vars, reco_kin_vars):
    truth_variation_weights = reweight_utils.extract_lhe_truth_dual_weight(truth_variation_file, var_edges=var_edges, kin_vars=truth_kin_vars)
    reco_variation_weights = reweight_utils.extract_reco_dual_weight(reco_variation_file, kin_vars=reco_kin_vars, var_edges=var_edges, unit_conversion=(1/1000,1), tree_name=tree)
    truth_variation_weights[ truth_variation_weights == 0 ] = float('inf')
    acceptance = reco_variation_weights / truth_variation_weights
    #print(truth_variation_weights)
    #print()
    #print('***')
    #print(reco_variation_weights)
    #print('***')
    return acceptance


def dump_acceptances(sm_acceptance, variation_acceptance):
    #variation_acceptance[ variation_acceptance == 0 ] = float('inf')
    fractional_difference = 1 - (sm_acceptance / variation_acceptance)
    #fractional_difference[ variation_acceptance == sm_acceptance ] = 0
    print(fractional_difference)
    print('\n---------------\n')



def make_title(coupling_parameters):
    kappa_labels = [ str(param) for param in coupling_parameters ]
    title  = 'k2V='+kappa_labels[0]+' '
    title += 'kLambda='+kappa_labels[1]+' '
    title += 'kV='+kappa_labels[2]
    return title


def main():
    tree = b'sig'
    mHH_edges = numpy.linspace(250, 1250, num=11)

    #alt_edges = numpy.linspace(3, 10, num=11)
    #truth_kin_vars = ('HH_m','jj_eta')
    #reco_kin_vars = ('truth_mhh','vbf_dEtajj')
    #reco_kin_vars = ('m_hh','vbf_dEta')

    #alt_edges = numpy.linspace(0, 600, num=10)
    #truth_kin_vars = ('HH_m','jj_pT')
    #reco_kin_vars = ('truth_mhh','vbf_pTvecsum')

    alt_edges = numpy.linspace(1000, 4000, num=11)
    truth_kin_vars = ('HH_m','jj_M')
    reco_kin_vars = ('truth_mhh','vbf_mjj')

    var_edges=(mHH_edges, alt_edges)

    truth_dir = '/home/cmilke/Documents/dihiggs/lhe_histograms/output/'
    reco_dir = '/home/cmilke/Documents/dihiggs/nano_ntuples/truth_mhh/'
    #reco_dir = '/home/cmilke/Documents/dihiggs/nano_ntuples/apr20/'

    truth_sm_file = 'slurm_l1cvv1cv1-tree.root'
    reco_sm_file = 'vbf4b_l1cvv1cv1_r10724.root'
    #reco_sm_file = 'ntuples_MC16e_VBF-HH-bbbb_cl1_cvv1_cv1.root'

    file_list = [
        #( (1, 1 ,1 ), 'slurm_l1cvv1cv1-tree.root', 'vbf4b_l1cvv1cv1_r10724.root'),
        #( (1, 0 ,1 ), 'slurm_l0cvv1cv1-tree.root', 'vbf4b_l0cvv1cv1_r10724.root'),
        ( (1, 2 ,1 ), 'slurm_l2cvv1cv1-tree.root', 'vbf4b_l2cvv1cv1_r10724.root'),
        ( (0, 1 ,1 ), 'slurm_l1cvv0cv1-tree.root', 'vbf4b_l1cvv0cv1_r10724.root'),
        ( (1, 10,1 ), 'slurm_l10cvv1cv1-tree.root', 'vbf4b_l10cvv1cv1_r10724.root'),
        ( (2, 1 ,1 ), 'slurm_l1cvv2cv1-tree.root', 'vbf4b_l1cvv2cv1_r10724.root'),
        #( (1, 10 ,1 ), 'slurm_l10cvv1cv1-tree.root', 'ntuples_MC16e_VBF-HH-bbbb_cl10_cvv1_cv1.root'),
        #( (2, 1 ,1 ), 'slurm_l1cvv2cv1-tree.root', 'ntuples_MC16e_VBF-HH-bbbb_cl1_cvv2_cv1.root')
    ]

    numpy.set_printoptions(precision=2, linewidth=400, sign=' ', floatmode='fixed')
    for var, truth_variation_file, reco_variation_file in file_list:
        print(make_title(var)+',')
        sm_acceptance = get_acceptance(tree, var_edges, truth_dir+truth_sm_file, reco_dir+reco_sm_file, truth_kin_vars, reco_kin_vars)
        variation_acceptance = get_acceptance(tree, var_edges, truth_dir+truth_variation_file, reco_dir+reco_variation_file, truth_kin_vars, reco_kin_vars)
        dump_acceptances(sm_acceptance, variation_acceptance)





if __name__ == '__main__': main()
