import sys
import re
import argparse
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import reweight_utils



def get_acceptance(var_edges, lhe_file, reco_variation_file, keys):
    truth_key, reco_key = keys
    truth_variation_weights =  reweight_utils.extract_lhe_truth_weight(lhe_file, var_edges, key=truth_key, force_cap = None)
    reco_variation_weights = reweight_utils.extract_reco_weight(reco_variation_file, var_edges, key=reco_key)

    truth_variation_weights[ truth_variation_weights == 0 ] = float('inf')
    acceptance = reco_variation_weights / truth_variation_weights
    average_acceptance = numpy.average(acceptance)
    print(average_acceptance)
    relative_difference = (acceptance - average_acceptance) / average_acceptance
    print(relative_difference)
    print(sum(abs(relative_difference)))
    return relative_difference


def dump_acceptances(sm_acceptance, variation_acceptance):
    #variation_acceptance[ variation_acceptance == 0 ] = float('inf')
    #fractional_difference = 1 - (sm_acceptance / variation_acceptance)
    #fractional_difference[ variation_acceptance == sm_acceptance ] = 0
    #print(fractional_difference)
    print('\n---------------\n')



def make_title(coupling_parameters):
    kappa_labels = [ str(param) for param in coupling_parameters ]
    title  = 'k2V='+kappa_labels[0]+' '
    title += 'kLambda='+kappa_labels[1]+' '
    title += 'kV='+kappa_labels[2]
    return title


def main():
    # Mhh
    edges = numpy.linspace(250, 1250, num=11)
    keys = ('HH_m','truth_mhh')

    # pt hh
    #edges = numpy.linspace(0, 500, num=21)
    #keys = ('HH_pt','pt_hh')

    # Delta eta jj
    #edges = numpy.linspace(3, 10, num=11)
    #keys = ('jj_eta', 'vbf_dEtajj')

    # pt jj
    #edges = numpy.linspace(0, 600, num=10)
    #keys = ('jj_pT', 'vbf_pTvecsum')

    # mjj
    #edges = numpy.linspace(1000, 4000, num=11)
    #keys = ('jj_M', 'vbf_mjj')


    truth_dir = '/home/cmilke/Documents/dihiggs/lhe_histograms/output/'
    reco_dir = '/home/cmilke/Documents/dihiggs/nano_ntuples/truth_mhh/'
    reco_dir = '/home/cmilke/Documents/dihiggs/nano_ntuples/ggF_variations/'
    #reco_dir = '/home/cmilke/Documents/dihiggs/nano_ntuples/apr20/'

    #truth_sm_file = 'slurm_l1cvv1cv1-tree.root'
    #reco_sm_file = 'vbf4b_l1cvv1cv1_r10724.root'
    #reco_sm_file = 'ntuples_MC16e_VBF-HH-bbbb_cl1_cvv1_cv1.root'

    file_list = [
        ( (1), 'slurm_l1cvv1cv1-tree.root' , '2018_kL1_nonres.root'),
        ( (10), 'ggF_lhe_kappa_1_collection-tree.root', '2018_kL10_nonres.root'),
        #( (1, 1 ,1 ), 'slurm_l1cvv1cv1-tree.root', 'vbf4b_l1cvv1cv1_r10724.root'),
        #( (1, 0 ,1 ), 'slurm_l0cvv1cv1-tree.root', 'vbf4b_l0cvv1cv1_r10724.root'),
        #( (1, 2 ,1 ), 'slurm_l2cvv1cv1-tree.root', 'vbf4b_l2cvv1cv1_r10724.root'),
        #( (0, 1 ,1 ), 'slurm_l1cvv0cv1-tree.root', 'vbf4b_l1cvv0cv1_r10724.root'),
        #( (1, 10,1 ), 'slurm_l10cvv1cv1-tree.root', 'vbf4b_l10cvv1cv1_r10724.root'),
        #( (2, 1 ,1 ), 'slurm_l1cvv2cv1-tree.root', 'vbf4b_l1cvv2cv1_r10724.root'),
        #( (1, 10 ,1 ), 'slurm_l10cvv1cv1-tree.root', 'ntuples_MC16e_VBF-HH-bbbb_cl10_cvv1_cv1.root'),
        #( (2, 1 ,1 ), 'slurm_l1cvv2cv1-tree.root', 'ntuples_MC16e_VBF-HH-bbbb_cl1_cvv2_cv1.root')
    ]

    numpy.set_printoptions(precision=2, linewidth=400, sign=' ', floatmode='fixed')
    for var, truth_variation_file, reco_variation_file in file_list:
        #print(make_title(var)+',')
        print(f'\nggF klambda = {var}')
        variation_acceptance = get_acceptance(edges, truth_dir+truth_variation_file, reco_dir+reco_variation_file, keys)
        #print(variation_acceptance)
        #dump_acceptances(sm_acceptance, variation_acceptance)





if __name__ == '__main__': main()
