import sys
import argparse
import math
import numpy
import statistics
import itertools
import pickle
import matplotlib
from matplotlib import pyplot as plt

import fileio_utils, combination_utils, negative_weight_map
#import pdb

def main():
    numpy.set_printoptions(precision=2, linewidth=400, threshold=100, sign=' ')
    
    var_edges = numpy.linspace(200, 1200, 31)
    kv_val = 1.0
    #num_bins = 21
    #k2v_val_range = numpy.linspace(-2,4,num_bins)
    #kl_val_range = numpy.linspace(-14,16,num_bins)

    truth_data_files = fileio_utils.read_coupling_file(coupling_file='basis_files/truth_LHE_couplings_prospective.dat')
    theory_function = combination_utils.get_theory_xsec_function()
    all_theory_xsec = [ theory_function(basis) for basis in truth_data_files.keys() ]

    init_truth_weights, init_truth_errors = fileio_utils.extract_lhe_truth_data(truth_data_files.values(), var_edges, stat_limit=None)
    init_xsec = [ weight_array.sum() for weight_array in init_truth_weights ]

    final_truth_weights, final_truth_errors = fileio_utils.extract_lhe_truth_data(truth_data_files.values(), var_edges, stat_limit=None, emulateSelection=True)
    final_xsec = [ weight_array.sum() for weight_array in final_truth_weights ]

    pretty_print = lambda basis, theory, init, final : print(f'{basis[0]:4.1f}, {basis[1]:4.1f}, {basis[2]:4.1f}, {theory:5.2f}, {int(init):5d}, {final:5.2f}, {final/init:5.2f}')
    print('    BASIS       , THRY , INIT , FINAL , ACCEP')
    for basis, theory, init, final in zip(truth_data_files.keys(), all_theory_xsec, init_xsec, final_xsec):
        pretty_print(basis, theory, init, final)

if __name__ == '__main__': main()
