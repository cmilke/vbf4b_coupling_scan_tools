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

    truth_data_files = fileio_utils.read_coupling_file(coupling_file='basis_files/truth_LHE_couplings.dat')
    truth_weights, truth_errors = fileio_utils.extract_lhe_truth_data(truth_data_files.values(), var_edges, stat_limit=None, emulateSelection=True)

if __name__ == '__main__': main()
