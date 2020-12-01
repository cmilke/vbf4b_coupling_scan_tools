import os
import sys
import argparse
import sympy
import numpy
import uproot
import inspect

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import reweight_utils


def main():
    # Sort out command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( "--basis", required = True, default = None, type=str,
        help = "File to provide basis states",)

    args = parser.parse_args()

    # Read in base parameters file
    basis_parameters = []
    basis_files = []
    with open(args.basis) as basis_list_file:
        for line in basis_list_file:
            if line.strip().startswith('#'): continue
            linedata = line.split()
            if len(linedata) < 3: continue
            basis_parameters.append(linedata[:3])
            basis_files.append(linedata[3])

    #coupling_range = numpy.arange(-20,20,10)
    #coupling_range = numpy.arange(-10,10,5)
    #coupling_range = [-20,-10,-5,-2,-1,0,1,2,5,10,20]
    #coupling_nested_list = [  [ [[k2v,kl,kv] for kv in coupling_range] for kl in coupling_range ] for k2v in coupling_range  ]
    coupling_nested_list = [ #k2v, kl, kv
        #[1,1,1],
        ##[1,1,2],
        ##[1,1,3],
        #[1,2,1],
        #[1,3,1],
        #[2,1,1],
        #[3,1,1],
        #[2,2,2],
        #[2,2,1],
        #[2,1,2],
        #[1,2,2],
        #[1,1,-1],
        #[1,-1,1],
        #[-1,1,1],

        #[1    , 1   , 1   ],
        #[0    , 1   , 1   ],
        #[0.5  , 1   , 1   ],
        #[1.5  , 1   , 1   ],
        #[2    , 1   , 1   ],
        #[3    , 1   , 1   ],
        #[1    , 0   , 1   ],
        #[1    , 2   , 1   ],
        #[1    , 10  , 1   ],
        #[1    , 1   , 0.5 ],
        #[1    , 1   , 1.5 ],
        #[0    , 0   , 1   ],

        #[0    , 1   , 1   ],
        #[3    , 1   , 1   ],
        #[1    , 2   , 1   ],
        #[1    , 1   , 0.5 ],
        #[1    , 1   , 1.5 ],
        #[0    , 0   , 1   ],

        [1.5  , 1   , 1   ],
        [2    , 1   , 1   ],
        [1  , 1   , 1.5   ],
        [1    , 1   , 1   ],
        [1    , 0   , 1   ],
        [1    , 10  , 1   ],

        #[3  , -1   , 1.2   ],
        #[2.6, -14.5, 0.4   ],
        #[4.6, -3   , -2.2   ],
        #[3.4, -0.5 , -1.8   ],
        #[0.2    , -10   , 0.2   ],
        #[1.2    , -10.5  , -1.6   ],

    ]
    coupling_parameter_array = numpy.array(coupling_nested_list[::-1])


    # Get amplitude function and perform reweighting
    amplitude_function = reweight_utils.get_amplitude_function(basis_parameters)
    reweight_utils.plot_all_couplings('valid_', amplitude_function, basis_files, coupling_parameter_array)



if __name__ == '__main__': main()
