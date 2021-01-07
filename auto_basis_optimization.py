import sys
import argparse
import sympy
import numpy
import uproot
import inspect
import itertools

import combination_utils


def perform_optimization(amplitude_function, basis_files, coupling_array, base_equations=combination_utils.full_equation_list):
    if len(basis_files) < 1: return

    hist_key = b'HH_m'
    basis_weight_list = []
    edge_list = []
    for base_file in basis_files:
        directory = uproot.open(base_file)
        root_hist = directory[hist_key]
        weights, edges = root_hist.numpy()
        basis_weight_list.append(weights)
        errors = numpy.sqrt(root_hist.variances)
        if len(edge_list) == 0: edge_list = edges
    combination_function = lambda params: amplitude_function(*params, *basis_weight_list)
    print('Combination Function Acquired, generating requested states...')

    # Get Normalized (per basis) weights
    norm_weight_list = []
    for couplings in coupling_array:
        bins = combination_function(couplings)
        total = bins.sum()
        norm_weight = bins if total == 0 else bins/total
        norm_weight_list.append(norm_weight)
    norm_weight_linear_array = numpy.array(norm_weight_list)

    # Get top contenders for basis states if there are too many
    if len(coupling_array) > 20:
        per_bin_max_linear_indices = norm_weight_linear_array.argmax(axis=0)
        top_contender_indices = numpy.unique(per_bin_max_linear_indices)
        coupling_array = coupling_array[ top_contender_indices ]
        norm_weight_linear_array = norm_weight_linear_array[ top_contender_indices ]
    print('Maximum States Identified, generating combinations')
    coupling_weight_tuples = zip(coupling_array, norm_weight_linear_array)
    coupling_combinations = itertools.combinations(coupling_weight_tuples, len(basis_files))
    weighted_combinations = [ list(zip(*c)) for c in coupling_combinations ]

    print('Filtering to valid combinations...')
    valid_combinations = [ (numpy.array(c),numpy.array(w)) for c,w in weighted_combinations if combination_utils.is_valid_combination(c, base_equations=base_equations) ]
    print('Sorting...')
    ordered_combinations = [ ( weights.max(axis=0).sum(), couplings ) for couplings, weights in valid_combinations ]
    ordered_combinations.sort(reverse=True,key=lambda c: c[0])
    final_couplings = ordered_combinations[0][1]
    #print( len(ordered_combinations))
    #for i, (w,b) in enumerate(ordered_combinations[:15]):
    #    print(i)
    #    print(b)
    #    print()

    #for index, (val, arr) in enumerate(ordered_combinations[:9]):
    for index, (val, arr) in enumerate([ordered_combinations[12]]):
        print(val)
        print(arr)
        combination_utils.get_amplitude_function(final_couplings, base_equations=base_equations, output='tex', name=f'recoR{index}')
        combination_utils.plot_all_couplings(f'kl_R{index}_', amplitude_function, basis_files, arr[::-1], plotdir='auto_chosen/')
        print()
    #print()
    #print(final_couplings)

    #get_amplitude_function(final_couplings) # Make sure the final states are actually invertable
    #plot_all_couplings('auto_chosen_', amplitude_function, basis_files, final_couplings)
        


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

    #coupling_nested_list = [  [ [[k2v,kl,kv] for kv in _kv_coupling_range] for kl in _kl_coupling_range ] for k2v in _k2v_coupling_range  ]
    #coupling_array = numpy.array(coupling_nested_list)
    #coupling_list = numpy.reshape(coupling_array, (_k2v_len*_kl_len*_kv_len,3))

    #validation_states = [
    #    [1    , 1   , 1   ],
    #    [0    , 1   , 1   ],
    #    [0.5  , 1   , 1   ],
    #    [1.5  , 1   , 1   ],
    #    [2    , 1   , 1   ],
    #    [3    , 1   , 1   ],
    #    [1    , 0   , 1   ],
    #    [1    , 2   , 1   ],
    #    [1    , 10  , 1   ],
    #    [1    , 1   , 0.5 ],
    #    [1    , 1   , 1.5 ],
    #    [0    , 0   , 1   ]
    #]
    #coupling_array = numpy.array(validation_states)

    #existing_states = [ #k2v, kl, kv
    #    [1  , 1   , 1   ], # 450044 ***
    #    [1  , 2   , 1   ], # 450045
    #    [2  , 1   , 1   ], # 450046 ***
    #    [1.5, 1   , 1   ], # 450047 ***
    #    [1  , 1   , 0.5 ], # 450048 - !!
    #    [0.5, 1   , 1   ], # 450049
    #    [0  , 1   , 1   ], # 450050
    #    [0  , 1   , 0.5 ], # 450051 - !! *** ???
    #    [1  , 0   , 1   ], # 450052 ***
    #    [0  , 0   , 1   ], # 450053 - !!
    #    [4  , 1   , 1   ], # 450054
    #    [1  , 10  , 1   ], # 450055 ***
    #    [1  , 1   , 1.5 ]  # 450056 - !! XXX
    #]
    #coupling_array = numpy.array(existing_states)

    kl_basis_states = [
        [0   , 0   , 1   ],
        [1   , 0   , 1   ],
        [0   , 1   , 1   ],
        [0   , 1   , 0.5 ],
        [0.5 , 1   , 1   ],
        [1   , 1   , 0.5 ],
        [1   , 1   , 1   ],
        [1.5 , 1   , 1   ],
        [2   , 1   , 1   ],
        [4   , 1   , 1   ],
        [1   , 2   , 1   ],
        [1   , 10  , 1   ],
        [1   , 11  , 1.5 ]
    ]
    coupling_array = numpy.array(kl_basis_states)
    base_equations=combination_utils.kl_scan_terms

    # Get amplitude function and perform reweighting
    amplitude_function = combination_utils.get_amplitude_function(basis_parameters,
            base_equations = base_equations)
    if amplitude_function == None:
        print('Encountered invalid basis state. Aborting')
        exit(1)
    perform_optimization(amplitude_function, basis_files, coupling_array, base_equations=base_equations)



if __name__ == '__main__': main()
