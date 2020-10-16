import sys
import argparse
import sympy
import numpy
import uproot
import inspect

from basis_scan import plot_all_couplings

_k2v_coupling_range = numpy.arange(-5,5,0.2)
_kl_coupling_range = numpy.arange(-15,15,0.5)
_kv_coupling_range = numpy.arange(-3,3,0.2)
_k2v_len = len(_k2v_coupling_range)
_kl_len = len(_kl_coupling_range)
_kv_len = len(_kv_coupling_range)


class basis_state():
    def __init__(self, basis_index, distro):
        self.basis_index = basis_index
        self.weight_distribution = distro
        self.bin_index_list = []
        self.weight_list = []

    def add_bin(self, bin_index):
        self.bin_index_list.append(bin_index)
        new_weight = self.weight_distribution[bin_index]
        self.weight_list.append(new_weight)

    def is_safe(self, index):
        if index == 0:
            return index+1 in self.bin_index_list
        elif index == len(self.weight_distribution)-1:
            return index-1 in self.bin_index_list
        else:
            return index-1 in self.bin_index_list and index+1 in self.bin_index_list

    def remove_most_vulnerable_bin(self):
        ranked_weights = sorted( [wi for wi in zip(self.weight_list, self.bin_index_list)] )
        for weight, index in ranked_weights:
            if self.is_safe(index): continue
            internal_list_index = self.bin_index_list.index(index)
            del self.bin_index_list[internal_list_index]
            del self.weight_list[internal_list_index]
            return index

    def is_out_of_bins(self):
        return len(self.bin_index_list) == 0

    def get_strength(self):
        #multiplier = len(self.bin_index_list)**1.2
        #multiplier = 1
        #unowned_bin_weights = [ w for i,w in enumerate(self.weight_distribution) if i not in self.bin_index_list ]
        #strength = sum(self.weight_list) - 10*sum(unowned_bin_weights)
        strength = sum(self.weight_list)
        return strength
        



def get_amplitude_function(basis_parameters):
    basis_states = [ [ sympy.Rational(param) for param in basis ] for basis in basis_parameters ]

    _k2v = sympy.Symbol('\kappa_{2V}')
    _kl = sympy.Symbol('\kappa_{\lambda}')
    _kv = sympy.Symbol('\kappa_{V}')

    gamma_list = [
        lambda k2v,kl,kv: kv**2 * kl**2
       ,lambda k2v,kl,kv: kv**4
       ,lambda k2v,kl,kv: k2v**2
       ,lambda k2v,kl,kv: kv**3 * kl
       ,lambda k2v,kl,kv: k2v * kl * kv
       ,lambda k2v,kl,kv: kv**2 * k2v
    ]

    kappa_matrix = sympy.Matrix([ [ g(*base) for g in gamma_list ] for base in basis_states])
    inversion = kappa_matrix.inv()
    kappa = sympy.Matrix([ [g(_k2v,_kl,_kv)] for g in gamma_list ])
    amplitudes = sympy.Matrix([ sympy.Symbol(f'A{n}') for n in range(6) ])

    ### THIS EQUATION IS THE CENTRAL POINT OF THIS ENTIRE PROGRAM! ###
    final_amplitude = (kappa.T*inversion*amplitudes)[0]
    # FYI, numpy outputs a 1x1 matrix here, so I use the [0] to get just the equation
    ##################################################################

    amplitude_function = sympy.lambdify([_k2v, _kl, _kv]+[*amplitudes], final_amplitude, 'numpy')
    return amplitude_function



def perform_optimization(amplitude_function, basis_files, coupling_array):
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

    # Get Normalized (per basis) weights
    array_length = len(coupling_array)
    coupling_linear_array = numpy.reshape(coupling_array, (_k2v_len*_kl_len*_kv_len,3))
    norm_weight_list = []
    for couplings in coupling_linear_array:
        bins = combination_function(couplings)
        total = bins.sum()
        norm_weight = bins if total == 0 else bins/total
        norm_weight_list.append(norm_weight)
    norm_weight_linear_array = numpy.array(norm_weight_list)

    # Get top contenders for basis states
    per_bin_max = norm_weight_linear_array.max(axis=0)
    per_bin_max_linear_indices = norm_weight_linear_array.argmax(axis=0)

    # Assemble the contending bases states
    basis_contenders = {}
    bin_ownership = { i:None for i in range(len(norm_weight_linear_array[0]))}
    for bin_index, basis_index in enumerate(per_bin_max_linear_indices):
        if basis_index not in basis_contenders:
            new_basis = basis_state(basis_index, norm_weight_linear_array[basis_index])
            basis_contenders[basis_index] = new_basis
        basis_contenders[basis_index].add_bin(bin_index)
        bin_ownership[bin_index] = basis_index

    # Begin the Basis State Hunger Games
    while len(basis_contenders) > 6:
        # Get strength of all contenders. Find the weakest
        strengths = [ (contender.get_strength(), basis_index) for basis_index, contender in basis_contenders.items() ]
        losing_basis_index = sorted(strengths)[0][1]
        weakest_basis = basis_contenders[losing_basis_index]
        lost_index = weakest_basis.remove_most_vulnerable_bin()

        # Choose who gets to eat the lost index
        winning_bases = []
        if lost_index-1 in bin_ownership and bin_ownership[lost_index-1] != losing_basis_index:
            winning_bases.append(bin_ownership[lost_index-1])
        if lost_index+1 in bin_ownership and bin_ownership[lost_index+1] != losing_basis_index:
            winning_bases.append(bin_ownership[lost_index+1])
        ranked_winners = [ (basis_contenders[index].get_strength(), index) for index in winning_bases ]
        winning_basis_index = sorted(ranked_winners, reverse=True)[0][1]

        # To the victor go the spoils. Remove the losing basis if it has run out of bins
        basis_contenders[winning_basis_index].add_bin(lost_index)
        bin_ownership[lost_index] = winning_basis_index
        if weakest_basis.is_out_of_bins(): del basis_contenders[losing_basis_index]
    final_couplings = [ [round(coupling,2) for coupling in coupling_linear_array[basis_index]] for basis_index in basis_contenders ]
    for c in final_couplings: print(c)
    get_amplitude_function(final_couplings) # Make sure the final states are actually invertable
    plot_all_couplings(amplitude_function, basis_files, final_couplings)
        
        
        
    #bin_index_conversion = lambda i: ( int(i/(_kl_len*_kv_len)), int( ( i%(_kl_len*_kv_len) ) / _kv_len ),  ( i%(_kl_len*_kv_len) ) % _kv_len  )
    #per_bin_max_indices = [ (bin_i,bin_index_conversion(index)) for bin_i, index in enumerate(per_bin_max_linear_indices) ] 
    #norm_weight_array = numpy.reshape( norm_weight_list, (_k2v_len,_kl_len,_kv_len,len(norm_weight_list[0])) )
    #print( numpy.array([ norm_weight_array[i][j][k][b] for b,(i,j,k) in per_bin_max_indices ]) )

    #print(len(norm_weight_list))
    #print(len(norm_weight_array))




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

    coupling_nested_list = [  [ [[k2v,kl,kv] for kv in _kv_coupling_range] for kl in _kl_coupling_range ] for k2v in _k2v_coupling_range  ]
    coupling_array = numpy.array(coupling_nested_list)

    # Get amplitude function and perform reweighting
    amplitude_function = get_amplitude_function(basis_parameters)
    perform_optimization(amplitude_function, basis_files, coupling_array)



if __name__ == '__main__': main()
