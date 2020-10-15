import sys
import argparse
import sympy
import numpy
import uproot
import inspect


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



def perform_reweighting(amplitude_function, basis_files, reweight_parameter_array):
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

    #print(reweight_parameter_array)
    array_length = len(reweight_parameter_array)
    reweight_linear_array = numpy.reshape(reweight_parameter_array, (array_length**3,3))
    #rint()
    #rint(reweight_linear_array)
    weight_list = [ combination_function(params)/basis_weight_list[0] for params in reweight_linear_array ]
    #rint()
    #rint(weight_list)
    weight_array = numpy.reshape( weight_list, (array_length,array_length,array_length,len(weight_list[0])) )
    #rint()
    #rint(weight_array)
    #print(weight_list)
    print(len(bin_heights))
    print(len(weight_array))


    #for index, coupling_parameters in enumerate(reweight_parameter_array):
        #linearly_combined_weights = combination_function(coupling_parameters)



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

    coupling_range = numpy.arange(-20,20,0.5)
    reweight_nested_list = [  [ [[k2v,kl,kv] for kv in coupling_range] for kl in coupling_range ] for k2v in coupling_range  ]
    reweight_parameter_array = numpy.array(reweight_nested_list)


    # Get amplitude function and perform reweighting
    amplitude_function = get_amplitude_function(basis_parameters)
    perform_reweighting(amplitude_function, basis_files, reweight_parameter_array)



if __name__ == '__main__': main()
