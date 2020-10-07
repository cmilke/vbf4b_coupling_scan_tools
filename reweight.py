import sys
import argparse
import sympy
import numpy
import uproot


def get_basis_amplitudes(basis_files):
    print('UNIMPLEMENTED')
    return None



def get_amplitude_function(basis_parameters, output_equation):
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
    final_amplitude = (kappa.T*inversion*amplitudes)[0]
    if output_equation != None:
        sympy.pprint(final_amplitude)
        if output_equation == 'tex':
            with open('final_amplitude.tex','w') as output:
                output.write('$\n'+sympy.latex(final_amplitude)+'\n$\n')

    amplitude_function = sympy.lambdify([_kv, _k2v, _kl]+[*amplitudes], final_amplitude, 'numpy')
    return amplitude_function



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--basis", required = True, default = None, type=str,
        help = "File to provide basis states",)

    parser.add_argument( "--equation", required = False, default = None, type=str,
        help = "output reweighting function to: \'ascii\' or \'tex\' (writes Latex-Readable file)",) 

    parser.add_argument( "--reweight", required = False, default = None, type=str,
        help = "Perform reweighting on provided file list",) 

    parser.add_argument( "--verify", required = False, default = False, action = 'store_true',
        help = "Verify reweighting against provided files",) 


    args = parser.parse_args()

    basis_parameters = []
    basis_files = []
    with open(args.basis) as basis_list_file:
        for line in basis_list_file:
            if line.strip().startswith('#'): continue
            linedata = line.split()
            if len(linedata) < 3: continue
            basis_parameters.append(linedata[:3])
            if args.reweight != None: basis_files.append(linedata[3])
        
    amplitude_function = get_amplitude_function(basis_parameters, args.equation)

    if args.reweight == None: return

    reweight_parameters = []
    verification_files = []
    with open(args.reweight) as reweight_list_file:
        for line in reweight_list_file:
            if line.strip().startswith('#'): continue
            linedata = line.split()
            params = [ float(p) for p in linedata[:3] ]
            reweight_parameters.append(params)
            if args.verify: verification_files.append(linedata[3])


    hist_key = b'HH_m'
    weight_list = []
    edge_list = []
    for base_file in basis_files:
        directory = uproot.open(base_file)
        root_hist = directory[hist_key]
        weights, edges = root_hist.numpy()
        print(weights)
        weight_list.append(weights)
        edge_list.append(edges)
    reweight_function = lambda params: amplitude_function(*params, *weight_list)

    print()
    for coupling_parameters in reweight_parameters:
        linearly_combined_weights = reweight_function(coupling_parameters)
        print(linearly_combined_weights)
        



if __name__ == '__main__': main()
