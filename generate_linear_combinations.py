import sys
import argparse
import sympy
import numpy
import uproot
import inspect

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



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

    amplitude_function = sympy.lambdify([_k2v, _kl, _kv]+[*amplitudes], final_amplitude, 'numpy')
    return amplitude_function


def plot_histogram(hist_name, edge_list, coupling_parameters, combination_function, verification_weights, verification_errors):
        linearly_combined_weights = combination_function(coupling_parameters)
        print('Plotting '+hist_name+' for ' + str(coupling_parameters))
        fig,ax = plt.subplots()

        counts, bins, hist = plt.errorbar( 0.5*(edge_list[1:]+edge_list[:-1]), linearly_combined_weights,
            label='Linear Combination', marker='.', color='blue', linestyle='none', linewidth=3, zorder=3)

        if len(verification_weights) > 0:
            vcounts, vbins, vhists = plt.hist( [edge_list[:-1]]*2,
                weights=[verification_weights-verification_errors, 2*verification_errors],
                label=['Generated MC', 'MC Statistical Error'],
                bins=edge_list, fill=True, histtype='barstacked', zorder=1, alpha=0.5, color=['green','red'])
            plt.setp(vhists[1], hatch='/////')

            #scounts, sbins, shist = plt.errorbar( 0.5*(vbins[1:]+vbins[:-1]), verification_weights,
                #yerr=verification_errors, label='MC Statistical Error check', linewidth=1, fmt='none', zorder=2)



        kappa_labels = [ str(param) for param in coupling_parameters ]
        title  = hist_name+' for '
        title += '$\kappa_{2V}='+kappa_labels[0]+'$, '
        title += '$\kappa_{\lambda}='+kappa_labels[1]+'$, '
        title += '$\kappa_{V}='+kappa_labels[2]+'$'
        plt.title(title)
        plt.grid()
        ax.legend(prop={'size':7})
        dpi=500
        kappa_strings = [ label.replace('.','p') for label in kappa_labels ]
        fig.savefig('plots/'+hist_name+'_'.join(kappa_strings)+'.png', dpi=dpi)
        plt.close()



def display_linear_combinations(amplitude_function, basis_files, verification_parameters, verification_files):
    if len(basis_files) < 1: return

    hist_key = b'HH_m'
    weight_list = []
    edge_list = []
    for base_file in basis_files:
        directory = uproot.open(base_file)
        root_hist = directory[hist_key]
        weights, edges = root_hist.numpy()
        weight_list.append(weights)
        if len(edge_list) == 0: edge_list = edges
    combination_function = lambda params: amplitude_function(*params, *weight_list)

    verification_weight_list = []
    verification_error_list = []
    for verification_file in verification_files:
        directory = uproot.open(verification_file)
        root_hist = directory[hist_key]
        weights, edges = root_hist.numpy()
        errors = numpy.sqrt(root_hist.variances)
        verification_weight_list.append(weights)
        verification_error_list.append(errors)

    for index, coupling_parameters in enumerate(verification_parameters):
        verification_weights = []
        if len(verification_weight_list) > 0:
            verification_weights = verification_weight_list[index]
            verification_errors = verification_error_list[index]
        plot_histogram(hist_key.decode(), edge_list, coupling_parameters, combination_function, verification_weights, verification_errors)




def main():
    # Sort out command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( "--basis", required = True, default = None, type=str,
        help = "File to provide basis states",)

    parser.add_argument( "--equation", required = False, default = None, type=str,
        help = "output reweighting function to: \'ascii\' or \'tex\' (writes Latex-Readable file)",) 

    parser.add_argument( "--plotlist", required = False, default = None, type=str,
        help = "Plot linear combinations of the provided parameters list",) 

    parser.add_argument( "--verify", required = False, default = False, action = 'store_true',
        help = "Verify linear combinations against provided files",) 

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
            if args.plotlist != None: basis_files.append(linedata[3])

    # Read in parameter list to be generated by linear combination
    verification_parameters = []
    verification_files = []
    if args.plotlist != None:
        with open(args.plotlist) as plot_list_file:
            for line in plot_list_file:
                if line.strip().startswith('#'): continue
                linedata = line.split()
                params = [ float(p) for p in linedata[:3] ]
                verification_parameters.append(params)
                if args.verify: verification_files.append(linedata[3])

    # Get amplitude function and perform reweighting
    amplitude_function = get_amplitude_function(basis_parameters, args.equation)
    display_linear_combinations(amplitude_function, basis_files, verification_parameters, verification_files)



if __name__ == '__main__': main()
