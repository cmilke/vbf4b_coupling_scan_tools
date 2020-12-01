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

    ### THIS EQUATION IS THE CENTRAL POINT OF THIS ENTIRE PROGRAM! ###
    final_amplitude = (kappa.T*inversion*amplitudes)[0]
    # FYI, numpy outputs a 1x1 matrix here, so I use the [0] to get just the equation
    ##################################################################

    if output_equation != None:
        sympy.pprint(final_amplitude)
        if output_equation == 'tex':
            with open('final_amplitude.tex','w') as output:
                output.write('$\n'+sympy.latex(final_amplitude)+'\n$\n')

    print( final_amplitude.subs([ (f'A{n}',a) for n, a in enumerate([1,2,3,4,5,6])]).subs(_kv,1) )
    exit(0)


    amplitude_function = sympy.lambdify([_k2v, _kl, _kv]+[*amplitudes], final_amplitude, 'numpy')
    return amplitude_function


def plot_histogram(hist_name, hist_title, edge_list, coupling_parameters, combination_function, error_function, verification_weights, verification_errors):
    linearly_combined_weights = combination_function(coupling_parameters)
    linearly_combined_errors = error_function(coupling_parameters)
    print('Plotting '+hist_name+' for ' + str(coupling_parameters))
    fig, (ax_main, ax_ratio) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios':[4,1]} )

    xpositions = 0.5*(edge_list[1:]+edge_list[:-1])
    counts, bins, points = ax_main.errorbar( xpositions, linearly_combined_weights,
        yerr=linearly_combined_errors, label='Linear Combination',
        marker='.', markersize=2, capsize=2, color='blue', linestyle='none', linewidth=1, zorder=3)

    if len(verification_weights) > 0:
        vcounts, vbins, vhists = ax_main.hist( [edge_list[:-1]]*2,
            weights=[verification_weights-verification_errors, 2*verification_errors],
            label=['Generated MC', 'MC Statistical Error'],
            bins=edge_list, fill=True, histtype='barstacked', zorder=1, alpha=0.5, color=['green','red'])
        plt.setp(vhists[1], hatch='/////')

        safe_verification = verification_errors.copy()
        safe_verification[ safe_verification == 0 ] = float('inf')
        rcounts, rbins, rpoints = ax_ratio.errorbar( xpositions, (linearly_combined_weights-verification_weights),
            yerr=linearly_combined_errors/safe_verification, label='MC Statistical Error check',
            marker='.', markersize=2, capsize=2, color='blue', linestyle='none', linewidth=1, zorder=3)
        
        zero_line = ax_ratio.hlines(0,xmin=edge_list[0],xmax=edge_list[-1],colors='black',zorder=2)



    kappa_labels = [ str(param) for param in coupling_parameters ]
    title  = hist_title+' for '
    title += '$\kappa_{2V}='+kappa_labels[0]+'$, '
    title += '$\kappa_{\lambda}='+kappa_labels[1]+'$, '
    title += '$\kappa_{V}='+kappa_labels[2]+'$'
    fig.suptitle(title)

    #ax_main.set(ylabel='')
    ax_ratio.set_ylim([-2,2])
    ax_ratio.set_yticks(ticks=[-2,-1,0,1,2])
    ax_ratio.set(ylabel=r'$\frac{lin. comb. - gen.}{stat. error}$', xlabel='Truth $m_{HH}$ (GeV)')
    ax_main.legend(prop={'size':7})
    ax_main.grid()
    ax_ratio.grid()

    dpi=500
    kappa_string_list = [ label.replace('.','p') for label in kappa_labels ]
    kappa_string = 'cvv'+kappa_string_list[0]+'cl'+kappa_string_list[1]+'cv'+kappa_string_list[2]
    fig.savefig('plots/'+hist_name+'_'+kappa_string+'.png', dpi=dpi)
    plt.close()



def display_linear_combinations(amplitude_function, basis_files, verification_parameters, verification_files):
    if len(basis_files) < 1: return

    hist_key = b'HH_m'
    basis_weight_list = []
    basis_error_list = []
    edge_list = []
    for base_file in basis_files:
        directory = uproot.open(base_file)
        root_hist = directory[hist_key]
        weights, edges = root_hist.numpy()
        basis_weight_list.append(weights)
        errors = numpy.sqrt(root_hist.variances)
        basis_error_list.append(errors)
        if len(edge_list) == 0: edge_list = edges
    combination_function = lambda params: amplitude_function(*params, *basis_weight_list)
    error_function = lambda params: amplitude_function(*params, *basis_error_list)

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
        plot_histogram('mHH', '$m_{HH}$', edge_list, coupling_parameters, combination_function, error_function, verification_weights, verification_errors)




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
