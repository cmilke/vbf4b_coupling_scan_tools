import sys
import argparse
import sympy
import numpy
import uproot
import inspect

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


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


def plot_scan(name,title, title_suffix, reweight_linear_array,bin_edges,xedges,yedges,weight_list,**kwargs):
    flat_counts = weight_list.flatten(order='F')

    fig,ax = plt.subplots()
    counts, xbins, ybins, hist = plt.hist2d( *bin_edges, bins=(xedges,yedges)
        , weights=flat_counts, **kwargs )
    param_ticks = numpy.array(yedges)[:-1]+0.5
    padded_params = list(reweight_linear_array)
    param_labels = [ f'{c2v},{cl},{cv}' for (c2v,cl,cv) in padded_params ]
    plt.xlabel('$m_{HH}$')
    plt.ylabel('Basis')
    #plt.yticks(ticks=param_ticks, labels=param_labels, fontsize=7, va='center_baseline')
    #ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(param_ticks))
    #ax.yaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(param_labels))
    ax.set_yticklabels('')
    ax.set_yticks(param_ticks-0.5)
    ax.set_yticks(param_ticks, minor=True)
    ax.set_yticklabels(param_labels, minor=True, fontsize=5)
    plt.colorbar()
    plt.grid(axis='y', which='major')
    plt.title(title+' Coupling Distribution '+title_suffix)
    plt.tight_layout()
    fig.savefig('plots/coupling_scan_'+name+'.png',dpi=500)
    plt.close()



def plot_all_couplings(amplitude_function, basis_files, coupling_parameter_array):
    if len(basis_files) < 1: return

    hist_key = b'HH_m'
    basis_weight_list = []
    basis_error_list = []
    edge_list = []
    for base_file in basis_files:
        directory = uproot.open(base_file)
        root_hist = directory[hist_key]
        weights, edges = root_hist.numpy()
        errors = numpy.sqrt(root_hist.variances)
        basis_weight_list.append(weights)
        basis_error_list.append(errors)
        if len(edge_list) == 0: edge_list = edges
    combination_function = lambda params: amplitude_function(*params, *basis_weight_list)
    error_function = lambda params: amplitude_function(*params, *basis_error_list)

    array_length = len(coupling_parameter_array)
    #reweight_linear_array = numpy.reshape(coupling_parameter_array, (array_length**3,3))
    coupling_linear_array = coupling_parameter_array
    weight_list = numpy.array([ combination_function(params) for params in coupling_linear_array ])
    error_list = numpy.array([ error_function(params) for params in coupling_linear_array ])

    xedges = edge_list
    yedges = range(len(coupling_linear_array)+1)
    bin_edges = numpy.array([ (x,y) for x in xedges[:-1] for y in yedges[:-1] ]).transpose()

    safe_divide = lambda l,s: l if s == 0 else l/s
    horizontally_normalized_weight_list = numpy.array( [ safe_divide(w,w.sum()) for w in weight_list] )
    vertically_normalized_weight_list =  weight_list / weight_list.sum(axis=0) 
    hash_normalized_weight_list = horizontally_normalized_weight_list / horizontally_normalized_weight_list.sum(axis=0)

    #error_list = weight_list / error_list
    #horizontally_normalized_error_list = numpy.array( [ e/e.sum() for e in error_list] )
    #vertically_normalized_error_list =  error_list / error_list.sum(axis=0) 
    #hash_normalized_error_list = horizontally_normalized_weight_list / horizontally_normalized_weight_list.sum(axis=0)

    plot_scan('base','', 'Weights',coupling_linear_array,bin_edges,xedges,yedges,weight_list)
    plot_scan('hori','Horizontally Normalized', 'Weights',coupling_linear_array,bin_edges,xedges,yedges,horizontally_normalized_weight_list)
    plot_scan('hori_log','(Log) Horizontally Normalized', 'Weights',coupling_linear_array,bin_edges,xedges,yedges,horizontally_normalized_weight_list,
        norm = matplotlib.colors.LogNorm(vmin=10e-4) )
    plot_scan('vert','Vertically Normalized', 'Weights',coupling_linear_array,bin_edges,xedges,yedges,vertically_normalized_weight_list)
    plot_scan('hash','Hash Normalized', 'Weights',coupling_linear_array,bin_edges,xedges,yedges,hash_normalized_weight_list)

    #plot_scan('E-base','', 'Weight/Errors',coupling_linear_array,bin_edges,xedges,yedges,error_list)
    #plot_scan('E-hori','Horizontally Normalized', 'Errors',coupling_linear_array,bin_edges,xedges,yedges,horizontally_normalized_error_list)
    #plot_scan('E-hori_log','(Log) Horizontally Normalized', 'Errors',coupling_linear_array,bin_edges,xedges,yedges,horizontally_normalized_error_list,
    #    normalize = matplotlib.colors.LogNorm(vmin=10e-4) )
    #plot_scan('E-vert','Vertically Normalized', 'Weight/Errors',coupling_linear_array,bin_edges,xedges,yedges,vertically_normalized_error_list, vmin=0)
    #plot_scan('E-hash','Hash Normalized', 'Errors',coupling_linear_array,bin_edges,xedges,yedges,hash_normalized_error_list)




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
        [1,1,1],
        #[1,1,2],
        #[1,1,3],
        [1,2,1],
        [1,3,1],
        [2,1,1],
        [3,1,1],
        [2,2,2],
        [2,2,1],
        [2,1,2],
        [1,2,2],
        [1,1,-1],
        [1,-1,1],
        [-1,1,1],
        [1    , 1   , 1   ],
        [0    , 1   , 1   ],
        [0.5  , 1   , 1   ],
        [1.5  , 1   , 1   ],
        [2    , 1   , 1   ],
        [3    , 1   , 1   ],
        [1    , 0   , 1   ],
        [1    , 2   , 1   ],
        [1    , 10  , 1   ],
        [1    , 1   , 0.5 ],
        [1    , 1   , 1.5 ],
        [0    , 0   , 1   ],

    ]
    coupling_parameter_array = numpy.array(coupling_nested_list[::-1])


    # Get amplitude function and perform reweighting
    amplitude_function = get_amplitude_function(basis_parameters)
    plot_all_couplings(amplitude_function, basis_files, coupling_parameter_array)



if __name__ == '__main__': main()
