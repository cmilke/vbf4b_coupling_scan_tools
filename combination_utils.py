import sympy
import numpy
import itertools
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

_k2v = sympy.Symbol('\kappa_{2V}')
_kl = sympy.Symbol('\kappa_{\lambda}')
_kv = sympy.Symbol('\kappa_{V}')

kl_scan_terms = [
    lambda k2v,kl,kv: kl**2,
    lambda k2v,kl,kv: kl,
    lambda k2v,kl,kv: 1
]

k2v_scan_terms = [
    lambda k2v,kl,kv: k2v**2
   ,lambda k2v,kl,kv: k2v
   ,lambda k2v,kl,kv: 1
]

kl_k2v_scan_terms = [
    lambda k2v,kl,kv: kl**2
   ,lambda k2v,kl,kv: k2v**2
   ,lambda k2v,kl,kv: kl
   ,lambda k2v,kl,kv: k2v * kl
   ,lambda k2v,kl,kv: k2v
   ,lambda k2v,kl,kv: 1
]

full_scan_terms = [
    lambda k2v,kl,kv: kv**2 * kl**2,
    lambda k2v,kl,kv: kv**4,
    lambda k2v,kl,kv: k2v**2,
    lambda k2v,kl,kv: kv**3 * kl,
    lambda k2v,kl,kv: k2v * kl * kv,
    lambda k2v,kl,kv: kv**2 * k2v
]



def is_valid_combination(basis_parameters, base_equations=full_scan_terms):
    basis_states = [ [ sympy.Rational(param) for param in basis ] for basis in basis_parameters ]
    combination_matrix = sympy.Matrix([ [ f(*base) for f in base_equations ] for base in basis_states])
    return combination_matrix.det() != 0



def get_amplitude_function( basis_parameters, as_scalar=True, base_equations=full_scan_terms, name='unnamed', output=None, preloaded_amplitudes=None):
    basis_states = [ [ sympy.Rational(param) for param in basis ] for basis in basis_parameters ]

    combination_matrix = sympy.Matrix([ [ g(*base) for g in base_equations ] for base in basis_states])

    if combination_matrix.det() == 0: return None

    inversion = combination_matrix.inv()
    term_vector = sympy.Matrix([ [g(_k2v,_kl,_kv)] for g in base_equations ])
    if type(preloaded_amplitudes) == type(None):
        amplitudes = sympy.Matrix([ sympy.Symbol(f'A{n}') for n in range(len(base_equations)) ])
    else:
        amplitudes = sympy.Matrix(preloaded_amplitudes)

    if as_scalar:
        # FYI, numpy outputs a 1x1 matrix here, so I use the [0] to get just the equation
        final_amplitude = (term_vector.T*inversion*amplitudes)[0]
        if output == 'ascii':
            avec = inversion*amplitudes
            sympy.pprint(avec)
            print()
            print()
            sympy.pprint(avec.T)
            print()
            print()
            sympy.pprint(avec*avec.T)
            #sympy.pprint(amplitudes.T*inversion)
            #sympy.pprint(sympy.simplify( (inversion*amplitudes) * (amplitudes.T*inversion)))
            #sympy.pprint(final_amplitude)
        if output == 'tex':
            substitutions = [ (a, f'Abs(A({i},{j},{k}))**2') for a, (i,j,k) in zip(amplitudes, basis_parameters) ]
            print(substitutions)
            print(final_amplitude)
            out_equation = final_amplitude.subs(substitutions)
            with open('final_amplitude_'+name+'.tex','w') as output:
                output.write('$\n'+sympy.latex(out_equation)+'\n$\n')

        if type(preloaded_amplitudes) == type(None):
            amplitude_function = sympy.lambdify([_k2v, _kl, _kv]+[*amplitudes], final_amplitude, 'numpy')
        else:
            amplitude_function = sympy.lambdify([_k2v, _kl, _kv], final_amplitude, 'numpy')
        return amplitude_function
    else:
        final_weight = term_vector.T * inversion
        reweight_vector = sympy.lambdify([_k2v, _kl, _kv], final_weight, 'numpy')
        return reweight_vector


def get_theory_xsec_function():
    basis_list = [
        ( 1.  ,  1. ,  1.  ),
        ( 1.5 ,  1. ,  1.  ),
        ( 2.  ,  1. ,  1.  ),
        ( 1.  ,  0. ,  1.  ),
        ( 1.  , 10. ,  1.  ),
        ( 1.  ,  1. ,  1.5 )
    ]

    theory_xsecs = [ #fb
         1.1836,
         2.3032,
         9.9726,
         3.1659,
        67.377,
        45.412
    ]

    amp_function = get_amplitude_function(basis_list, preloaded_amplitudes=theory_xsecs, output='ascii')
    amp_function = get_amplitude_function(basis_list)
    theory_xsec_function = lambda couplings: amp_function(*couplings, *theory_xsecs)
    return theory_xsec_function



def plot_scan(name,title, title_suffix, reweight_linear_array,
            bin_edges, xedges, yedges, weight_list, 
            plotdir='', **kwargs):

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
    fig.savefig('plots/'+plotdir+'coupling_scan_'+name+'.png',dpi=500)
    plt.close()



def plot_all_couplings(prefix, amplitude_function, basis_files, coupling_parameter_array, plotdir=''):
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
    hash_max_normalized_weight_list = horizontally_normalized_weight_list / horizontally_normalized_weight_list.max(axis=0)

    #error_list = weight_list / error_list
    #horizontally_normalized_error_list = numpy.array( [ e/e.sum() for e in error_list] )
    #vertically_normalized_error_list =  error_list / error_list.sum(axis=0) 
    #hash_normalized_error_list = horizontally_normalized_weight_list / horizontally_normalized_weight_list.sum(axis=0)

    if not os.path.isdir('plots/'+plotdir): os.mkdir('plots/'+plotdir)

    #plot_scan(prefix+'base','', 'Weights',coupling_linear_array,bin_edges,xedges,yedges,weight_list, plotdir=plotdir)
    #plot_scan(prefix+'hori','Horizontally Normalized', 'Weights',coupling_linear_array,bin_edges,xedges,yedges,horizontally_normalized_weight_list, plotdir=plotdir)
    #plot_scan(prefix+'hori_log','(Log) Horizontally Normalized', 'Weights',coupling_linear_array,bin_edges,xedges,yedges,horizontally_normalized_weight_list,
        #norm = matplotlib.colors.LogNorm(vmin=10e-4) , plotdir=plotdir)
    #plot_scan(prefix+'vert','Vertically Normalized', 'Weights',coupling_linear_array,bin_edges,xedges,yedges,vertically_normalized_weight_list, plotdir=plotdir)
    #plot_scan(prefix+'hash','Hash Normalized', 'Weights',coupling_linear_array,bin_edges,xedges,yedges,hash_normalized_weight_list, plotdir=plotdir)
    plot_scan(prefix+'hash_max','Hash-Max Normalized', 'Weights',coupling_linear_array,bin_edges,xedges,yedges,hash_max_normalized_weight_list, plotdir=plotdir)

    #plot_scan('E-base','', 'Weight/Errors',coupling_linear_array,bin_edges,xedges,yedges,error_list)
    #plot_scan('E-hori','Horizontally Normalized', 'Errors',coupling_linear_array,bin_edges,xedges,yedges,horizontally_normalized_error_list)
    #plot_scan('E-hori_log','(Log) Horizontally Normalized', 'Errors',coupling_linear_array,bin_edges,xedges,yedges,horizontally_normalized_error_list,
    #    normalize = matplotlib.colors.LogNorm(vmin=10e-4) )
    #plot_scan('E-vert','Vertically Normalized', 'Weight/Errors',coupling_linear_array,bin_edges,xedges,yedges,vertically_normalized_error_list, vmin=0)
    #plot_scan('E-hash','Hash Normalized', 'Errors',coupling_linear_array,bin_edges,xedges,yedges,hash_normalized_error_list)



def plot_all_couplings_reco(prefix, file_name, plotdir=''):
    # Read in base parameters file
    basis_parameters = []
    basis_files = []
    with open(file_name) as basis_list_file:
        for line in basis_list_file:
            if line.strip().startswith('#'): continue
            linedata = line.split()
            if len(linedata) < 3: continue
            basis_parameters.append(linedata[:3])
            basis_files.append(linedata[3])

    coupling_linear_array = basis_parameters

    var_edges = numpy.linspace(250, 2000, 51)
    from reweight_utils import extract_ntuple_events, retrieve_reco_weights
    base_events_list = [ extract_ntuple_events(b,key='m_hh',filter_vbf=False) for b in basis_files ]
    base_histograms = [ retrieve_reco_weights(var_edges, base_events) for base_events in base_events_list ]
    weight_list, error_list = numpy.array(list(zip(*base_histograms)))

    xedges = var_edges
    yedges = range(len(coupling_linear_array)+1)
    bin_edges = numpy.array([ (x,y) for x in xedges[:-1] for y in yedges[:-1] ]).transpose()

    safe_divide = lambda l,s: l if s == 0 else l/s
    horizontally_normalized_weight_list = numpy.array( [ safe_divide(w,w.sum()) for w in weight_list] )
    vertically_normalized_weight_list =  weight_list / weight_list.sum(axis=0) 
    hash_normalized_weight_list = horizontally_normalized_weight_list / horizontally_normalized_weight_list.sum(axis=0)
    hash_max_normalized_weight_list = horizontally_normalized_weight_list / horizontally_normalized_weight_list.max(axis=0)

    if not os.path.isdir('plots/'+plotdir): os.mkdir('plots/'+plotdir)
    plot_scan(prefix+'hash_max','Hash-Max Normalized', 'Weights',coupling_linear_array[::-1],bin_edges,xedges,yedges,hash_max_normalized_weight_list[::-1], plotdir=plotdir)






def main():
    #full_basis_states = [
    #    ('1'  , '1', '1'  ),
    #    ('1'  , '0', '-1' ),
    #    ('0'  , '1', '1'  ),
    #    ('3/2', '1', '1'  ),
    #    ('1'  , '2', '1'  ),
    #    ('2'  , '1', '-1' )
    #]

    validation_states = [
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
        [0    , 0   , 1   ]
    ]

    #possible_validation_combinations = itertools.combinations(validation_states,6)
    #total_possible = 0
    #for combination in possible_validation_combinations:
    #    total_possible += get_amplitude_function(str(combination), full_scan_terms, combination)
    #print()
    #print(total_possible)

    #validation_basis = [
    #    [1.5  , 1   , 1   ], #
    #    [2    , 1   , 1   ], #
    #    [1  , 1   , 1.5   ],
    #    [1    , 1   , 1   ], #
    #    [1    , 0   , 1   ], #
    #    [1    , 10  , 1   ], #
    #]
    ##get_amplitude_function('validation', full_scan_terms, validation_basis)

    existing_states = [ #k2v, kl, kv
        [1  , 1   , 1   ], # 450044
        #[1  , 2   , 1   ], # 450045
        [2  , 1   , 1   ], # 450046
        #[1.5, 1   , 1   ], # 450047
        #[1  , 1   , 0.5 ], # 450048 - !!
        [0.5, 1   , 1   ], # 450049
        #[0  , 1   , 1   ], # 450050
        [0  , 1   , 0.5 ], # 450051 - !!
        [1  , 0   , 1   ], # 450052 - ***
        #[0  , 0   , 1   ], # 450053 - !!
        #[4  , 1   , 1   ], # 450054
        [1  , 10  , 1   ], # 450055 - ***
        #[1  , 1   , 1.5 ]  # 450056 - !!
    ]

    ##basis_states = [ [ sympy.Rational(param) for param in basis ] for basis in existing_states ]
    ##kappa_matrix = sympy.Matrix([ [ g(*base) for g in full_scan_terms ] for base in basis_states])
    ##sympy.pprint(kappa_matrix)

    #possible_existing_combinations = itertools.combinations(existing_states,6)
    #possible_existing_combinations = itertools.combinations(validation_states,6)
    #total_possible = 0
    #for combination in possible_existing_combinations:
    #    if is_valid_combination(combination): total_possible += 1
    #print()
    #print(total_possible)

    #existing_basis = [ #k2v, kl, kv
    #    [1.5  , 1   , 1   ],
    #    [2    , 1   , 1   ],
    #    [1    , 2   , 1   ],
    #    [1    , 1   , 1   ],
    #    [1    , 0   , 1   ],
    #    [1    , 10  , 1   ]
    #]
    #get_amplitude_function('existing', full_scan_terms, existing_basis)

    #file_name = 'basis_files/nnt_basis.dat'
    #plot_all_couplings_reco('reco', file_name, plotdir='distro_heatmaps/')

    current_3D_reco_basis = [ #k2v, kl, kv
        [1    , 1   , 1   ],
        [2    , 1   , 1   ],
        [1.5  , 1   , 1   ],
        [0    , 1   , 0.5 ],
        [1    , 0   , 1   ],
        [1    , 10  , 1   ]
    ]
    #get_amplitude_function( current_3D_reco_basis, output='ascii' )
    #print('\n\n\n-------------------------\n\n')
    truth_basis = [
        [1   ,   1  ,  1  ],
        [1.5 ,   1  ,  1  ],
        [2   ,   1  ,  1  ],
        [1   ,   0  ,  1  ],
        [1   ,   10 ,  1  ],
        [1   ,   1  ,  1.5]
    ]
    #get_amplitude_function( truth_basis, output='ascii' )

    #kl_basis_states = [
    #    ('0  ' , '0 '  , '1  ' ),
    #    ('1  ' , '0 '  , '1  ' ),
    #    ('0  ' , '1 '  , '1  ' ),
    #    ('0  ' , '1 '  , '0.5' ),
    #    ('0.5' , '1 '  , '1  ' ),
    #    ('1  ' , '1 '  , '0.5' ),
    #    ('1  ' , '1 '  , '1  ' ),
    #    ('1.5' , '1 '  , '1  ' ),
    #    ('2  ' , '1 '  , '1  ' ),
    #    ('4  ' , '1 '  , '1  ' ),
    #    ('1  ' , '2 '  , '1  ' ),
    #    ('1  ' , '10'  , '1  ' ),
    #    ('1  ' , '11'  , '1.5' )
    #]
    #possible_existing_combinations = itertools.combinations(existing_states,6)
    #total_possible = 0
    #for combination in possible_existing_combinations:
    #    total_possible += get_amplitude_function(str(combination), full_scan_terms, combination)
    #print()
    #print(total_possible)

    #kl_basis = [
    #    [1   ,   0  ,  1  ],
    #    [1   ,   1  ,  1  ],
    #    [1   ,   20 ,  1  ],
    #]
    #get_amplitude_function(kl_basis, base_equations=kl_scan_terms, output='tex', name='kl_test' )
    #vector_function = get_amplitude_function(kl_basis, base_equations=kl_scan_terms, as_scalar=False)
    #vector = vector_function(1,2,1)[0]
    #print(vector)

    #k2v_basis_states = [
    #    ('1'  , '1', '1'  ),
    #    ('0'  , '1', '1'  ),
    #    ('2'  , '1', '1' )
    #]
    #get_amplitude_function(k2v_basis_states, base_equations=k2v_scan_terms, output='tex', name='alle_test' )

    #kl_k2v_basis_states = [
    #    ('1'  , '1', '1'  ),
    #    ('-1'  , '0', '1' ),
    #    ('0'  , '1', '1'  ),
    #    ('3/2', '1', '1'  ),
    #    ('1'  , '3/2', '1'  ),
    #    ('1'  , '-1', '1' )
    #]

    #get_amplitude_function('all', full_scan_terms, full_basis_states)
    #print('\n\n\n')
    #get_amplitude_function('kl', kl_equation_list, kl_basis_states)
    #print('\n\n\n')
    #get_amplitude_function('k2v', k2v_equation_list, k2v_basis_states)
    #get_amplitude_function('kl_k2v', kl_k2v_equation_list, kl_k2v_basis_states)
    get_theory_xsec_function()


if __name__ == '__main__': main()
