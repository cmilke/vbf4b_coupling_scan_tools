import sympy
import numpy
import re
import itertools
import os
import matplotlib
#matplotlib.use('Agg')
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
    lambda k2v,kl,kv: k2v**2,
    lambda k2v,kl,kv: k2v,
    lambda k2v,kl,kv: 1
]

kl_k2v_scan_terms = [
    lambda k2v,kl,kv: kl**2,
    lambda k2v,kl,kv: k2v**2,
    lambda k2v,kl,kv: kl,
    lambda k2v,kl,kv: k2v * kl,
    lambda k2v,kl,kv: k2v,
    lambda k2v,kl,kv: 1
]

full_scan_terms = [
    lambda k2v,kl,kv: kv**2 * kl**2,
    lambda k2v,kl,kv: kv**4,
    lambda k2v,kl,kv: k2v**2,
    lambda k2v,kl,kv: kv**3 * kl,
    lambda k2v,kl,kv: k2v * kl * kv,
    lambda k2v,kl,kv: kv**2 * k2v
]


basis_full3D_max = [ 
    (1, 1, 1),
    (2, 1, 1),
    (1.5, 1, 1),
    (0, 1, 0.5),
    (1, 0, 1),
    (1, 10, 1)
]


basis_full3D_old_minN = [
    (1, 1, 1),
    (0, 1, 0.5),
    (1, 0, 1),
    (1, 10, 1),
    (0.5, 1, 1),
    (4, 1, 1)
]

basis_full3D_2021May_minN = [ (1.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.5, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 10.0, 1.0), (0.0, 0.0, 1.0) ]



def nice_coupling_string(coupling):
    str_list = []
    for kappa in coupling:
        if type(kappa) == int or kappa.is_integer():
            str_list.append( f'{int(kappa): 3d}  ' )
        else:
            str_list.append( f'{kappa: 5.1f}' )
    coupling_string = f'{str_list[0]}, {str_list[1]}, {str_list[2]}'
    return coupling_string



def is_valid_combination(basis_parameters, base_equations=full_scan_terms):
    basis_states = [ [ sympy.Rational(param) for param in basis ] for basis in basis_parameters ]
    combination_matrix = sympy.Matrix([ [ f(*base) for f in base_equations ] for base in basis_states])
    return combination_matrix.det() != 0



def orthogonality_metric(basis_parameters, base_equations=full_scan_terms):
    #kv_val = sympy.Symbol('kvA')
    #k2v_val = sympy.Symbol('k2vB')
    #kl_val = sympy.Symbol('klC')
    #kv_range = (-kv_val, kv_val)
    #k2v_range = (-k2v_val, k2v_val)
    #kl_range = (-kl_val, kl_val)

    kv_range = (.8, 1.2)
    k2v_range = (-2, 4)
    kl_range = (-15, 15)

    basis_states = [ [ sympy.Rational(param) for param in basis ] for basis in basis_parameters ]
    combination_matrix = sympy.Matrix([ [ g(*base) for g in base_equations ] for base in basis_states])

    if combination_matrix.det() == 0: return None

    inversion = combination_matrix.inv()
    term_vector = sympy.Matrix([ [g(_k2v,_kl,_kv)] for g in base_equations ])

    #print('\n\n\n\nBASIS FUNCTION VECTOR')
    basis_function_vector = inversion * term_vector
    #sympy.pprint(basis_function_vector)

    #print('NORMALIZED VECTOR')
    normalized_basis_vector = sympy.Matrix([ fi / sympy.sqrt(sympy.integrate(fi*fi, (_kv, *kv_range), (_k2v, *k2v_range), (_kl, *kl_range) )) for fi in basis_function_vector])
    #sympy.pprint(normalized_basis_vector)

    #print('SCALAR PRODUCT')
    scalar_product = sympy.Matrix([ [ sympy.integrate(fi*fj, (_kv, *kv_range), (_k2v, *k2v_range), (_kl, *kl_range) ) for fj in normalized_basis_vector ] for fi in normalized_basis_vector])
    sympy.pprint(scalar_product)

    #print('ORTHOGONALITY METRIC')
    identity = sympy.eye(len(base_equations))
    orthogonality_metric = abs(sympy.det( scalar_product - identity ))
    sympy.pprint(orthogonality_metric)


    return orthogonality_metric


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
            substitutions = [ (a, r'sigma'f'({i},{j},{k})') for a, (i,j,k) in zip(amplitudes, basis_states) ]
            #substitutions = [ (a, 'blah') for a, (i,j,k) in zip(amplitudes, basis_parameters) ]
            out_equation = final_amplitude.subs(substitutions)
            with open('final_amplitude_'+name+'.tex','w') as output:
                formatted_equation = sympy.latex(out_equation)
                formatted_equation = formatted_equation.replace(r'\sigma', r'\times \sigma')
                formatted_equation = re.sub(r'(\\sigma\{\\left\([^,]*,[^,]*,[^\\]*\\right\)\} \+)', r'\1$\n\n$', formatted_equation)
                formatted_equation = '$'+formatted_equation+'$\n'
                print('\n'+formatted_equation)
                output.write(formatted_equation)

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

   #amp_function = get_amplitude_function(basis_list, preloaded_amplitudes=theory_xsecs, output='ascii')
    amp_function = get_amplitude_function(basis_list)
    theory_xsec_function = lambda couplings: amp_function(*couplings, *theory_xsecs)
    return theory_xsec_function



def get_inversion_vector(couplings):
    basis_states = [ [ sympy.Rational(param) for param in basis ] for basis in couplings ]
    combination_matrix = sympy.Matrix([ [ g(*base) for g in full_scan_terms ] for base in basis_states])
    if combination_matrix.det() == 0: return None
    inversion = numpy.array(combination_matrix.inv().tolist()).astype(numpy.float64)
    theory_xsec_function = get_theory_xsec_function()
    xsec_vector = numpy.array([ theory_xsec_function(c) for c in couplings ])

    inversion_array = (xsec_vector * inversion.transpose()).transpose()
    #numpy.set_printoptions(formatter={'float':lambda n: f'{n: 4.2f}'})
    return inversion_array


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
    current_states = [
        (1    , 1   , 1   ),
        (0    , 1   , 1   ),
        (0.5  , 1   , 1   ),
        (1.5  , 1   , 1   ),
        (2    , 1   , 1   ),
        (3    , 1   , 1   ),
        (1    , 0   , 1   ),
        (1    , 2   , 1   ),
        (1    , 10  , 1   ),
        (1    , 1   , 0.5 ),
        (1    , 1   , 1.5 ),
        (0    , 0   , 1   )
    ]

    old_states = [ #k2v, kl, kv
        [1  , 1   , 1   ], # 450044
        [1  , 2   , 1   ], # 450045
        [2  , 1   , 1   ], # 450046
        [1.5, 1   , 1   ], # 450047
        #[1  , 1   , 0.5 ], # 450048 - !!
        [0.5, 1   , 1   ], # 450049
        [0  , 1   , 1   ], # 450050
        [0  , 1   , 0.5 ], # 450051 - !!
        [1  , 0   , 1   ], # 450052 - ***
        #[0  , 0   , 1   ], # 450053 - !!
        [4  , 1   , 1   ], # 450054
        [1  , 10  , 1   ] # 450055 - ***
        #[1  , 1   , 1.5 ]  # 450056 - !!
    ]

    current_3D_reco_basis = [ #k2v, kl, kv
       [1    , 1   , 1   ],
       [2    , 1   , 1   ],
       [1.5  , 1   , 1   ],
       [0    , 1   , 0.5 ],
       [1    , 0   , 1   ],
       [1    , 10  , 1   ]
    ]

    new_3D_reco_basis = [ #k2v, kl, kv
        [1  , 1   , 1  ],
        [0  , 1   , 0.5],
        [1  , 0   , 1  ],
        [1  , 10  , 1  ],
        [0.5, 1   , 1  ],
        [4  , 1   , 1  ]
    ]

    #orthogonality_metric(current_3D_reco_basis)
    #orthogonality_metric(new_3D_reco_basis)
    qk_basis = [
        (3,1,1),
        (2,1,1),
        (1.5,1,1)
    ]
    #get_amplitude_function(qk_basis, base_equations=k2v_scan_terms, name='3d_temp', output='tex')

    recommended3D_basis_202106 = [
        (1.0, 1.0, 1.0),
        (0.5, 1.0, 1.0),
        (3.0, 1.0, 1.0),
        (1.0, 2.0, 1.0),
        (1.0, 10.0, 1.0),
        (0.0, 0.0, 1.0)
    ]
    combination_function = get_amplitude_function(recommended3D_basis_202106, as_scalar=True)
    theory_function = get_theory_xsec_function()

    import fileio_utils
    lumi_total = 3.2 + 24.6 + 43.65 + 58.45
    data_files = fileio_utils.read_coupling_file()
    all_cutflows = fileio_utils.get_combined_cutflow_values(data_files.keys(), data_files)
    all_theory_xsec = [ theory_function(basis)*lumi_total for basis in current_states ]
    all_init_xsec = [ all_cutflows[basis]['Initial'] for basis in current_states ]
    all_final_xsec = [ all_cutflows[basis]['Final'] for basis in current_states ]
    all_num_events = [ all_cutflows[basis]['FinalCount'] for basis in current_states ]


    init_xsec = [ all_cutflows[basis]['Initial'] for basis in recommended3D_basis_202106 ]
    final_xsec = [ all_cutflows[basis]['Final'] for basis in recommended3D_basis_202106 ]
    num_events = [ all_cutflows[basis]['FinalCount'] for basis in recommended3D_basis_202106 ]

    sampleA = (3, -9, 1)
    sampleB = (1, -5, 0.5)
    theoryA = theory_function(sampleA)*lumi_total
    theoryB = theory_function(sampleB)*lumi_total
    init_xsecA = combination_function(*sampleA, *init_xsec)
    init_xsecB = combination_function(*sampleB, *init_xsec)
    final_xsecA = combination_function(*sampleA, *final_xsec)
    final_xsecB = combination_function(*sampleB, *final_xsec)
    numA = combination_function(*sampleA, *num_events)
    numB = combination_function(*sampleB, *num_events)
    pretty_print = lambda basis, theory, init, final, num: print(f'{basis[0]:4.1f}, {basis[1]:4.1f}, {basis[2]:4.1f}, {int(theory):4d}, {int(init):7d}, {final:5.2f}, {1e4*final/theory:5.2f}, {int(num):6d}')
    print('    BASIS   , THRY, INTIAL , FINAL, ACCEP, COUNT ')
    for basis, theory, init, final, num in zip(current_states, all_theory_xsec, all_init_xsec, all_final_xsec, all_num_events):
        pretty_print(basis, theory, init, final, num)
    print()
    pretty_print(sampleA, theoryA, init_xsecA, final_xsecA, numA)
    pretty_print(sampleB, theoryB, init_xsecB, final_xsecB, numB)




if __name__ == '__main__': main()
