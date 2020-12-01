import sympy
import numpy
import itertools

_k2v = sympy.Symbol('\kappa_{2V}')
_kl = sympy.Symbol('\kappa_{\lambda}')
_kv = sympy.Symbol('\kappa_{V}')


def get_amplitude_function(name, base_equations, basis_parameters):
    if [1,1,1] not in basis_parameters: return 0
    if [0,1,0.5] not in basis_parameters: return 0
    basis_states = [ [ sympy.Rational(param) for param in basis ] for basis in basis_parameters ]

    kappa_matrix = sympy.Matrix([ [ g(*base) for g in base_equations ] for base in basis_states])
    #sympy.pprint(kappa_matrix)

    if kappa_matrix.det() == 0: return 0
    else: 
        print(basis_parameters)
        return 1

    inversion = kappa_matrix.inv()
    kappa = sympy.Matrix([ [g(_k2v,_kl,_kv)] for g in base_equations ])
    amplitudes = sympy.Matrix([ sympy.Symbol(f'A{n}') for n in range(len(base_equations)) ])

    ### THIS EQUATION IS THE CENTRAL POINT OF THIS ENTIRE PROGRAM! ###
    final_amplitude = (kappa.T*inversion*amplitudes)[0]
    # FYI, numpy outputs a 1x1 matrix here, so I use the [0] to get just the equation
    ##################################################################

    sympy.pprint(final_amplitude)
    with open('final_amplitude_'+name+'.tex','w') as output:
        output.write('$\n'+sympy.latex(final_amplitude)+'\n$\n')

    amplitude_function = sympy.lambdify([_k2v, _kl, _kv]+[*amplitudes], final_amplitude, 'numpy')
    #return amplitude_function
    return 1




def main():
    full_equation_list = [
        lambda k2v,kl,kv: kv**2 * kl**2
       ,lambda k2v,kl,kv: kv**4
       ,lambda k2v,kl,kv: k2v**2
       ,lambda k2v,kl,kv: kv**3 * kl
       ,lambda k2v,kl,kv: k2v * kl * kv
       ,lambda k2v,kl,kv: kv**2 * k2v
    ]

    #full_basis_states = [
    #    ('1'  , '1', '1'  ),
    #    ('1'  , '0', '-1' ),
    #    ('0'  , '1', '1'  ),
    #    ('3/2', '1', '1'  ),
    #    ('1'  , '2', '1'  ),
    #    ('2'  , '1', '-1' )
    #]

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

    #possible_validation_combinations = itertools.combinations(validation_states,6)
    #total_possible = 0
    #for combination in possible_validation_combinations:
    #    total_possible += get_amplitude_function(str(combination), full_equation_list, combination)
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
    ##get_amplitude_function('validation', full_equation_list, validation_basis)

    #existing_states = [ #k2v, kl, kv
    #    [1  , 1   , 1   ], # 450044
    #    #[1  , 2   , 1   ], # 450045
    #    [2  , 1   , 1   ], # 450046
    #    #[1.5, 1   , 1   ], # 450047
    #    #[1  , 1   , 0.5 ], # 450048 - !!
    #    [0.5, 1   , 1   ], # 450049
    #    #[0  , 1   , 1   ], # 450050
    #    [0  , 1   , 0.5 ], # 450051 - !!
    #    [1  , 0   , 1   ], # 450052 - ***
    #    #[0  , 0   , 1   ], # 450053 - !!
    #    #[4  , 1   , 1   ], # 450054
    #    [1  , 10  , 1   ], # 450055 - ***
    #    #[1  , 1   , 1.5 ]  # 450056 - !!
    #]

    ##basis_states = [ [ sympy.Rational(param) for param in basis ] for basis in existing_states ]
    ##kappa_matrix = sympy.Matrix([ [ g(*base) for g in full_equation_list ] for base in basis_states])
    ##sympy.pprint(kappa_matrix)

    #possible_existing_combinations = itertools.combinations(existing_states,6)
    #total_possible = 0
    #for combination in possible_existing_combinations:
    #    total_possible += get_amplitude_function(str(combination), full_equation_list, combination)
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
    #get_amplitude_function('existing', full_equation_list, existing_basis)



    kl_equation_list = [
        lambda k2v,kl,kv: kl**2
       ,lambda k2v,kl,kv: kl
       ,lambda k2v,kl,kv: 1
    ]
    kl_basis_states = [
        ('0  ' , '0 '  , '1  ' ),
        ('1  ' , '0 '  , '1  ' ),
        ('0  ' , '1 '  , '1  ' ),
        ('0  ' , '1 '  , '0.5' ),
        ('0.5' , '1 '  , '1  ' ),
        ('1  ' , '1 '  , '0.5' ),
        ('1  ' , '1 '  , '1  ' ),
        ('1.5' , '1 '  , '1  ' ),
        ('2  ' , '1 '  , '1  ' ),
        ('4  ' , '1 '  , '1  ' ),
        ('1  ' , '2 '  , '1  ' ),
        ('1  ' , '10'  , '1  ' ),
        ('1  ' , '11'  , '1.5' )
    ]
    #possible_existing_combinations = itertools.combinations(existing_states,6)
    #total_possible = 0
    #for combination in possible_existing_combinations:
    #    total_possible += get_amplitude_function(str(combination), full_equation_list, combination)
    #print()
    #print(total_possible)

    k2v_equation_list = [
        lambda k2v,kl,kv: k2v**2
       ,lambda k2v,kl,kv: k2v
       ,lambda k2v,kl,kv: 1
    ]
    k2v_basis_states = [
        ('1'  , '1', '1'  ),
        ('0'  , '1', '1'  ),
        ('2'  , '1', '1' )
    ]

    kl_k2v_equation_list = [
        lambda k2v,kl,kv: kl**2
       ,lambda k2v,kl,kv: k2v**2
       ,lambda k2v,kl,kv: kl
       ,lambda k2v,kl,kv: k2v * kl
       ,lambda k2v,kl,kv: k2v
       ,lambda k2v,kl,kv: 1
    ]
    kl_k2v_basis_states = [
        ('1'  , '1', '1'  ),
        ('-1'  , '0', '1' ),
        ('0'  , '1', '1'  ),
        ('3/2', '1', '1'  ),
        ('1'  , '3/2', '1'  ),
        ('1'  , '-1', '1' )
    ]

    #get_amplitude_function('all', full_equation_list, full_basis_states)
    #print('\n\n\n')
    #get_amplitude_function('kl', kl_equation_list, kl_basis_states)
    #print('\n\n\n')
    #get_amplitude_function('k2v', k2v_equation_list, k2v_basis_states)
    #get_amplitude_function('kl_k2v', kl_k2v_equation_list, kl_k2v_basis_states)


if __name__ == '__main__': main()
