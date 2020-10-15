import sympy
import numpy

_k2v = sympy.Symbol('\kappa_{2V}')
_kl = sympy.Symbol('\kappa_{\lambda}')
_kv = sympy.Symbol('\kappa_{V}')


def get_amplitude_function(name, base_equations, basis_parameters):
    basis_states = [ [ sympy.Rational(param) for param in basis ] for basis in basis_parameters ]



    kappa_matrix = sympy.Matrix([ [ g(*base) for g in base_equations ] for base in basis_states])
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
    return amplitude_function



def main():
    full_equation_list = [
        lambda k2v,kl,kv: kv**2 * kl**2
       ,lambda k2v,kl,kv: kv**4
       ,lambda k2v,kl,kv: k2v**2
       ,lambda k2v,kl,kv: kv**3 * kl
       ,lambda k2v,kl,kv: k2v * kl * kv
       ,lambda k2v,kl,kv: kv**2 * k2v
    ]
    full_basis_states = [
        ('1'  , '1', '1'  ),
        ('1'  , '0', '-1' ),
        ('0'  , '1', '1'  ),
        ('3/2', '1', '1'  ),
        ('1'  , '2', '1'  ),
        ('2'  , '1', '-1' )
    ]


    kl_equation_list = [
        lambda k2v,kl,kv: kl**2
       ,lambda k2v,kl,kv: kl
       ,lambda k2v,kl,kv: 1
    ]
    kl_basis_states = [
        ('1'  , '1', '1'  ),
        ('1'  , '0', '1' ),
        ('1'  , '2', '1'  ),
    ]

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

    get_amplitude_function('all', full_equation_list, full_basis_states)
    print('\n\n\n')
    get_amplitude_function('kl', kl_equation_list, kl_basis_states)
    print('\n\n\n')
    get_amplitude_function('k2v', k2v_equation_list, k2v_basis_states)
    get_amplitude_function('kl_k2v', kl_k2v_equation_list, kl_k2v_basis_states)


if __name__ == '__main__': main()
