import numpy

import combination_utils

_scan_terms = combination_utils.full_scan_terms

def obtain_linear_combination(coupling_parameters, combination_function, coefficient_function, basis_error_list):
        linearly_combined_weights = combination_function(coupling_parameters)
        vector_coefficients = coefficient_function(coupling_parameters)
        error_term_list = [ (c*err)**2 for c, err in zip(vector_coefficients, basis_error_list) ]
        linearly_combined_errors = numpy.sqrt( sum(error_term_list) )
        return linearly_combined_weights, linearly_combined_errors
    


def truth_truth_reweight( basis_parameters, combination_components, coupling_parameters,
                events, base_distro, key, bin_edges, base_equations=_scan_terms):

    combination_function, basis_weight_list, basis_error_list, base_edge_list = combination_components

    linear_combination = combination_function(coupling_parameters)
    reweight_vector = combination_utils.get_amplitude_function(basis_parameters, base_equations=base_equations, as_scalar=False)(*coupling_parameters)[0]

    truth_base_weights = basis_weight_list[0].copy()
    truth_base_weights[ truth_base_weights == 0. ] = float('inf') # Just to avoid Nan issues
    reweight_array = linear_combination / truth_base_weights

    events = events.copy()
    indices = numpy.digitize(events[base_distro],base_edge_list)-1
    events['reweight'] = events['weight'] * reweight_array[indices]

    events['kinematic error'] = (reweight_array[indices] * events['weight'])**2
    events['base SM error']  = ( (events['weight']/truth_base_weights[indices]) * (reweight_vector[0] - reweight_array[indices]) * basis_error_list[0][indices] )**2
    events['base other error'] = numpy.zeros(len(events))
    for i in range(len(reweight_vector)-1):
        events['base other error'] += ( (events['weight']/truth_base_weights[indices]) * reweight_vector[i+1] * basis_error_list[i+1][indices] )**2
    events['error'] = numpy.sqrt( events['kinematic error'] + events['base SM error'] + events['base other error'] )

    rw_weights = numpy.histogram(events[key], bins=bin_edges, weights=events['reweight'])[0]
    rw_errors = numpy.zeros( len(rw_weights) )
    for i in range(len(rw_errors)): rw_errors[i] = numpy.sqrt( sum(events[ indices == i ]['error']**2) )

    return rw_weights, rw_errors



def truth_reweight( basis_parameters, combination_components, coupling_parameters, reco_base_bins, mHH_edges, scan_terms=_scan_terms, normalize=False):
    combination_function, basis_weight_list, basis_error_list = combination_components
    basis_list = [ [eval(n) for n in b ] for b in basis_parameters ]

    linear_combination = combination_function(coupling_parameters)
    reweight_vector = combination_utils.get_amplitude_function(basis_parameters, base_equations=scan_terms, as_scalar=False)(*coupling_parameters)[0]

    truth_base_weights = basis_weight_list[0].copy()
    if normalize:
        linear_combination /= linear_combination.sum()
        truth_base_weights /= truth_base_weights.sum()
    truth_base_weights[ truth_base_weights == 0. ] = float('inf') # Just to avoid Nan issues
    reweight_array = linear_combination / truth_base_weights
    reco_truth_ratio = reco_base_bins[0] / truth_base_weights

    reweighted_reco_weights = reweight_array * reco_base_bins[0]

    base_error2 = ( reweight_array * reco_base_bins[1] )**2
    truth_error2 = reco_truth_ratio**2 * numpy.array([ (e*m)**2 for e,m in zip(basis_error_list[1:], reweight_vector[1:]) ]).sum(axis=0)
    base_truth_error2 = 0 #reco_truth_ratio**2 * ( reweight_vector[0] - reweight_array )**2
    combined_errors = numpy.sqrt( base_error2 + truth_error2 + base_truth_error2 )
    combined_errors = numpy.zeros( len(reweighted_reco_weights) )

    return reweighted_reco_weights, combined_errors



def reco_reweight(reweight_vector, coupling_parameters, base_weights, base_errors):
    multiplier_vector = reweight_vector(*coupling_parameters)[0]

    reweighted_weights = numpy.array([ w*m for w,m in zip(base_weights, multiplier_vector) ])
    linearly_combined_weights = reweighted_weights.sum(axis=0)

    #print(coupling_parameters)
    #print([int(v) for v in multiplier_vector])
    #for arr in reweighted_weights: print( ''.join([f'{int(a*100): 3d} ' for a in arr]) )
    #print()
    #print(''.join([f'{int(a*100): 3d} ' for a in linearly_combined_weights]) )

    reweighted_errors2 = numpy.array([ (w*m)**2 for w,m in zip(base_errors, multiplier_vector) ])
    linearly_combined_errors = numpy.sqrt( reweighted_errors2.sum(axis=0) )

    return linearly_combined_weights, linearly_combined_errors
