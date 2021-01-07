import math
import numpy
import uproot

import combination_utils

_scan_terms = combination_utils.full_scan_terms

##########################################
#######     Data Retrieval         #######
##########################################

def extract_lhe_events(rootfile, key_list):
    ttree = uproot.rootio.open(rootfile)['tree']
    events = ttree.pandas.df(branches=['weight',*key_list])
    return events



def retrieve_lhe_weights(lhe_events, kinematic_variable, bin_edges):
    event_weights = lhe_events['weight'].values
    event_kinematics = lhe_events[kinematic_variable].values
    reco_weights = numpy.histogram(event_kinematics, weights=event_weights, bins=bin_edges)[0]
    reco_errors = numpy.zeros( len(reco_weights) )
    event_bins = numpy.digitize(event_kinematics,bin_edges)
    for i in range(len(reco_errors)):
        binned_weights = event_weights[ event_bins == i ]
        error2_array = binned_weights**2
        error = math.sqrt( error2_array.sum() )
        reco_errors[i] = error
    return reco_weights, reco_errors



def extract_lhe_truth_data(file_list, mHH_edges):
    weight_list, error_list = [], []
    for f in file_list:
        events = extract_lhe_events(f,['HH_m'])
        weights, errors = retrieve_lhe_weights(events, 'HH_m', mHH_edges)
        weight_list.append(weights)
        error_list.append(errors)
    return weight_list, error_list



def extract_truth_data(file_list, hist_key=b'HH_m'):
    yedge = 2000
    final_bin = None
    weight_list = []
    error_list = []
    edge_list = []
    for f in file_list:
        directory = uproot.open(f)
        root_hist = directory[hist_key]
        weights, edges = root_hist.numpy()
        errors = numpy.sqrt(root_hist.variances)
        if final_bin == None:
            final_bin = numpy.argmax(edges > yedge)
            edge_list = edges[:final_bin]

        weight_list.append(weights[:final_bin-1])
        error_list.append(errors[:final_bin-1])
    return weight_list, error_list, edge_list



def retrieve_lhe_truth_combination(amplitude_function, basis_files, mHH_edges):
    basis_weight_list, basis_error_list = extract_lhe_truth_data(basis_files, mHH_edges)
    combination_function = lambda params: amplitude_function(*params, *basis_weight_list)
    return combination_function, basis_weight_list, basis_error_list, mHH_edges


def extract_ntuple_events(ntuple, mhh_key='m_hh', unit_conversion=1):
    ttree = uproot.rootio.open(ntuple)[b'sig']
    frame = ttree.pandas.df(branches=[mhh_key,'mc_sf'])
    #frame = ttree.pandas.df(branches=[mhh_key,'mc_sf','pass_vbf_sel'])
    #frame = frame[ frame['pass_vbf_sel'] ]
    masses  = frame[mhh_key][:,0].values * unit_conversion
    weights = frame['mc_sf'][:,0].values
    #weights *= 58.45 #for MC16e
    events = (masses,weights)
    return events


def retrieve_reco_weights(mHH_edges, reco_events):
    event_weights = reco_events[1]
    reco_weights = numpy.histogram(reco_events[0], bins=mHH_edges, weights=event_weights)[0]
    reco_errors = numpy.zeros( len(reco_weights) )
    event_bins = numpy.digitize(reco_events[0],mHH_edges)
    for i in range(len(reco_errors)):
        binned_weights = event_weights[ event_bins == i ]
        error2_array = binned_weights**2
        error = math.sqrt( error2_array.sum() )
        reco_errors[i] = error

    return reco_weights, reco_errors



##########################################
### Reweighting/combination Techniques ###
##########################################


def obtain_linear_combination(coupling_parameters, combination_function, coefficient_function, basis_error_list):
        linearly_combined_weights = combination_function(coupling_parameters)
        vector_coefficients = coefficient_function(coupling_parameters)
        error_term_list = [ (c*err)**2 for c, err in zip(vector_coefficients, basis_error_list) ]
        linearly_combined_errors = numpy.sqrt( sum(error_term_list) )
        return linearly_combined_weights, linearly_combined_errors
    


def truth_truth_reweight( basis_parameters, combination_components, coupling_parameters,
                events, base_distro, key, bin_edges):

    combination_function, basis_weight_list, basis_error_list, base_edge_list = combination_components

    linear_combination = combination_function(coupling_parameters)
    reweight_vector = combination_utils.get_amplitude_function(basis_parameters, base_equations=_scan_terms, as_scalar=False)(*coupling_parameters)[0]

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



def truth_reweight( basis_parameters, combination_components, coupling_parameters, reco_base_bins, mHH_edges):
    combination_function, basis_weight_list, basis_error_list = combination_components
    basis_list = [ [eval(n) for n in b ] for b in basis_parameters ]

    linear_combination = combination_function(coupling_parameters)
    reweight_vector = combination_utils.get_amplitude_function(basis_parameters, base_equations=_scan_terms, as_scalar=False)(*coupling_parameters)[0]

    truth_base_weights = basis_weight_list[0].copy()
    truth_base_weights[ truth_base_weights == 0. ] = float('inf') # Just to avoid Nan issues
    reweight_array = linear_combination / truth_base_weights
    reco_truth_ratio = reco_base_bins[0] / truth_base_weights

    reweighted_reco_weights = reweight_array * reco_base_bins[0]

    base_error2 = ( reweight_array * reco_base_bins[1] )**2
    truth_error2 = reco_truth_ratio**2 * numpy.array([ (e*m)**2 for e,m in zip(basis_error_list[1:], reweight_vector[1:]) ]).sum(axis=0)
    base_truth_error2 = 0 #reco_truth_ratio**2 * ( reweight_vector[0] - reweight_array )**2
    combined_errors = numpy.sqrt( base_error2 + truth_error2 + base_truth_error2 )

    return reweighted_reco_weights, combined_errors



def reco_reweight(mHH_edges, reweight_vector, coupling_parameters, base_weights, base_errors):
    multiplier_vector = reweight_vector(*coupling_parameters)[0]

    reweighted_weights = numpy.array([ w*m for w,m in zip(base_weights, multiplier_vector) ])
    linearly_combined_weights = reweighted_weights.sum(axis=0)

    reweighted_errors2 = numpy.array([ (w*m)**2 for w,m in zip(base_errors, multiplier_vector) ])
    linearly_combined_errors = numpy.sqrt( reweighted_errors2.sum(axis=0) )

    return linearly_combined_weights, linearly_combined_errors
