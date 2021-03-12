import math
import numpy
import uproot3 as uproot
from uproot_methods import TLorentzVector as LV

import combination_utils

_scan_terms = combination_utils.full_scan_terms

##########################################
#######     Data Retrieval         #######
##########################################

def extract_lhe_events(rootfile, key_list):
    ttree = uproot.rootio.open(rootfile)['tree']
    events = ttree.pandas.df(branches=['weight',*key_list])
    return events



def retrieve_lhe_dual_weights(lhe_events, var_edges=None, kin_vars=None):
    event_weights = lhe_events['weight'].values

    if 'dEta_hh' in kin_vars: lhe_events['dEta_hh'] = abs(lhe_events['dEta_hh'])

    event_kinematics0 = lhe_events[kin_vars[0]].values
    event_kinematics1 = lhe_events[kin_vars[1]].values
    weights = numpy.histogram2d(event_kinematics0, event_kinematics1, weights=event_weights, bins=var_edges)[0]
    return weights



def retrieve_lhe_weights(lhe_events, kinematic_variable, bin_edges):
    event_weights = lhe_events['weight'].values
    event_kinematics = lhe_events[kinematic_variable].values
    weights = numpy.histogram(event_kinematics, weights=event_weights, bins=bin_edges)[0]
    errors = numpy.zeros( len(weights) )
    event_bins = numpy.digitize(event_kinematics,bin_edges)-1
    for i in range(len(errors)):
        binned_weights = event_weights[ event_bins == i ]
        error2_array = binned_weights**2
        error = math.sqrt( error2_array.sum() )
        errors[i] = error
    return weights, errors



def extract_lhe_truth_weight(lhe_file, edges, key='HH_m', force_cap = None):
    events = extract_lhe_events(lhe_file,[key])
    if type(force_cap) != None: events = events[:force_cap]
    weights = retrieve_lhe_weights(events, key, edges)[0]
    return weights



def extract_lhe_truth_dual_weight(lhe_file, var_edges=None, kin_vars=None, **kwargs):
    branches = list(kin_vars)

    if 'ptsumjj' in kin_vars:
        branches[kin_vars.index('ptsumjj')] = 'j1_pt'
        branches.append('j2_pt')

    if 'jj_Deta' in kin_vars:
        branches[kin_vars.index('jj_Deta') ] = 'j1_eta'
        branches.append('j2_eta')

    ttree = uproot.rootio.open(lhe_file)['tree']
    events = ttree.pandas.df(branches=['weight',*branches])

    if 'ptsumjj' in kin_vars: events['ptsumjj'] = events['j1_pt'] + events['j2_pt'] 
    if 'jj_Deta' in kin_vars: events['jj_Deta'] = abs(events['j1_eta'] - events['j2_eta'])
    if 'dEta_hh' in kin_vars: events['dEta_hh'] = abs(events['dEta_hh'])

    event_weights = events['weight'].values
    event_kinematics0 = events[kin_vars[0]].values
    event_kinematics1 = events[kin_vars[1]].values
    weights = numpy.histogram2d(event_kinematics0, event_kinematics1, weights=event_weights, bins=var_edges)[0]

    return weights



def extract_lhe_truth_data(file_list, mHH_edges, normalize=False):
    weight_list, error_list = [], []
    for f in file_list:
        events = extract_lhe_events(f,['HH_m'])
        weights, errors = retrieve_lhe_weights(events, 'HH_m', mHH_edges)
        if normalize:
            norm = weights.sum()
            weights /= norm
            errors /= norm
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



def extract_ntuple_events(ntuple, key=None, unit_conversion=None, tree_name=b'sig', filter_vbf=True):
    ttree = uproot.rootio.open(ntuple)[tree_name]
    if filter_vbf:
        frame = ttree.pandas.df(branches=[key,'mc_sf','pass_vbf_sel'])
        frame = frame[ frame['pass_vbf_sel'] ]
    else:
        frame = ttree.pandas.df(branches=[key,'mc_sf'])
    unit_conversion = 1
    if key == 'truth_mhh': unit_conversion = 1/1000
    vals  = frame[key].values * unit_conversion
    weights = frame['mc_sf'].values
    #weights *= 58.45 #for MC16e
    events = (vals,weights)
    return events



def extract_reco_weight(reco_file, edges, key='m_hh', unit_conversion=1, tree_name=b'sig', filter_vbf=True):
    reco_events = extract_ntuple_events(reco_file, key=key, unit_conversion=unit_conversion, tree_name=tree_name, filter_vbf=filter_vbf)
    event_weights = reco_events[1]
    reco_weights = numpy.histogram(reco_events[0], bins=edges, weights=event_weights)[0]
    return reco_weights



def extract_reco_dual_weight(reco_file, tree_name=b'sig', filter_vbf=True, **kwargs):
    ttree = uproot.rootio.open(reco_file)[tree_name]
    kin_vars = kwargs['kin_vars']
    branch_list = list(kin_vars)
    branch_list.append('mc_sf')
    if 'jj_pTvecsum' in kin_vars:
        branch_list[kin_vars.index('jj_pTvecsum')] = 'pT_vbf_j1'
        branch_list.append('eta_vbf_j1')
        branch_list.append('phi_vbf_j1')
        branch_list.append('E_vbf_j1')
        branch_list.append('pT_vbf_j2')
        branch_list.append('eta_vbf_j2')
        branch_list.append('phi_vbf_j2')
        branch_list.append('E_vbf_j2')
    if 'ptsumjj' in kin_vars:
        branch_list[kin_vars.index('ptsumjj')] = 'pT_vbf_j1'
        branch_list.append('pT_vbf_j2')

    if filter_vbf: branch_list.append('pass_vbf_sel')
    frame = ttree.pandas.df(branches=branch_list)
    if filter_vbf: frame = frame[ frame['pass_vbf_sel'] ]

    if 'ptsumjj' in kin_vars: frame['ptsumjj'] = frame['pT_vbf_j1'] + frame['pT_vbf_j2'] 
    if 'truth_mhh' in kin_vars: frame['truth_mhh'] *= 1/1000
    if 'jj_pTvecsum' in kin_vars:
        frame['j1vec'] = frame.apply( lambda row : LV.from_ptetaphie(row['pT_vbf_j1'], row['eta_vbf_j1'], row['phi_vbf_j1'], row['E_vbf_j1']), axis=1 )
        frame['j2vec'] = frame.apply( lambda row : LV.from_ptetaphie(row['pT_vbf_j2'], row['eta_vbf_j2'], row['phi_vbf_j2'], row['E_vbf_j2']), axis=1 )
        frame['jj_pTvecsum'] = frame.apply( lambda row : (row['j1vec'] + row['j2vec']).pt, axis=1 )


    var0 = frame[kin_vars[0]].values
    var1 = frame[kin_vars[1]].values
    weights = frame['mc_sf'].values
    events = (var0,var1,weights)
    reco_weights = numpy.histogram2d(var0, var1, bins=kwargs['var_edges'], weights=weights)[0]
    return reco_weights



def retrieve_reco_weights(var_edges, reco_events):
    event_weights = reco_events[1]
    reco_weights = numpy.histogram(reco_events[0], bins=var_edges, weights=event_weights)[0]
    reco_errors = numpy.zeros( len(reco_weights) )
    event_bins = numpy.digitize(reco_events[0],var_edges)-1
    for i in range(len(reco_errors)):
        binned_weights = event_weights[ event_bins == i ]
        error2_array = binned_weights**2
        error = math.sqrt( error2_array.sum() )
        reco_errors[i] = error

    return [reco_weights, reco_errors]



def get_cutflow_values(filename, hist_name='FourTagCutflow'):
    directory = uproot.open(filename)
    cutflow_hist = directory[hist_name]
    labeled_values = { k:v for k,v in zip(cutflow_hist.xlabels, cutflow_hist.values) }
    return labeled_values



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

    reweighted_errors2 = numpy.array([ (w*m)**2 for w,m in zip(base_errors, multiplier_vector) ])
    linearly_combined_errors = numpy.sqrt( reweighted_errors2.sum(axis=0) )

    return linearly_combined_weights, linearly_combined_errors
