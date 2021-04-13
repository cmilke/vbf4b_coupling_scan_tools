import math
import numpy
import uproot3 as uproot
from uproot_methods import TLorentzVector as LV

def extract_lhe_events(rootfile, key_list):
    ttree = uproot.rootio.open(rootfile)['tree']
    events = ttree.pandas.df(branches=['weight',*key_list])
    return events



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



def extract_ntuple_events(ntuple, key=None, unit_conversion=None, tree_name=b'sig', filter_vbf=True, lumi_weight=True):
    ttree = uproot.rootio.open(ntuple)[tree_name]
    branches = [key,'mc_sf']
    if filter_vbf: branches.append('pass_vbf_sel')
    if lumi_weight: branches.append('run_number')

    frame = ttree.pandas.df(branches=branches)
    if filter_vbf: frame = frame[ frame['pass_vbf_sel'] ]

    unit_conversion = 1
    if key == 'truth_mhh': unit_conversion = 1/1000
    vals  = frame[key].values * unit_conversion
    weights = frame['mc_sf'].values
    if lumi_weight:
        run_number = frame['run_number']
        mc2015 = run_number < 296939
        mc2016 = numpy.logical_and(296939 < run_number, run_number < 320000)
        mc2017 = numpy.logical_and(320000 < run_number, run_number < 350000)
        mc2018 = numpy.logical_and(350000 < run_number, run_number < 370000)
        weights[mc2015] *=  3.2
        weights[mc2016] *= 24.6
        weights[mc2017] *= 43.65
        weights[mc2018] *= 58.45

    events = numpy.array([vals,weights])
    return events



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


def read_coupling_file(coupling_file):
    data_files = {}
    with open(coupling_file) as coupling_list:
        for line in coupling_list:
            if line.strip().startswith('#'): continue
            linedata = line.split()
            couplings = tuple([ float(p) for p in linedata[:3] ])
            data_file = linedata[3]
            if couplings not in data_files:
                data_files[couplings] = [data_file]
            else:
                data_files[couplings].append(data_file)
    return data_files


def get_events(parameter_list, data_files, reco=True):
    events_list = []
    for couplings in parameter_list:
        new_events = []
        for f in data_files[couplings]:
            new_events.append( extract_ntuple_events(f,key='m_hh',filter_vbf=False) )
        events = numpy.concatenate(new_events, axis=1)
        events_list.append(events)
    return events_list
