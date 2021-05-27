import math
import numpy
#import uproot3 as uproot
import uproot

coupling_file = 'basis_files/nnt_coupling_file_2021May_crypto.dat'

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



def extract_ntuple_events(ntuple, key=None, tree_name=None):
    #tree_name = 'sig_highPtcat'
    tree_name = 'sig'

    rootfile = uproot.open(ntuple)
    #DSID = rootfile['DSID']._members['fVal']
    #nfiles = 1
    #while( DSID / nfiles > 600050 ): nfiles += 1
    #DSID = int(DSID / nfiles)
    #print(ntuple, DSID)
    ttree = rootfile[tree_name]

    #if tree_name == 'sig':
    #if True:
    if False:
        kinvals = ttree['m_hh'].array()
        weights = ttree['mc_sf'].array()[:,0]
        run_number = ttree['run_number'].array()
    else: # Selections
        pass_vbf_sel = ttree['pass_vbf_sel'].array()
        x_wt_tag = ttree['X_wt_tag'].array() > 1.5
        ntag = ttree['ntag'].array() >= 4
        valid_event = numpy.logical_and.reduce( (pass_vbf_sel, x_wt_tag, ntag) )

        kinvals = ttree['m_hh'].array()[valid_event]
        weights = ttree['mc_sf'].array()[:,0][valid_event]
        run_number = ttree['run_number'].array()[valid_event]

    mc2015 = ( run_number < 296939 ) * 3.2
    mc2016 = ( numpy.logical_and(296939 < run_number, run_number < 320000) ) * 24.6
    mc2017 = ( numpy.logical_and(320000 < run_number, run_number < 350000) ) * 43.65
    mc2018 = ( numpy.logical_and(350000 < run_number, run_number < 370000) ) * 58.45
    all_years = mc2015 + mc2016 + mc2017 + mc2018
    lumi_weights = weights * all_years

    events = numpy.array([kinvals,lumi_weights])
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


#FIXME: you can't just use the cutflow anymore because there are too many post nnt cuts
def get_combined_cutflow_values(parameter_list, data_files):
    combined_cutflows = {}
    for couplings in parameter_list:
        for f in data_files[couplings]:
            frame = uproot.rootio.open(f)[b'sig_highPtcat'].pandas.df(branches=['run_number'])
            run_number = frame['run_number'].values[0]
            lumi_weight = None
            if   run_number < 296939: lumi_weight = 3.2 # MC2015
            elif 296939 < run_number and run_number < 320000: lumi_weight = 24.6  # MC2016
            elif 320000 < run_number and run_number < 350000: lumi_weight = 43.65 # MC2017
            elif 350000 < run_number and run_number < 370000: lumi_weight = 58.45 # MC2018
            else:
                print("UNKNOWN RUN NUMBER!! -- " + str(run_number))
                exit(1)

            cutflows = get_cutflow_values(f)
            lumi_weighted_cutflows = { key:val*lumi_weight for key,val in cutflows.items() }
            if couplings not in combined_cutflows:
                combined_cutflows[couplings] = lumi_weighted_cutflows
            else:
                for key,val in lumi_weighted_cutflows.items():
                    combined_cutflows[couplings][key] += val
    return combined_cutflows


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
            new_events.append( extract_ntuple_events(f,key='m_hh') )
        events = numpy.concatenate(new_events, axis=1)
        events_list.append(events)
    return events_list
