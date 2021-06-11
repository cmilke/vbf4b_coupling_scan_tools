import sys
import argparse
import numpy
import itertools
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#import pdb

import combination_utils
import fileio_utils
from negative_weight_map import get_Nweight_sum, draw_error_map
import validate_linear_combinations


def draw_rankings(ranks_to_draw, valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range,
        name_infix, only_heatmap=False, skip_preview=False, truth_level=False, truth_data_files=None):

    print('Drawing basis ranks ' + str(ranks_to_draw))
    max_negative_weight = 0
    for rank in ranks_to_draw:
        basis = valid_bases[rank]
        nWeight_grid = get_Nweight_sum(basis[1], basis[2], kv_val, k2v_val_range, kl_val_range, grid=True)
        max_for_rank = numpy.max(nWeight_grid)
        if max_for_rank > max_negative_weight:
            max_negative_weight = max_for_rank

    for rank in ranks_to_draw:
        print('Drawing rank ' + str(rank) + '...')
        basis = valid_bases[rank]
        nWeight_grid = get_Nweight_sum(basis[1], basis[2], kv_val, k2v_val_range, kl_val_range, grid=True)
        draw_error_map(basis[1], var_edges, kv_val, k2v_val_range, kl_val_range, nWeight_grid, 
                vmax = max_negative_weight, name_suffix=f'_{name_infix}rank{int(rank)}', 
                title_suffix=f'Rank {rank+1}/{len(valid_bases)}, Integral={int(basis[0])}')

    if only_heatmap: return
    comp_couplings = [ valid_bases[ranks_to_draw[0]][1], valid_bases[ranks_to_draw[1]][1] ]
    if truth_level:
        data_files = fileio_utils.read_coupling_file(coupling_file='basis_files/truth_LHE_couplings_extended.dat')
    else:
        data_files = fileio_utils.read_coupling_file()
    validate_linear_combinations.compare_bases_reco_method(comp_couplings, list(data_files.keys()),
         name_suffix=f'_auto_{name_infix}_3D_{ranks_to_draw[0]}-{ranks_to_draw[1]}',
         labels=(f'Rank {ranks_to_draw[0]}', f'Rank {ranks_to_draw[1]}'),
         truth_level=truth_level, truth_data_files=truth_data_files )

    if skip_preview: return
    k2v_vals = [-1.5, 0.5, 2, 3.5]
    kl_vals = [-9, -3, 5, 14]
    preview_couplings = []
    for k2v in k2v_vals:
        for kl in kl_vals:
            preview_couplings.append( (k2v, kl, 1) )
    validate_linear_combinations.compare_bases_reco_method(comp_couplings, preview_couplings,
         name_suffix='_preview_auto_'+name_infix+'_3D_'f'{ranks_to_draw[0]}-{ranks_to_draw[1]}',
         labels=(f'Rank {ranks_to_draw[0]}', f'Rank {ranks_to_draw[1]}'), is_verification=False, truth_level=truth_level)


def optimize_reco( mode='reco', extra_files={}, extra_name='' ):
    var_edges = numpy.linspace(200, 1200, 31)
    kv_val = 1.0
    num_kappa_bins = 100
    k2v_val_range = numpy.linspace(-2,4,num_kappa_bins+1)
    kl_val_range = numpy.linspace(-14,16,num_kappa_bins+1)
    grid_pixel_area = (k2v_val_range[1] - k2v_val_range[0]) * (kl_val_range[1] - kl_val_range[0])

    truth_data_files = None
    if mode == 'reco':
        data_files = fileio_utils.read_coupling_file()
        all_events = fileio_utils.get_events(data_files.keys(), data_files)
        all_histograms = [ fileio_utils.retrieve_reco_weights(var_edges,events) for events in all_events ]
        all_weights, all_errors = numpy.array( list(zip(*all_histograms)) )
        # Wrap all variations up together with their histograms so I can find combinations
        all_variations = list(zip(data_files.keys(), all_weights))
    elif mode == 'truth':
        truth_data_files = fileio_utils.read_coupling_file(coupling_file='basis_files/truth_LHE_couplings.dat')
        truth_data_files.update(extra_files)
        truth_weights, truth_errors = fileio_utils.extract_lhe_truth_data(truth_data_files.values(), var_edges)
        all_variations = list(zip(truth_data_files.keys(), truth_weights))
    else:
        print('What are you doing?!')
        print(mode)
        exit(1)
    print('Histograms loaded, proceeding to integrate Nweight grids...')

    valid_bases = []
    total = 0
    for basis_set in itertools.combinations(all_variations,6):
        # Unwrap each combination
        couplings, weights = list(zip(*basis_set))
        if (1.0,1.0,1.0) not in couplings: continue
        if not combination_utils.is_valid_combination(couplings, base_equations=combination_utils.full_scan_terms): continue

        nWeight_integral = get_Nweight_sum(couplings, weights, kv_val, k2v_val_range, kl_val_range)
        #nWeight_integral = get_Nweight_sum(couplings, weights, kv_val, k2v_val_range, kl_val_range,
        #        mask=lambda k2v, kl: ((k2v-1)/1)**2 + ((kl-1)/10)**2 < 1 )
        valid_bases.append( (nWeight_integral, couplings, weights) )
        total += 1
        if total % 10 == 0: print(total)
    print('Integrals computed, sorting and printing...')
    valid_bases.sort()
    for rank, (integral, couplings, weight) in enumerate(valid_bases): print(rank, int(integral), couplings)

    #draw_rankings([0,1,2,3], valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, 'quad', only_heatmap=True)
    #draw_rankings([0,1], valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, 'top')

    #draw_rankings([0,1,2,3], valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, extra_name+'_truth_quad', only_heatmap=True)
    draw_rankings([0,1], valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, extra_name+'_truth_top', only_heatmap=False, truth_level=True, truth_data_files=truth_data_files, skip_preview=True)

    #draw_rankings([0,1,2,3], valid_bases, var_edges, kv_val, k2v_val_range, kl_val_range, 'masktop', only_heatmap=True)
    #combination_utils.get_amplitude_function(valid_bases[0][1], base_equations=combination_utils.full_scan_terms, name='optimal_3DR0', output='tex')
    #combination_utils.get_amplitude_function(valid_bases[1][1], base_equations=combination_utils.full_scan_terms, name='optimal_3DR1', output='tex')



def main():
    # Sort out command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( "--mode", required = False, default = 'reco', type=str,
        help = "Options are: 'truth' or 'reco'",) 

    args = parser.parse_args()

    #pdb.set_trace()
    #numpy.set_printoptions(precision=None, linewidth=400, threshold=10000, sign=' ', formatter={'float':lambda f: f'{int(f):2d}'}, floatmode='fixed')
    #numpy.set_printoptions(precision=1, linewidth=400, threshold=10000, sign=' ', floatmode='fixed')
    if args.mode == 'reco': optimize_reco()
    elif args.mode == 'truth': optimize_reco(mode='truth')
    elif args.mode == 'truth_iter':
        file_list = [
            { (3    ,  -9   ,  1   ):     ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo_l-9cvv3cv1-tree.root']},
            { (1    ,  -5   ,  0.5 ):     ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo_l-5cvv1cv0p5-tree.root']},
            #{ (3    ,  -6   ,  1  ):      ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo3_l-6cvv3cv1-tree.root']},
            #{ ( 2.5 , -4  , 1   ):      ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo3_l-4cvv2p5cv1-tree.root']},
            #{ (-0.5 ,  8    ,  0.5 ):     ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo_l8cvv-0p5cv0p5-tree.root']},
            #{ (  1  ,  7  , 1.5 ):      ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo3_l7cvv1cv1p5-tree.root']},
            #{ (3    , 4  ,  1.5  ):    ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo2_l4cvv3cv1p5-tree.root']},


            #{ (-1   ,  10   ,  0.5 ):     ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo_l10cvv-1cv0p5-tree.root']},
            #{ (1    ,  -3   ,  0.5 ):     ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo_l-3cvv1cv0p5-tree.root']},
            ##{ (1    ,  -5   ,  0.5 ):     ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo_l-5cvv1cv0p5-tree.root']},
            #{ (3    ,  -5   ,  1   ):     ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo_l-5cvv3cv1-tree.root']},
            ##{ (-0.5 ,  8    ,  0.5 ):     ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo_l8cvv-0p5cv0p5-tree.root']},
            #{ (1.5  ,  -9   ,  0.5 ):     ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo_l-9cvv1p5cv0p5-tree.root']},
            #{ (2    ,  -9   ,  0.5 ):     ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo_l-9cvv2cv0p5-tree.root']},
            ##{ (3    ,  -9   ,  1   ):     ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo_l-9cvv3cv1-tree.root']},
            #{ (3.5  ,  -9   ,  1   ):     ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo_l-9cvv3p5cv1-tree.root']},
            #{ (4    ,  -9   ,  1   ):     ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo_l-9cvv4cv1-tree.root']}

            #{ (0    , 10 ,  1.5  ):    ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo2_l10cvv0cv1p5-tree.root']},
            #{ (-0.5 , 10 ,  1.5  ):    ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo2_l10cvv-0p5cv1p5-tree.root']},
            #{ (1.5  , -1 ,  1    ):    ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo2_l-1cvv1p5cv1-tree.root']},
            #{ (0    , 2  ,  0.5  ):    ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo2_l2cvv0cv0p5-tree.root']},
            #{ (0.5  , -3 ,  0.5  ):    ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo2_l-3cvv0p5cv0p5-tree.root']},
            #{ (3.5  , -3 ,  1.5  ):    ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo2_l-3cvv3p5cv1p5-tree.root']},
            #{ (4    , -3 ,  1.5  ):    ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo2_l-3cvv4cv1p5-tree.root']},
            #{ (0    , 4  ,  0.5  ):    ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo2_l4cvv0cv0p5-tree.root']},
            #{ (0    , 4  ,  1    ):    ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo2_l4cvv0cv1-tree.root']},
            #{ (0.5  , 4  ,  1    ):    ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo2_l4cvv0p5cv1-tree.root']},
            #{ (2    , -5 ,  1    ):    ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo2_l-5cvv2cv1-tree.root']},
            #{ (2.5  , -5 ,  1    ):    ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo2_l-5cvv2p5cv1-tree.root']},
            #{ (3.5  , -5 ,  1.5  ):    ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo2_l-5cvv3p5cv1p5-tree.root']},
            #{ (4    , -5 ,  1.5  ):    ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo2_l-5cvv4cv1p5-tree.root']},
            #{ (0    , 8  ,  1.5  ):    ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo2_l8cvv0cv1p5-tree.root']},
            #{ (0    , 8  ,  1.   ):    ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo2_l8cvv0cv1-tree.root']},
            #{ (-0.5 , 8  ,  0.5  ):    ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo2_l8cvv-0p5cv0p5-tree.root']},
            #{ (0.5  , 8  ,  1.5  ):    ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo2_l8cvv0p5cv1p5-tree.root']},
            #{ (1    , -9 ,  0.5  ):    ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo2_l-9cvv1cv0p5-tree.root']}

            ##{ ( 2.5 , -4  , 1   ):      ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo3_l-4cvv2p5cv1-tree.root']},
            ##{ (  3  , -6  , 1   ):      ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo3_l-6cvv3cv1-tree.root']},
            ##{ (  1  ,  7  , 1.5 ):      ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo3_l7cvv1cv1p5-tree.root']}


            #{ ( 2   ,  7 ,  1   ): ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo4_l7cvv2cv1-tree.root']},
            #{ ( 0.5 ,  3 ,  0.5 ): ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo4_l3cvv0p5cv0p5-tree.root']},
            #{ (-0.5 ,  8 ,  1   ): ['/home/cmilke/Documents/dihiggs/lhe_histograms/output/neo4_l8cvv-0p5cv1-tree.root']}


        ]
        for index, file_dict in enumerate(file_list):
            print('\n\nINDEX: '+str(index))
            optimize_reco(mode='truth',extra_files=file_dict,extra_name=f'iterC{index:02d}')
            #optimize_reco(mode='truth',extra_files=file_dict,extra_name=f'iterA{index:02d}')
            #optimize_reco(mode='truth',extra_files=file_dict,extra_name=f'itertmp{index:02d}')
    else:
        print('Mode - '+str(args.mode)+' - is not valid.')
        print('Please choose from:\ntruth\nrwgt_truth\nreweight\nreco\n')
        print('Aborting')
        exit(1)


if __name__ == '__main__': main()
