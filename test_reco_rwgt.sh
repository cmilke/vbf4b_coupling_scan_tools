#!/bin/bash

python validate_linear_combinations.py \
     --mode reweight \
    --basis basis_files/truth_LHE_basis.dat \
    --verify basis_files/rwgt_test_truth.dat
