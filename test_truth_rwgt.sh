#!/bin/bash

python validate_linear_combinations.py \
    --mode rwgt_truth \
    --basis basis_files/truth_LHE_basis.dat \
    --verify basis_files/truth_rwgt_test.dat
