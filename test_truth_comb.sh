#!/bin/bash

python validate_linear_combinations.py \
    --mode truth \
    --basis basis_files/truth_LHE_basis.dat \
    --verify basis_files/truth_LHE_test.dat
