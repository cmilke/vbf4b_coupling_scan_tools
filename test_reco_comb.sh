#!/bin/bash

python validate_linear_combinations.py \
     --mode reco \
    --basis basis_files/nnt_basis.dat \
    --verify basis_files/nnt_test.dat
