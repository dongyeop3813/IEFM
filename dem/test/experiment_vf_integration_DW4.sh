#!/bin/bash

SIGMA_MAX="1.0 2.0 2.5 3.0 4.0 6.0"

for sigma in ${SIGMA_MAX[@]} 
do
    python3 dem/test/vf_integrate.py -E DW4 -M ${sigma} -m 1e-12 -s 0.00 -p VE
done
