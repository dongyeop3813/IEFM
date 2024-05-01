#!/bin/bash

ENERGY_FUNCTION="GMM DW4"
PROB_PATH="OT VE"
SIGMA_MAX="0.1 1.00 5.00 10.00 50.0"

for energy in ${ENERGY_FUNCTION[@]}
do
    for sigma in ${SIGMA_MAX[@]} 
    do
        python3 dem/test/vf_integrate.py -E ${energy} -M ${sigma} -p OT
    done
done

for energy in ${ENERGY_FUNCTION[@]}
do
    for sigma in ${SIGMA_MAX[@]} 
    do
        python3 dem/test/vf_integrate.py -E ${energy} -M ${sigma} -s 0.00 -p VE
    done
done
