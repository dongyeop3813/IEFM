#!/bin/bash

TIMES="0.0 0.2 0.4 0.6 0.8 1.0"
SIGMA_MAX="1.00 5.00 10.00 30.00"

for sigma in ${SIGMA_MAX[@]} 
do
    for t in ${TIMES[@]}
    do
        python3 dem/test/plot_vf.py --time ${t} --sigma_min 0.01 --sigma_max ${sigma}
    done
done