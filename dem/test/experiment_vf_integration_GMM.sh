#!/bin/bash

OT_SIGMA_MAX="0.1 0.5 1.0 5.0 10.0"
OT_START_TIME="0.001 0.005 0.01"

VE_SIGMA_MAX="50.0 100.0 150.0 200.0 200.0"

for time in ${START_TIME[@]}
do
    for sigma in ${OT_SIGMA_MAX[@]} 
    do
        python3 dem/test/vf_integrate.py -E GMM -M ${sigma} -s ${time} -p OT
    done
done

for sigma in ${VE_SIGMA_MAX[@]} 
do
    python3 dem/test/vf_integrate.py -E GMM -M ${sigma} -s 0 -p VE
done
