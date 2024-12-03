#!/bin/bash

SEED="12345 24019 13104"

for seed in ${SEED[@]}
do
    python3 dem/train.py experiment=gmm_iefm_VE_unnormalized seed=${seed}
done
