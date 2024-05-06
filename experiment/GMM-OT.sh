#!/bin/bash

SEED="12345 24019 13104"

for seed in ${SEED[@]}
do
    python3 dem/train.py experiment=gmm_iefm_OT seed=${seed}
done