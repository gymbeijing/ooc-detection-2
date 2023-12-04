#!/bin/bash

# custom config
DATA=/import/network-temp/yimengg/data/
# DATA=/path/to/datasets
TRAINER=SimpleTaskRes

DATASET=$1
CFG=$2      # config file
SHOTS=$3    # number of shots (1, 2, 4, 8, 16)
SCALE=$4    # scaling factor alpha

for SEED in 1
do
    DIR=output/${CFG}_${SHOTS}shots/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        CUDA_VISIBLE_DEVICES=3 python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.SimpleTaskRes.RESIDUAL_SCALE ${SCALE}
    fi
done