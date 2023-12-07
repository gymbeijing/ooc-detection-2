#!/bin/bash


BATCH_SIZE=$1
MAX_EPOCHS=$2      # config file
FEW_SHOT_TOPIC=$3    # number of shots (1, 2, 4, 8, 16)
BASE_MODEL=$4    # scaling factor alpha
ALPHA=$5

do
  python -m trainers.train_simpleTaskRes \
  --batch_size ${BATCH_SIZE} \
  --max_epochs ${MAX_EPOCHS} \
  --few_shot_topic ${FEW_SHOT_TOPIC} \
  --base_model ${FEW_SHOT_TOPIC} \
  --alpha ${ALPHA}
done