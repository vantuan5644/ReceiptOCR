#!/bin/bash

DATASET_DIR="datasets/COOP"

python3 scripts/preprocessing/partition_dataset.py \
  -i $DATASET_DIR/easy \
  -o $DATASET_DIR \
  -r 0.1 \
  -x
