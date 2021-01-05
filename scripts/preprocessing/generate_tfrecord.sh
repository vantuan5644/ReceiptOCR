#!/bin/bash

DIR=`dirname "$BASH_SOURCE"`

DATASET_DIR="./datasets/COOP"

python scripts/preprocessing/generate_tfrecord.py -x $DATASET_DIR/annotations -l $DATASET_DIR/label_map.pbtxt -o $DATASET_DIR/train.record -i $DATASET_DIR/padded -c $DATASET_DIR/report.csv
