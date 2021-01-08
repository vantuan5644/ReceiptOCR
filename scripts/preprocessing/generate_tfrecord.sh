#!/bin/bash

DIR=$(dirname "$BASH_SOURCE")

DATASET_DIR="datasets/COOP"

python scripts/preprocessing/generate_tfrecord.py \
  --xml_dir $DATASET_DIR/train \
  --labels_path $DATASET_DIR/label_map.pbtxt \
  --output_path $DATASET_DIR/train.record \
  --csv_path $DATASET_DIR/train_report.csv

python scripts/preprocessing/generate_tfrecord.py \
  --xml_dir $DATASET_DIR/train \
  --labels_path $DATASET_DIR/label_map.pbtxt \
  --output_path $DATASET_DIR/train.record \
  --csv_path $DATASET_DIR/test_report.csv
