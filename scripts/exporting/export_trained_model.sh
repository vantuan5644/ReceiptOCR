#!/usr/bin/env python

MODEL_NAME=efficientdet_d1_coco17_tpu-32

PIPELINE_CONFIG_PATH=models/field_detection/$MODEL_NAME/pipeline.config

CHECKPOINT_DIR=models/field_detection/$MODEL_NAME

OUTPUT_DIR=pretrained_models/field_detection/$MODEL_NAME/

python scripts/exporting/exporter_main_v2.py \
  --input_type image_tensor \
  --pipeline_config_path $PIPELINE_CONFIG_PATH \
  --trained_checkpoint_dir $CHECKPOINT_DIR \
  --output_directory $OUTPUT_DIR
