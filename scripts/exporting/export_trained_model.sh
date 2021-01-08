#!/usr/bin/env python

MODEL_NAME=efficientdet_d1_coco17_tpu-32

PIPELINE_CONFIG_PATH=alignment/$MODEL_NAME/pipeline.config

CHECKPOINT_DIR=alignment/$MODEL_NAME

OUTPUT_DIR=alignment/$MODEL_NAME/exported_model

python scripts/exporting/exporter_main_v2.py \
  --input_type image_tensor \
  --pipeline_config_path $PIPELINE_CONFIG_PATH \
  --trained_checkpoint_dir $CHECKPOINT_DIR \
  --output_directory $OUTPUT_DIR
