#!/usr/bin/env python

PIPELINE_CONFIG_PATH=alignment/efficientdet_d0_coco17_tpu-32/pipeline.config

CHECKPOINT_DIR=alignment/efficientdet_d0_coco17_tpu-32

OUTPUT_DIR=alignment/efficientdet_d0_coco17_tpu-32/weight

python scripts/exporting/exporter_main_v2.py \
 --input_type image_tensor --pipeline_config_path $PIPELINE_CONFIG_PATH \
 --trained_checkpoint_dir $CHECKPOINT_DIR \
 --output_directory $OUTPUT_DIR
