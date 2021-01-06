#!/usr/bin/env python

PIPELINE_CONFIG_PATH=alignment/models/efficientdet_d0_coco17_tpu-32/pipeline.config

MODEL_DIR=alignment/models/efficientdet_d0_coco17_tpu-32

python scripts/training/model_main_tf2.py \
  --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
  --model_dir=${MODEL_DIR} \
  --alsologtostderr
