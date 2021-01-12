#!/usr/bin/env python

MODEL_NAME=efficientdet_d1_coco17_tpu-32

PIPELINE_CONFIG_PATH=alignment/$MODEL_NAME/pipeline.config

MODEL_DIR=alignment/$MODEL_NAME

python scripts/training/model_main_tf2.py \
  --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
  --model_dir=${MODEL_DIR} \
  --alsologtostderr \
  --eval_timeout=60