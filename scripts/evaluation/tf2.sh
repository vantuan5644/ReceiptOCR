#!/usr/bin/env python

MODEL_NAME=efficientdet_d1_coco17_tpu-32

PIPELINE_CONFIG_PATH=models/alignment/$MODEL_NAME/pipeline.config

MODEL_DIR=models/field_detection/$MODEL_NAME

CKPT_DIR=models/field_detection/$MODEL_NAME/

python scripts/training/model_main_tf2.py \
  --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
  --model_dir=${MODEL_DIR} \
  --alsologtostderr \
  --checkpoint_dir=$CKPT_DIR\
  --sample_1_of_n_eval_examples=1
