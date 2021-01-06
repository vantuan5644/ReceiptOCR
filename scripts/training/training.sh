#!/usr/bin/env python

DIR=`dirname "$BASH_SOURCE"`

python scripts/training/model_main_tf2.py --model_dir=alignment/models/efficientdet_d0_coco17_tpu-32 --pipeline_config_path=alignment/models/efficientdet_d0_coco17_tpu-32/pipeline.config
