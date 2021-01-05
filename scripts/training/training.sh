#!/bin/bash

export PYTHONPATH=$PYTHONPATH:../TensorFlow/models

DIR=`dirname "$BASH_SOURCE"`

python $DIR/training.py --model_dir=alignment/models/efficientdet_d0_coco17_tpu-32 --pipeline_config_path=alignment/models/efficientdet_d0_coco17_tpu-32/pipeline.config
