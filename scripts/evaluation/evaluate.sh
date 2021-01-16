MODEL_NAME=efficientdet_d1_coco17_tpu-32
CKPT_DIR=alignment/$MODEL_NAME
EVAL_DIR=datasets/COOP/evaluation
PIPELINE_CONFIG_PATH=alignment/$MODEL_NAME/pipeline.config

python3 scripts/evaluation/eval.py \
  --logtostderr \
  --checkpoint_dir=$CKPT_DIR \
  --eval_dir=$EVAL_DIR \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \

