#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

SCENE=flower
EXPERIMENT=llff
DATA_DIR=/usr/local/google/home/barron/tmp/nerf_data/nerf_llff_data
CHECKPOINT_DIR=/usr/local/google/home/barron/tmp/nerf_results/"$EXPERIMENT"/"$SCENE"

rm "$CHECKPOINT_DIR"/*
python -m train \
  --gin_configs=configs/llff_256.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr
