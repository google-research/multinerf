#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

SCENE=toaster
EXPERIMENT=shinyblender
DATA_DIR=/usr/local/google/home/barron/tmp/nerf_data/dors_nerf_synthetic
CHECKPOINT_DIR=/usr/local/google/home/barron/tmp/nerf_results/"$EXPERIMENT"/"$SCENE"

python -m eval \
  --gin_configs=configs/blender_refnerf.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --logtostderr
