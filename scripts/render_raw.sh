#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

SCENE=nightpiano
EXPERIMENT=raw
DATA_DIR=/usr/local/google/home/barron/tmp/nerf_data/rawnerf/scenes
CHECKPOINT_DIR=/usr/local/google/home/barron/tmp/nerf_results/"$EXPERIMENT"/"$SCENE"

python -m render \
  --gin_configs=configs/llff_raw.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}/${SCENE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 10" \
  --gin_bindings="Config.render_dir = '${CHECKPOINT_DIR}/render/'" \
  --gin_bindings="Config.render_video_fps = 2" \
  --logtostderr
