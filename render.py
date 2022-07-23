# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render script."""

import concurrent.futures
import functools
import glob
import os
import time

from absl import app
from flax.training import checkpoints
import gin
from internal import configs
from internal import datasets
from internal import models
from internal import train_utils
from internal import utils
import jax
from jax import random
from matplotlib import cm
import mediapy as media
import numpy as np

configs.define_common_flags()
jax.config.parse_flags_with_absl()


def create_videos(config, base_dir, out_dir, out_name, num_frames):
  """Creates videos out of the images saved to disk."""
  names = [n for n in config.checkpoint_dir.split('/') if n]
  # Last two parts of checkpoint path are experiment name and scene name.
  exp_name, scene_name = names[-2:]
  video_prefix = f'{scene_name}_{exp_name}_{out_name}'

  zpad = max(3, len(str(num_frames - 1)))
  idx_to_str = lambda idx: str(idx).zfill(zpad)

  utils.makedirs(base_dir)

  # Load one example frame to get image shape and depth range.
  depth_file = os.path.join(out_dir, f'distance_mean_{idx_to_str(0)}.tiff')
  depth_frame = utils.load_img(depth_file)
  shape = depth_frame.shape
  p = config.render_dist_percentile
  distance_limits = np.percentile(depth_frame.flatten(), [p, 100 - p])
  lo, hi = [config.render_dist_curve_fn(x) for x in distance_limits]
  print(f'Video shape is {shape[:2]}')

  video_kwargs = {
      'shape': shape[:2],
      'codec': 'h264',
      'fps': config.render_video_fps,
      'crf': config.render_video_crf,
  }

  for k in ['color', 'normals', 'acc', 'distance_mean', 'distance_median']:
    video_file = os.path.join(base_dir, f'{video_prefix}_{k}.mp4')
    input_format = 'gray' if k == 'acc' else 'rgb'
    file_ext = 'png' if k in ['color', 'normals'] else 'tiff'
    idx = 0
    file0 = os.path.join(out_dir, f'{k}_{idx_to_str(0)}.{file_ext}')
    if not utils.file_exists(file0):
      print(f'Images missing for tag {k}')
      continue
    print(f'Making video {video_file}...')
    with media.VideoWriter(
        video_file, **video_kwargs, input_format=input_format) as writer:
      for idx in range(num_frames):
        img_file = os.path.join(out_dir, f'{k}_{idx_to_str(idx)}.{file_ext}')
        if not utils.file_exists(img_file):
          ValueError(f'Image file {img_file} does not exist.')
        img = utils.load_img(img_file)
        if k in ['color', 'normals']:
          img = img / 255.
        elif k.startswith('distance'):
          img = config.render_dist_curve_fn(img)
          img = np.clip((img - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1)
          img = cm.get_cmap('turbo')(img)[..., :3]

        frame = (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)
        writer.add_image(frame)
        idx += 1


def main(unused_argv):

  config = configs.load_config(save_config=False)

  dataset = datasets.load_dataset('test', config.data_dir, config)

  key = random.PRNGKey(20200823)
  _, state, render_eval_pfn, _, _ = train_utils.setup_model(config, key)

  if config.rawnerf_mode:
    postprocess_fn = dataset.metadata['postprocess_fn']
  else:
    postprocess_fn = lambda z: z

  state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)
  step = int(state.step)
  print(f'Rendering checkpoint at step {step}.')

  out_name = 'path_renders' if config.render_path else 'test_preds'
  out_name = f'{out_name}_step_{step}'
  base_dir = config.render_dir
  if base_dir is None:
    base_dir = os.path.join(config.checkpoint_dir, 'render')
  out_dir = os.path.join(base_dir, out_name)
  if not utils.isdir(out_dir):
    utils.makedirs(out_dir)

  path_fn = lambda x: os.path.join(out_dir, x)

  # Ensure sufficient zero-padding of image indices in output filenames.
  zpad = max(3, len(str(dataset.size - 1)))
  idx_to_str = lambda idx: str(idx).zfill(zpad)

  if config.render_save_async:
    async_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    async_futures = []
    def save_fn(fn, *args, **kwargs):
      async_futures.append(async_executor.submit(fn, *args, **kwargs))
  else:
    def save_fn(fn, *args, **kwargs):
      fn(*args, **kwargs)

  for idx in range(dataset.size):
    if idx % config.render_num_jobs != config.render_job_id:
      continue
    # If current image and next image both already exist, skip ahead.
    idx_str = idx_to_str(idx)
    curr_file = path_fn(f'color_{idx_str}.png')
    next_idx_str = idx_to_str(idx + config.render_num_jobs)
    next_file = path_fn(f'color_{next_idx_str}.png')
    if utils.file_exists(curr_file) and utils.file_exists(next_file):
      print(f'Image {idx}/{dataset.size} already exists, skipping')
      continue
    print(f'Evaluating image {idx+1}/{dataset.size}')
    eval_start_time = time.time()
    rays = dataset.generate_ray_batch(idx).rays
    train_frac = 1.
    rendering = models.render_image(
        functools.partial(render_eval_pfn, state.params, train_frac),
        rays, None, config)
    print(f'Rendered in {(time.time() - eval_start_time):0.3f}s')

    if jax.host_id() != 0:  # Only record via host 0.
      continue

    rendering['rgb'] = postprocess_fn(rendering['rgb'])

    save_fn(
        utils.save_img_u8, rendering['rgb'], path_fn(f'color_{idx_str}.png'))
    if 'normals' in rendering:
      save_fn(
          utils.save_img_u8, rendering['normals'] / 2. + 0.5,
          path_fn(f'normals_{idx_str}.png'))
    save_fn(
        utils.save_img_f32, rendering['distance_mean'],
        path_fn(f'distance_mean_{idx_str}.tiff'))
    save_fn(
        utils.save_img_f32, rendering['distance_median'],
        path_fn(f'distance_median_{idx_str}.tiff'))
    save_fn(
        utils.save_img_f32, rendering['acc'], path_fn(f'acc_{idx_str}.tiff'))

  if config.render_save_async:
    # Wait until all worker threads finish.
    async_executor.shutdown(wait=True)

    # This will ensure that exceptions in child threads are raised to the
    # main thread.
    for future in async_futures:
      future.result()

  time.sleep(1)
  num_files = len(glob.glob(path_fn('acc_*.tiff')))
  time.sleep(10)
  if jax.host_id() == 0 and num_files == dataset.size:
    print(f'All files found, creating videos (job {config.render_job_id}).')
    create_videos(config, base_dir, out_dir, out_name, dataset.size)

  # A hack that forces Jax to keep all TPUs alive until every TPU is finished.
  x = jax.numpy.ones([jax.local_device_count()])
  x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
  print(x)


if __name__ == '__main__':
  with gin.config_scope('eval'):  # Use the same scope as eval.py
    app.run(main)
