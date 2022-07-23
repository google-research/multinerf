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

"""Evaluation script."""

import functools
from os import path
import sys
import time

from absl import app
from flax.metrics import tensorboard
from flax.training import checkpoints
import gin
from internal import configs
from internal import datasets
from internal import image
from internal import models
from internal import raw_utils
from internal import ref_utils
from internal import train_utils
from internal import utils
from internal import vis
import jax
from jax import random
import jax.numpy as jnp
import numpy as np

configs.define_common_flags()
jax.config.parse_flags_with_absl()


def main(unused_argv):
  config = configs.load_config(save_config=False)

  dataset = datasets.load_dataset('test', config.data_dir, config)

  key = random.PRNGKey(20200823)
  _, state, render_eval_pfn, _, _ = train_utils.setup_model(config, key)

  if config.rawnerf_mode:
    postprocess_fn = dataset.metadata['postprocess_fn']
  else:
    postprocess_fn = lambda z: z

  if config.eval_raw_affine_cc:
    cc_fun = raw_utils.match_images_affine
  else:
    cc_fun = image.color_correct

  metric_harness = image.MetricHarness()

  last_step = 0
  out_dir = path.join(config.checkpoint_dir,
                      'path_renders' if config.render_path else 'test_preds')
  path_fn = lambda x: path.join(out_dir, x)

  if not config.eval_only_once:
    summary_writer = tensorboard.SummaryWriter(
        path.join(config.checkpoint_dir, 'eval'))
  while True:
    state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)
    step = int(state.step)
    if step <= last_step:
      print(f'Checkpoint step {step} <= last step {last_step}, sleeping.')
      time.sleep(10)
      continue
    print(f'Evaluating checkpoint at step {step}.')
    if config.eval_save_output and (not utils.isdir(out_dir)):
      utils.makedirs(out_dir)

    num_eval = min(dataset.size, config.eval_dataset_limit)
    key = random.PRNGKey(0 if config.deterministic_showcase else step)
    perm = random.permutation(key, num_eval)
    showcase_indices = np.sort(perm[:config.num_showcase_images])

    metrics = []
    metrics_cc = []
    showcases = []
    render_times = []
    for idx in range(dataset.size):
      eval_start_time = time.time()
      batch = next(dataset)
      if idx >= num_eval:
        print(f'Skipping image {idx+1}/{dataset.size}')
        continue
      print(f'Evaluating image {idx+1}/{dataset.size}')
      rays = batch.rays
      train_frac = state.step / config.max_steps
      rendering = models.render_image(
          functools.partial(
              render_eval_pfn,
              state.params,
              train_frac,
          ),
          rays,
          None,
          config,
      )

      if jax.host_id() != 0:  # Only record via host 0.
        continue

      render_times.append((time.time() - eval_start_time))
      print(f'Rendered in {render_times[-1]:0.3f}s')

      # Cast to 64-bit to ensure high precision for color correction function.
      gt_rgb = np.array(batch.rgb, dtype=np.float64)
      rendering['rgb'] = np.array(rendering['rgb'], dtype=np.float64)

      cc_start_time = time.time()
      rendering['rgb_cc'] = cc_fun(rendering['rgb'], gt_rgb)
      print(f'Color corrected in {(time.time() - cc_start_time):0.3f}s')

      if not config.eval_only_once and idx in showcase_indices:
        showcase_idx = idx if config.deterministic_showcase else len(showcases)
        showcases.append((showcase_idx, rendering, batch))
      if not config.render_path:
        rgb = postprocess_fn(rendering['rgb'])
        rgb_cc = postprocess_fn(rendering['rgb_cc'])
        rgb_gt = postprocess_fn(gt_rgb)

        if config.eval_quantize_metrics:
          # Ensures that the images written to disk reproduce the metrics.
          rgb = np.round(rgb * 255) / 255
          rgb_cc = np.round(rgb_cc * 255) / 255

        if config.eval_crop_borders > 0:
          crop_fn = lambda x, c=config.eval_crop_borders: x[c:-c, c:-c]
          rgb = crop_fn(rgb)
          rgb_cc = crop_fn(rgb_cc)
          rgb_gt = crop_fn(rgb_gt)

        metric = metric_harness(rgb, rgb_gt)
        metric_cc = metric_harness(rgb_cc, rgb_gt)

        if config.compute_disp_metrics:
          for tag in ['mean', 'median']:
            key = f'distance_{tag}'
            if key in rendering:
              disparity = 1 / (1 + rendering[key])
              metric[f'disparity_{tag}_mse'] = float(
                  ((disparity - batch.disps)**2).mean())

        if config.compute_normal_metrics:
          weights = rendering['acc'] * batch.alphas
          normalized_normals_gt = ref_utils.l2_normalize(batch.normals)
          for key, val in rendering.items():
            if key.startswith('normals') and val is not None:
              normalized_normals = ref_utils.l2_normalize(val)
              metric[key + '_mae'] = ref_utils.compute_weighted_mae(
                  weights, normalized_normals, normalized_normals_gt)

        for m, v in metric.items():
          print(f'{m:30s} = {v:.4f}')

        metrics.append(metric)
        metrics_cc.append(metric_cc)

      if config.eval_save_output and (config.eval_render_interval > 0):
        if (idx % config.eval_render_interval) == 0:
          utils.save_img_u8(postprocess_fn(rendering['rgb']),
                            path_fn(f'color_{idx:03d}.png'))
          utils.save_img_u8(postprocess_fn(rendering['rgb_cc']),
                            path_fn(f'color_cc_{idx:03d}.png'))

          for key in ['distance_mean', 'distance_median']:
            if key in rendering:
              utils.save_img_f32(rendering[key],
                                 path_fn(f'{key}_{idx:03d}.tiff'))

          for key in ['normals']:
            if key in rendering:
              utils.save_img_u8(rendering[key] / 2. + 0.5,
                                path_fn(f'{key}_{idx:03d}.png'))

          utils.save_img_f32(rendering['acc'], path_fn(f'acc_{idx:03d}.tiff'))

    if (not config.eval_only_once) and (jax.host_id() == 0):
      summary_writer.scalar('eval_median_render_time', np.median(render_times),
                            step)
      for name in metrics[0]:
        scores = [m[name] for m in metrics]
        summary_writer.scalar('eval_metrics/' + name, np.mean(scores), step)
        summary_writer.histogram('eval_metrics/' + 'perimage_' + name, scores,
                                 step)
      for name in metrics_cc[0]:
        scores = [m[name] for m in metrics_cc]
        summary_writer.scalar('eval_metrics_cc/' + name, np.mean(scores), step)
        summary_writer.histogram('eval_metrics_cc/' + 'perimage_' + name,
                                 scores, step)

      for i, r, b in showcases:
        if config.vis_decimate > 1:
          d = config.vis_decimate
          decimate_fn = lambda x, d=d: None if x is None else x[::d, ::d]
        else:
          decimate_fn = lambda x: x
        r = jax.tree_util.tree_map(decimate_fn, r)
        b = jax.tree_util.tree_map(decimate_fn, b)
        visualizations = vis.visualize_suite(r, b.rays)
        for k, v in visualizations.items():
          if k == 'color':
            v = postprocess_fn(v)
          summary_writer.image(f'output_{k}_{i}', v, step)
        if not config.render_path:
          target = postprocess_fn(b.rgb)
          summary_writer.image(f'true_color_{i}', target, step)
          pred = postprocess_fn(visualizations['color'])
          residual = np.clip(pred - target + 0.5, 0, 1)
          summary_writer.image(f'true_residual_{i}', residual, step)
          if config.compute_normal_metrics:
            summary_writer.image(f'true_normals_{i}', b.normals / 2. + 0.5,
                                 step)

    if (config.eval_save_output and (not config.render_path) and
        (jax.host_id() == 0)):
      with utils.open_file(path_fn(f'render_times_{step}.txt'), 'w') as f:
        f.write(' '.join([str(r) for r in render_times]))
      for name in metrics[0]:
        with utils.open_file(path_fn(f'metric_{name}_{step}.txt'), 'w') as f:
          f.write(' '.join([str(m[name]) for m in metrics]))
      for name in metrics_cc[0]:
        with utils.open_file(path_fn(f'metric_cc_{name}_{step}.txt'), 'w') as f:
          f.write(' '.join([str(m[name]) for m in metrics_cc]))
      if config.eval_save_ray_data:
        for i, r, b in showcases:
          rays = {k: v for k, v in r.items() if 'ray_' in k}
          np.set_printoptions(threshold=sys.maxsize)
          with utils.open_file(path_fn(f'ray_data_{step}_{i}.txt'), 'w') as f:
            f.write(repr(rays))

    # A hack that forces Jax to keep all TPUs alive until every TPU is finished.
    x = jnp.ones([jax.local_device_count()])
    x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
    print(x)

    if config.eval_only_once:
      break
    if config.early_exit_steps is not None:
      num_steps = config.early_exit_steps
    else:
      num_steps = config.max_steps
    if int(step) >= num_steps:
      break
    last_step = step


if __name__ == '__main__':
  with gin.config_scope('eval'):
    app.run(main)
