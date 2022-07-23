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

"""Training script."""

import functools
import gc
import time

from absl import app
import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
import gin
from internal import configs
from internal import datasets
from internal import image
from internal import models
from internal import train_utils
from internal import utils
from internal import vis
import jax
from jax import random
import jax.numpy as jnp
import numpy as np

configs.define_common_flags()
jax.config.parse_flags_with_absl()

TIME_PRECISION = 1000  # Internally represent integer times in milliseconds.


def main(unused_argv):
  rng = random.PRNGKey(20200823)
  # Shift the numpy random seed by host_id() to shuffle data loaded by different
  # hosts.
  np.random.seed(20201473 + jax.host_id())

  config = configs.load_config()

  if config.batch_size % jax.device_count() != 0:
    raise ValueError('Batch size must be divisible by the number of devices.')

  dataset = datasets.load_dataset('train', config.data_dir, config)
  test_dataset = datasets.load_dataset('test', config.data_dir, config)

  np_to_jax = lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x
  cameras = tuple(np_to_jax(x) for x in dataset.cameras)

  if config.rawnerf_mode:
    postprocess_fn = test_dataset.metadata['postprocess_fn']
  else:
    postprocess_fn = lambda z, _=None: z

  rng, key = random.split(rng)
  setup = train_utils.setup_model(config, key, dataset=dataset)
  model, state, render_eval_pfn, train_pstep, lr_fn = setup

  variables = state.params
  num_params = jax.tree_util.tree_reduce(
      lambda x, y: x + jnp.prod(jnp.array(y.shape)), variables, initializer=0)
  print(f'Number of parameters being optimized: {num_params}')

  if (dataset.size > model.num_glo_embeddings and model.num_glo_features > 0):
    raise ValueError(f'Number of glo embeddings {model.num_glo_embeddings} '
                     f'must be at least equal to number of train images '
                     f'{dataset.size}')

  metric_harness = image.MetricHarness()

  if not utils.isdir(config.checkpoint_dir):
    utils.makedirs(config.checkpoint_dir)
  state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)
  # Resume training at the step of the last checkpoint.
  init_step = state.step + 1
  state = flax.jax_utils.replicate(state)

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(config.checkpoint_dir)
    if config.rawnerf_mode:
      for name, data in zip(['train', 'test'], [dataset, test_dataset]):
        # Log shutter speed metadata in TensorBoard for debug purposes.
        for key in ['exposure_idx', 'exposure_values', 'unique_shutters']:
          summary_writer.text(f'{name}_{key}', str(data.metadata[key]), 0)

  # Prefetch_buffer_size = 3 x batch_size.
  pdataset = flax.jax_utils.prefetch_to_device(dataset, 3)
  rng = rng + jax.host_id()  # Make random seed separate across hosts.
  rngs = random.split(rng, jax.local_device_count())  # For pmapping RNG keys.
  gc.disable()  # Disable automatic garbage collection for efficiency.
  total_time = 0
  total_steps = 0
  reset_stats = True
  if config.early_exit_steps is not None:
    num_steps = config.early_exit_steps
  else:
    num_steps = config.max_steps
  for step, batch in zip(range(init_step, num_steps + 1), pdataset):

    if reset_stats and (jax.host_id() == 0):
      stats_buffer = []
      train_start_time = time.time()
      reset_stats = False

    learning_rate = lr_fn(step)
    train_frac = jnp.clip((step - 1) / (config.max_steps - 1), 0, 1)

    state, stats, rngs = train_pstep(
        rngs,
        state,
        batch,
        cameras,
        train_frac,
    )

    if step % config.gc_every == 0:
      gc.collect()  # Disable automatic garbage collection for efficiency.

    # Log training summaries. This is put behind a host_id check because in
    # multi-host evaluation, all hosts need to run inference even though we
    # only use host 0 to record results.
    if jax.host_id() == 0:
      stats = flax.jax_utils.unreplicate(stats)

      stats_buffer.append(stats)

      if step == init_step or step % config.print_every == 0:
        elapsed_time = time.time() - train_start_time
        steps_per_sec = config.print_every / elapsed_time
        rays_per_sec = config.batch_size * steps_per_sec

        # A robust approximation of total training time, in case of pre-emption.
        total_time += int(round(TIME_PRECISION * elapsed_time))
        total_steps += config.print_every
        approx_total_time = int(round(step * total_time / total_steps))

        # Transpose and stack stats_buffer along axis 0.
        fs = [flax.traverse_util.flatten_dict(s, sep='/') for s in stats_buffer]
        stats_stacked = {k: jnp.stack([f[k] for f in fs]) for k in fs[0].keys()}

        # Split every statistic that isn't a vector into a set of statistics.
        stats_split = {}
        for k, v in stats_stacked.items():
          if v.ndim not in [1, 2] and v.shape[0] != len(stats_buffer):
            raise ValueError('statistics must be of size [n], or [n, k].')
          if v.ndim == 1:
            stats_split[k] = v
          elif v.ndim == 2:
            for i, vi in enumerate(tuple(v.T)):
              stats_split[f'{k}/{i}'] = vi

        # Summarize the entire histogram of each statistic.
        for k, v in stats_split.items():
          summary_writer.histogram('train_' + k, v, step)

        # Take the mean and max of each statistic since the last summary.
        avg_stats = {k: jnp.mean(v) for k, v in stats_split.items()}
        max_stats = {k: jnp.max(v) for k, v in stats_split.items()}

        summ_fn = lambda s, v: summary_writer.scalar(s, v, step)  # pylint:disable=cell-var-from-loop

        # Summarize the mean and max of each statistic.
        for k, v in avg_stats.items():
          summ_fn(f'train_avg_{k}', v)
        for k, v in max_stats.items():
          summ_fn(f'train_max_{k}', v)

        summ_fn('train_num_params', num_params)
        summ_fn('train_learning_rate', learning_rate)
        summ_fn('train_steps_per_sec', steps_per_sec)
        summ_fn('train_rays_per_sec', rays_per_sec)

        summary_writer.scalar('train_avg_psnr_timed', avg_stats['psnr'],
                              total_time // TIME_PRECISION)
        summary_writer.scalar('train_avg_psnr_timed_approx', avg_stats['psnr'],
                              approx_total_time // TIME_PRECISION)

        if dataset.metadata is not None and model.learned_exposure_scaling:
          params = state.params['params']
          scalings = params['exposure_scaling_offsets']['embedding'][0]
          num_shutter_speeds = dataset.metadata['unique_shutters'].shape[0]
          for i_s in range(num_shutter_speeds):
            for j_s, value in enumerate(scalings[i_s]):
              summary_name = f'exposure/scaling_{i_s}_{j_s}'
              summary_writer.scalar(summary_name, value, step)

        precision = int(np.ceil(np.log10(config.max_steps))) + 1
        avg_loss = avg_stats['loss']
        avg_psnr = avg_stats['psnr']
        str_losses = {  # Grab each "losses_{x}" field and print it as "x[:4]".
            k[7:11]: (f'{v:0.5f}' if v >= 1e-4 and v < 10 else f'{v:0.1e}')
            for k, v in avg_stats.items()
            if k.startswith('losses/')
        }
        print(f'{step:{precision}d}' + f'/{config.max_steps:d}: ' +
              f'loss={avg_loss:0.5f}, ' + f'psnr={avg_psnr:6.3f}, ' +
              f'lr={learning_rate:0.2e} | ' +
              ', '.join([f'{k}={s}' for k, s in str_losses.items()]) +
              f', {rays_per_sec:0.0f} r/s')

        # Reset everything we are tracking between summarizations.
        reset_stats = True

      if step == 1 or step % config.checkpoint_every == 0:
        state_to_save = jax.device_get(
            flax.jax_utils.unreplicate(state))
        checkpoints.save_checkpoint(
            config.checkpoint_dir, state_to_save, int(step), keep=100)

    # Test-set evaluation.
    if config.train_render_every > 0 and step % config.train_render_every == 0:
      # We reuse the same random number generator from the optimization step
      # here on purpose so that the visualization matches what happened in
      # training.
      eval_start_time = time.time()
      eval_variables = flax.jax_utils.unreplicate(state).params
      test_case = next(test_dataset)
      rendering = models.render_image(
          functools.partial(render_eval_pfn, eval_variables, train_frac),
          test_case.rays, rngs[0], config)

      # Log eval summaries on host 0.
      if jax.host_id() == 0:
        eval_time = time.time() - eval_start_time
        num_rays = jnp.prod(jnp.array(test_case.rays.directions.shape[:-1]))
        rays_per_sec = num_rays / eval_time
        summary_writer.scalar('test_rays_per_sec', rays_per_sec, step)
        print(f'Eval {step}: {eval_time:0.3f}s., {rays_per_sec:0.0f} rays/sec')

        metric_start_time = time.time()
        metric = metric_harness(
            postprocess_fn(rendering['rgb']), postprocess_fn(test_case.rgb))
        print(f'Metrics computed in {(time.time() - metric_start_time):0.3f}s')
        for name, val in metric.items():
          if not np.isnan(val):
            print(f'{name} = {val:.4f}')
            summary_writer.scalar('train_metrics/' + name, val, step)

        if config.vis_decimate > 1:
          d = config.vis_decimate
          decimate_fn = lambda x, d=d: None if x is None else x[::d, ::d]
        else:
          decimate_fn = lambda x: x
        rendering = jax.tree_util.tree_map(decimate_fn, rendering)
        test_case = jax.tree_util.tree_map(decimate_fn, test_case)
        vis_start_time = time.time()
        vis_suite = vis.visualize_suite(rendering, test_case.rays)
        print(f'Visualized in {(time.time() - vis_start_time):0.3f}s')
        if config.rawnerf_mode:
          # Unprocess raw output.
          vis_suite['color_raw'] = rendering['rgb']
          # Autoexposed colors.
          vis_suite['color_auto'] = postprocess_fn(rendering['rgb'], None)
          summary_writer.image('test_true_auto',
                               postprocess_fn(test_case.rgb, None), step)
          # Exposure sweep colors.
          exposures = test_dataset.metadata['exposure_levels']
          for p, x in list(exposures.items()):
            vis_suite[f'color/{p}'] = postprocess_fn(rendering['rgb'], x)
            summary_writer.image(f'test_true_color/{p}',
                                 postprocess_fn(test_case.rgb, x), step)
        summary_writer.image('test_true_color', test_case.rgb, step)
        if config.compute_normal_metrics:
          summary_writer.image('test_true_normals',
                               test_case.normals / 2. + 0.5, step)
        for k, v in vis_suite.items():
          summary_writer.image('test_output_' + k, v, step)

  if jax.host_id() == 0 and config.max_steps % config.checkpoint_every != 0:
    state = jax.device_get(flax.jax_utils.unreplicate(state))
    checkpoints.save_checkpoint(
        config.checkpoint_dir, state, int(config.max_steps), keep=100)


if __name__ == '__main__':
  with gin.config_scope('train'):
    app.run(main)
