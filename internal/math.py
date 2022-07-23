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

"""Mathy utility functions."""

import jax
import jax.numpy as jnp


def matmul(a, b):
  """jnp.matmul defaults to bfloat16, but this helper function doesn't."""
  return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def safe_trig_helper(x, fn, t=100 * jnp.pi):
  """Helper function used by safe_cos/safe_sin: mods x before sin()/cos()."""
  return fn(jnp.where(jnp.abs(x) < t, x, x % t))


def safe_cos(x):
  """jnp.cos() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, jnp.cos)


def safe_sin(x):
  """jnp.sin() on a TPU may NaN out for large values."""
  return safe_trig_helper(x, jnp.sin)


@jax.custom_jvp
def safe_exp(x):
  """jnp.exp() but with finite output and gradients for large inputs."""
  return jnp.exp(jnp.minimum(x, 88.))  # jnp.exp(89) is infinity.


@safe_exp.defjvp
def safe_exp_jvp(primals, tangents):
  """Override safe_exp()'s gradient so that it's large when inputs are large."""
  x, = primals
  x_dot, = tangents
  exp_x = safe_exp(x)
  exp_x_dot = exp_x * x_dot
  return exp_x, exp_x_dot


def log_lerp(t, v0, v1):
  """Interpolate log-linearly from `v0` (t=0) to `v1` (t=1)."""
  if v0 <= 0 or v1 <= 0:
    raise ValueError(f'Interpolants {v0} and {v1} must be positive.')
  lv0 = jnp.log(v0)
  lv1 = jnp.log(v1)
  return jnp.exp(jnp.clip(t, 0, 1) * (lv1 - lv0) + lv0)


def learning_rate_decay(step,
                        lr_init,
                        lr_final,
                        max_steps,
                        lr_delay_steps=0,
                        lr_delay_mult=1):
  """Continuous learning rate decay function.

  The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
  is log-linearly interpolated elsewhere (equivalent to exponential decay).
  If lr_delay_steps>0 then the learning rate will be scaled by some smooth
  function of lr_delay_mult, such that the initial learning rate is
  lr_init*lr_delay_mult at the beginning of optimization but will be eased back
  to the normal learning rate when steps>lr_delay_steps.

  Args:
    step: int, the current optimization step.
    lr_init: float, the initial learning rate.
    lr_final: float, the final learning rate.
    max_steps: int, the number of steps during optimization.
    lr_delay_steps: int, the number of steps to delay the full learning rate.
    lr_delay_mult: float, the multiplier on the rate when delaying it.

  Returns:
    lr: the learning for current step 'step'.
  """
  if lr_delay_steps > 0:
    # A kind of reverse cosine decay.
    delay_rate = lr_delay_mult + (1 - lr_delay_mult) * jnp.sin(
        0.5 * jnp.pi * jnp.clip(step / lr_delay_steps, 0, 1))
  else:
    delay_rate = 1.
  return delay_rate * log_lerp(step / max_steps, lr_init, lr_final)


def interp(*args):
  """A gather-based (GPU-friendly) vectorized replacement for jnp.interp()."""
  args_flat = [x.reshape([-1, x.shape[-1]]) for x in args]
  ret = jax.vmap(jnp.interp)(*args_flat).reshape(args[0].shape)
  return ret


def sorted_interp(x, xp, fp):
  """A TPU-friendly version of interp(), where xp and fp must be sorted."""

  # Identify the location in `xp` that corresponds to each `x`.
  # The final `True` index in `mask` is the start of the matching interval.
  mask = x[..., None, :] >= xp[..., :, None]

  def find_interval(x):
    # Grab the value where `mask` switches from True to False, and vice versa.
    # This approach takes advantage of the fact that `x` is sorted.
    x0 = jnp.max(jnp.where(mask, x[..., None], x[..., :1, None]), -2)
    x1 = jnp.min(jnp.where(~mask, x[..., None], x[..., -1:, None]), -2)
    return x0, x1

  fp0, fp1 = find_interval(fp)
  xp0, xp1 = find_interval(xp)

  offset = jnp.clip(jnp.nan_to_num((x - xp0) / (xp1 - xp0), 0), 0, 1)
  ret = fp0 + offset * (fp1 - fp0)
  return ret
