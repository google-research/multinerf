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

"""Tools for manipulating step functions (piecewise-constant 1D functions).

We have a shared naming and dimension convention for these functions.
All input/output step functions are assumed to be aligned along the last axis.
`t` always indicates the x coordinates of the *endpoints* of a step function.
`y` indicates unconstrained values for the *bins* of a step function
`w` indicates bin weights that sum to <= 1. `p` indicates non-negative bin
values that *integrate* to <= 1.
"""

from internal import math
import jax
import jax.numpy as jnp


def searchsorted(a, v):
  """Find indices where v should be inserted into a to maintain order.

  This behaves like jnp.searchsorted (its second output is the same as
  jnp.searchsorted's output if all elements of v are in [a[0], a[-1]]) but is
  faster because it wastes memory to save some compute.

  Args:
    a: tensor, the sorted reference points that we are scanning to see where v
      should lie.
    v: tensor, the query points that we are pretending to insert into a. Does
      not need to be sorted. All but the last dimensions should match or expand
      to those of a, the last dimension can differ.

  Returns:
    (idx_lo, idx_hi), where a[idx_lo] <= v < a[idx_hi], unless v is out of the
    range [a[0], a[-1]] in which case idx_lo and idx_hi are both the first or
    last index of a.
  """
  i = jnp.arange(a.shape[-1])
  v_ge_a = v[..., None, :] >= a[..., :, None]
  idx_lo = jnp.max(jnp.where(v_ge_a, i[..., :, None], i[..., :1, None]), -2)
  idx_hi = jnp.min(jnp.where(~v_ge_a, i[..., :, None], i[..., -1:, None]), -2)
  return idx_lo, idx_hi


def query(tq, t, y, outside_value=0):
  """Look up the values of the step function (t, y) at locations tq."""
  idx_lo, idx_hi = searchsorted(t, tq)
  yq = jnp.where(idx_lo == idx_hi, outside_value,
                 jnp.take_along_axis(y, idx_lo, axis=-1))
  return yq


def inner_outer(t0, t1, y1):
  """Construct inner and outer measures on (t1, y1) for t0."""
  cy1 = jnp.concatenate([jnp.zeros_like(y1[..., :1]),
                         jnp.cumsum(y1, axis=-1)],
                        axis=-1)
  idx_lo, idx_hi = searchsorted(t1, t0)

  cy1_lo = jnp.take_along_axis(cy1, idx_lo, axis=-1)
  cy1_hi = jnp.take_along_axis(cy1, idx_hi, axis=-1)

  y0_outer = cy1_hi[..., 1:] - cy1_lo[..., :-1]
  y0_inner = jnp.where(idx_hi[..., :-1] <= idx_lo[..., 1:],
                       cy1_lo[..., 1:] - cy1_hi[..., :-1], 0)
  return y0_inner, y0_outer


def lossfun_outer(t, w, t_env, w_env, eps=jnp.finfo(jnp.float32).eps):
  """The proposal weight should be an upper envelope on the nerf weight."""
  _, w_outer = inner_outer(t, t_env, w_env)
  # We assume w_inner <= w <= w_outer. We don't penalize w_inner because it's
  # more effective to pull w_outer up than it is to push w_inner down.
  # Scaled half-quadratic loss that gives a constant gradient at w_outer = 0.
  return jnp.maximum(0, w - w_outer)**2 / (w + eps)


def weight_to_pdf(t, w, eps=jnp.finfo(jnp.float32).eps**2):
  """Turn a vector of weights that sums to 1 into a PDF that integrates to 1."""
  return w / jnp.maximum(eps, (t[..., 1:] - t[..., :-1]))


def pdf_to_weight(t, p):
  """Turn a PDF that integrates to 1 into a vector of weights that sums to 1."""
  return p * (t[..., 1:] - t[..., :-1])


def max_dilate(t, w, dilation, domain=(-jnp.inf, jnp.inf)):
  """Dilate (via max-pooling) a non-negative step function."""
  t0 = t[..., :-1] - dilation
  t1 = t[..., 1:] + dilation
  t_dilate = jnp.sort(jnp.concatenate([t, t0, t1], axis=-1), axis=-1)
  t_dilate = jnp.clip(t_dilate, *domain)
  w_dilate = jnp.max(
      jnp.where(
          (t0[..., None, :] <= t_dilate[..., None])
          & (t1[..., None, :] > t_dilate[..., None]),
          w[..., None, :],
          0,
      ),
      axis=-1)[..., :-1]
  return t_dilate, w_dilate


def max_dilate_weights(t,
                       w,
                       dilation,
                       domain=(-jnp.inf, jnp.inf),
                       renormalize=False,
                       eps=jnp.finfo(jnp.float32).eps**2):
  """Dilate (via max-pooling) a set of weights."""
  p = weight_to_pdf(t, w)
  t_dilate, p_dilate = max_dilate(t, p, dilation, domain=domain)
  w_dilate = pdf_to_weight(t_dilate, p_dilate)
  if renormalize:
    w_dilate /= jnp.maximum(eps, jnp.sum(w_dilate, axis=-1, keepdims=True))
  return t_dilate, w_dilate


def integrate_weights(w):
  """Compute the cumulative sum of w, assuming all weight vectors sum to 1.

  The output's size on the last dimension is one greater than that of the input,
  because we're computing the integral corresponding to the endpoints of a step
  function, not the integral of the interior/bin values.

  Args:
    w: Tensor, which will be integrated along the last axis. This is assumed to
      sum to 1 along the last axis, and this function will (silently) break if
      that is not the case.

  Returns:
    cw0: Tensor, the integral of w, where cw0[..., 0] = 0 and cw0[..., -1] = 1
  """
  cw = jnp.minimum(1, jnp.cumsum(w[..., :-1], axis=-1))
  shape = cw.shape[:-1] + (1,)
  # Ensure that the CDF starts with exactly 0 and ends with exactly 1.
  cw0 = jnp.concatenate([jnp.zeros(shape), cw, jnp.ones(shape)], axis=-1)
  return cw0


def invert_cdf(u, t, w_logits, use_gpu_resampling=False):
  """Invert the CDF defined by (t, w) at the points specified by u in [0, 1)."""
  # Compute the PDF and CDF for each weight vector.
  w = jax.nn.softmax(w_logits, axis=-1)
  cw = integrate_weights(w)
  # Interpolate into the inverse CDF.
  interp_fn = math.interp if use_gpu_resampling else math.sorted_interp
  t_new = interp_fn(u, cw, t)
  return t_new


def sample(rng,
           t,
           w_logits,
           num_samples,
           single_jitter=False,
           deterministic_center=False,
           use_gpu_resampling=False):
  """Piecewise-Constant PDF sampling from a step function.

  Args:
    rng: random number generator (or None for `linspace` sampling).
    t: [..., num_bins + 1], bin endpoint coordinates (must be sorted)
    w_logits: [..., num_bins], logits corresponding to bin weights
    num_samples: int, the number of samples.
    single_jitter: bool, if True, jitter every sample along each ray by the same
      amount in the inverse CDF. Otherwise, jitter each sample independently.
    deterministic_center: bool, if False, when `rng` is None return samples that
      linspace the entire PDF. If True, skip the front and back of the linspace
      so that the centers of each PDF interval are returned.
    use_gpu_resampling: bool, If True this resamples the rays based on a
      "gather" instruction, which is fast on GPUs but slow on TPUs. If False,
      this resamples the rays based on brute-force searches, which is fast on
      TPUs, but slow on GPUs.

  Returns:
    t_samples: jnp.ndarray(float32), [batch_size, num_samples].
  """
  eps = jnp.finfo(jnp.float32).eps

  # Draw uniform samples.
  if rng is None:
    # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
    if deterministic_center:
      pad = 1 / (2 * num_samples)
      u = jnp.linspace(pad, 1. - pad - eps, num_samples)
    else:
      u = jnp.linspace(0, 1. - eps, num_samples)
    u = jnp.broadcast_to(u, t.shape[:-1] + (num_samples,))
  else:
    # `u` is in [0, 1) --- it can be zero, but it can never be 1.
    u_max = eps + (1 - eps) / num_samples
    max_jitter = (1 - u_max) / (num_samples - 1) - eps
    d = 1 if single_jitter else num_samples
    u = (
        jnp.linspace(0, 1 - u_max, num_samples) +
        jax.random.uniform(rng, t.shape[:-1] + (d,), maxval=max_jitter))

  return invert_cdf(u, t, w_logits, use_gpu_resampling=use_gpu_resampling)


def sample_intervals(rng,
                     t,
                     w_logits,
                     num_samples,
                     single_jitter=False,
                     domain=(-jnp.inf, jnp.inf),
                     use_gpu_resampling=False):
  """Sample *intervals* (rather than points) from a step function.

  Args:
    rng: random number generator (or None for `linspace` sampling).
    t: [..., num_bins + 1], bin endpoint coordinates (must be sorted)
    w_logits: [..., num_bins], logits corresponding to bin weights
    num_samples: int, the number of intervals to sample.
    single_jitter: bool, if True, jitter every sample along each ray by the same
      amount in the inverse CDF. Otherwise, jitter each sample independently.
    domain: (minval, maxval), the range of valid values for `t`.
    use_gpu_resampling:  bool, If True this resamples the rays based on a
      "gather" instruction, which is fast on GPUs but slow on TPUs. If False,
      this resamples the rays based on brute-force searches, which is fast on
      TPUs, but slow on GPUs.

  Returns:
    t_samples: jnp.ndarray(float32), [batch_size, num_samples].
  """
  if num_samples <= 1:
    raise ValueError(f'num_samples must be > 1, is {num_samples}.')

  # Sample a set of points from the step function.
  centers = sample(
      rng,
      t,
      w_logits,
      num_samples,
      single_jitter,
      deterministic_center=True,
      use_gpu_resampling=use_gpu_resampling)

  # The intervals we return will span the midpoints of each adjacent sample.
  mid = (centers[..., 1:] + centers[..., :-1]) / 2

  # Each first/last fencepost is the reflection of the first/last midpoint
  # around the first/last sampled center. We clamp to the limits of the input
  # domain, provided by the caller.
  minval, maxval = domain
  first = jnp.maximum(minval, 2 * centers[..., :1] - mid[..., :1])
  last = jnp.minimum(maxval, 2 * centers[..., -1:] - mid[..., -1:])

  t_samples = jnp.concatenate([first, mid, last], axis=-1)
  return t_samples


def lossfun_distortion(t, w):
  """Compute iint w[i] w[j] |t[i] - t[j]| di dj."""
  # The loss incurred between all pairs of intervals.
  ut = (t[..., 1:] + t[..., :-1]) / 2
  dut = jnp.abs(ut[..., :, None] - ut[..., None, :])
  loss_inter = jnp.sum(w * jnp.sum(w[..., None, :] * dut, axis=-1), axis=-1)

  # The loss incurred within each individual interval with itself.
  loss_intra = jnp.sum(w**2 * (t[..., 1:] - t[..., :-1]), axis=-1) / 3

  return loss_inter + loss_intra


def interval_distortion(t0_lo, t0_hi, t1_lo, t1_hi):
  """Compute mean(abs(x-y); x in [t0_lo, t0_hi], y in [t1_lo, t1_hi])."""
  # Distortion when the intervals do not overlap.
  d_disjoint = jnp.abs((t1_lo + t1_hi) / 2 - (t0_lo + t0_hi) / 2)

  # Distortion when the intervals overlap.
  d_overlap = (2 *
               (jnp.minimum(t0_hi, t1_hi)**3 - jnp.maximum(t0_lo, t1_lo)**3) +
               3 * (t1_hi * t0_hi * jnp.abs(t1_hi - t0_hi) +
                    t1_lo * t0_lo * jnp.abs(t1_lo - t0_lo) + t1_hi * t0_lo *
                    (t0_lo - t1_hi) + t1_lo * t0_hi *
                    (t1_lo - t0_hi))) / (6 * (t0_hi - t0_lo) * (t1_hi - t1_lo))

  # Are the two intervals not overlapping?
  are_disjoint = (t0_lo > t1_hi) | (t1_lo > t0_hi)

  return jnp.where(are_disjoint, d_disjoint, d_overlap)


def weighted_percentile(t, w, ps):
  """Compute the weighted percentiles of a step function. w's must sum to 1."""
  cw = integrate_weights(w)
  # We want to interpolate into the integrated weights according to `ps`.
  fn = lambda cw_i, t_i: jnp.interp(jnp.array(ps) / 100, cw_i, t_i)
  # Vmap fn to an arbitrary number of leading dimensions.
  cw_mat = cw.reshape([-1, cw.shape[-1]])
  t_mat = t.reshape([-1, t.shape[-1]])
  wprctile_mat = (jax.vmap(fn, 0)(cw_mat, t_mat))
  wprctile = wprctile_mat.reshape(cw.shape[:-1] + (len(ps),))
  return wprctile


def resample(t, tp, vp, use_avg=False, eps=jnp.finfo(jnp.float32).eps):
  """Resample a step function defined by (tp, vp) into intervals t.

  Notation roughly matches jnp.interp. Resamples by summation by default.

  Args:
    t: tensor with shape (..., n+1), the endpoints to resample into.
    tp: tensor with shape (..., m+1), the endpoints of the step function being
      resampled.
    vp: tensor with shape (..., m), the values of the step function being
      resampled.
    use_avg: bool, if False, return the sum of the step function for each
      interval in `t`. If True, return the average, weighted by the width of
      each interval in `t`.
    eps: float, a small value to prevent division by zero when use_avg=True.

  Returns:
    v: tensor with shape (..., n), the values of the resampled step function.
  """
  if use_avg:
    wp = jnp.diff(tp, axis=-1)
    v_numer = resample(t, tp, vp * wp, use_avg=False)
    v_denom = resample(t, tp, wp, use_avg=False)
    v = v_numer / jnp.maximum(eps, v_denom)
    return v

  acc = jnp.cumsum(vp, axis=-1)
  acc0 = jnp.concatenate([jnp.zeros(acc.shape[:-1] + (1,)), acc], axis=-1)
  acc0_resampled = jnp.vectorize(
      jnp.interp, signature='(n),(m),(m)->(n)')(t, tp, acc0)
  v = jnp.diff(acc0_resampled, axis=-1)
  return v
