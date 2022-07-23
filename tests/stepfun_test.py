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

"""Unit tests for stepfun."""

from absl.testing import absltest
from absl.testing import parameterized
from internal import stepfun
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import scipy as sp


def inner(t0, t1, w1):
  """A reference implementation for computing the inner measure of (t1, w1)."""
  w0_inner = []
  for i in range(len(t0) - 1):
    w_sum = 0
    for j in range(len(t1) - 1):
      if (t1[j] >= t0[i]) and (t1[j + 1] < t0[i + 1]):
        w_sum += w1[j]
    w0_inner.append(w_sum)
  w0_inner = jnp.array(w0_inner)
  return w0_inner


def outer(t0, t1, w1):
  """A reference implementation for computing the outer measure of (t1, w1)."""
  w0_outer = []
  for i in range(len(t0) - 1):
    w_sum = 0
    for j in range(len(t1) - 1):
      if (t1[j + 1] >= t0[i]) and (t1[j] <= t0[i + 1]):
        w_sum += w1[j]
    w0_outer.append(w_sum)
  w0_outer = jnp.array(w0_outer)
  return w0_outer


class StepFunTest(parameterized.TestCase):

  def test_searchsorted_in_bounds(self):
    """Test that a[i] <= v < a[j], with (i, j) = searchsorted(a, v)."""
    rng = random.PRNGKey(0)
    eps = 1e-7
    for _ in range(10):
      # Sample vector lengths.
      key, rng = random.split(rng)
      n = random.randint(key, (), 10, 100)
      key, rng = random.split(rng)
      m = random.randint(key, (), 10, 100)

      # Generate query points in [eps, 1-eps].
      key, rng = random.split(rng)
      v = random.uniform(key, [n], minval=eps, maxval=1 - eps)

      # Generate sorted reference points that span all of [0, 1].
      key, rng = random.split(rng)
      a = jnp.sort(random.uniform(key, [m]))
      a = jnp.concatenate([jnp.array([0.]), a, jnp.array([1.])])
      idx_lo, idx_hi = stepfun.searchsorted(a, v)

      self.assertTrue(jnp.all(a[idx_lo] <= v))
      self.assertTrue(jnp.all(v < a[idx_hi]))

  def test_searchsorted_out_of_bounds(self):
    """searchsorted should produce the first/last indices when out of bounds."""
    rng = random.PRNGKey(0)
    for _ in range(10):
      # Sample vector lengths.
      key, rng = random.split(rng)
      n = random.randint(key, (), 10, 100)
      key, rng = random.split(rng)
      m = random.randint(key, (), 10, 100)

      # Generate sorted reference points that span [1, 2].
      key, rng = random.split(rng)
      a = jnp.sort(random.uniform(key, [m], minval=1, maxval=2))

      # Generated queries below and above the reference points.
      key, rng = random.split(rng)
      v_lo = random.uniform(key, [n], minval=0., maxval=0.9)

      key, rng = random.split(rng)
      v_hi = random.uniform(key, [n], minval=2.1, maxval=3)

      idx_lo, idx_hi = stepfun.searchsorted(a, v_lo)
      np.testing.assert_array_equal(idx_lo, jnp.zeros_like(idx_lo))
      np.testing.assert_array_equal(idx_hi, jnp.zeros_like(idx_hi))

      idx_lo, idx_hi = stepfun.searchsorted(a, v_hi)
      np.testing.assert_array_equal(idx_lo, jnp.full_like(idx_lo, m - 1))
      np.testing.assert_array_equal(idx_hi, jnp.full_like(idx_hi, m - 1))

  def test_searchsorted_reference(self):
    """Test against jnp.searchsorted, which behaves similarly to ours."""
    rng = random.PRNGKey(0)
    eps = 1e-7
    n = 30
    m = 40

    # Generate query points in [eps, 1-eps].
    key, rng = random.split(rng)
    v = random.uniform(key, [n], minval=eps, maxval=1 - eps)

    # Generate sorted reference points that span all of [0, 1].
    key, rng = random.split(rng)
    a = jnp.sort(random.uniform(key, [m]))
    a = jnp.concatenate([jnp.array([0.]), a, jnp.array([1.])])
    _, idx_hi = stepfun.searchsorted(a, v)
    np.testing.assert_array_equal(jnp.searchsorted(a, v), idx_hi)

  def test_searchsorted(self):
    """An alternative correctness test for in-range queries to searchsorted."""
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    a = jnp.sort(random.uniform(key, [10], minval=-4, maxval=4))

    key, rng = random.split(rng)
    v = random.uniform(key, [100], minval=-6, maxval=6)

    idx_lo, idx_hi = stepfun.searchsorted(a, v)

    for x, i0, i1 in zip(v, idx_lo, idx_hi):
      if x < jnp.min(a):
        i0_true, i1_true = [0] * 2
      elif x > jnp.max(a):
        i0_true, i1_true = [len(a) - 1] * 2
      else:
        i0_true = jnp.argmax(jnp.where(x >= a, a, -jnp.inf))
        i1_true = jnp.argmin(jnp.where(x < a, a, jnp.inf))
      np.testing.assert_array_equal(i0_true, i0)
      np.testing.assert_array_equal(i1_true, i1)

  @parameterized.named_parameters(
      ('front_delta_0', 'front', 0.),  # Include the front of each span.
      ('front_delta_0.05', 'front', 0.05),
      ('front_delta_0.099', 'front', 0.099),
      ('back_delta_1e-6', 'back', 1e-6),  # Exclude the back of each span.
      ('back_delta_0.05', 'back', 0.05),
      ('back_delta_0.099', 'back', 0.099),
      ('before', 'before', 1e-6),
      ('after', 'after', 0.),
  )
  def test_query(self, mode, delta):
    """Test that query() behaves sensibly in easy cases."""
    n, d = 10, 8
    outside_value = -10.
    max_delta = 0.1

    key0, key1 = random.split(random.PRNGKey(0))
    # Each t value is at least max_delta more than the one before.
    t = -d / 2 + jnp.cumsum(
        random.uniform(key0, minval=max_delta, shape=(n, d + 1)), axis=-1)
    y = random.normal(key1, shape=(n, d))

    query = lambda tq: stepfun.query(tq, t, y, outside_value=outside_value)

    if mode == 'front':
      # Query the a point relative to the front of each span, shifted by delta
      # (if delta < max_delta this will not take you out of the current span).
      assert delta >= 0
      assert delta < max_delta
      yq = query(t[..., :-1] + delta)
      np.testing.assert_array_equal(yq, y)
    elif mode == 'back':
      # Query the a point relative to the back of each span, shifted by delta
      # (if delta < max_delta this will not take you out of the current span).
      assert delta >= 0
      assert delta < max_delta
      yq = query(t[..., 1:] - delta)
      np.testing.assert_array_equal(yq, y)
    elif mode == 'before':
      # Query values before the domain of the step function (exclusive).
      min_val = jnp.min(t, axis=-1)
      assert delta >= 0
      tq = min_val[:, None] + jnp.linspace(-10, -delta, 100)[None, :]
      yq = query(tq)
      np.testing.assert_array_equal(yq, outside_value)
    elif mode == 'after':
      # Queries values after the domain of the step function (inclusive).
      max_val = jnp.max(t, axis=-1)
      assert delta >= 0
      tq = max_val[:, None] + jnp.linspace(delta, 10, 100)[None, :]
      yq = query(tq)
      np.testing.assert_array_equal(yq, outside_value)

  def test_distortion_loss_against_sampling(self):
    """Test that the distortion loss matches a stochastic approximation."""
    # Construct a random step function that defines a weight distribution.
    n, d = 10, 8
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    t = random.uniform(key, minval=-3, maxval=3, shape=(n, d + 1))
    t = jnp.sort(t, axis=-1)
    key, rng = random.split(rng)
    logits = 2 * random.normal(key, shape=(n, d))

    # Compute the distortion loss.
    w = jax.nn.softmax(logits, axis=-1)
    losses = stepfun.lossfun_distortion(t, w)

    # Approximate the distortion loss using samples from the step function.
    key, rng = random.split(rng)
    samples = stepfun.sample(key, t, logits, 10000, single_jitter=False)
    losses_stoch = []
    for i in range(n):
      losses_stoch.append(
          jnp.mean(jnp.abs(samples[i][:, None] - samples[i][None, :])))
    losses_stoch = jnp.array(losses_stoch)

    np.testing.assert_allclose(losses, losses_stoch, atol=1e-4, rtol=1e-4)

  def test_interval_distortion_against_brute_force(self):
    n, d = 3, 7
    rng = random.PRNGKey(0)

    key, rng = random.split(rng)
    t0 = random.uniform(key, minval=-3, maxval=3, shape=(n, d + 1))
    t0 = jnp.sort(t0, axis=-1)

    key, rng = random.split(rng)
    t1 = random.uniform(key, minval=-3, maxval=3, shape=(n, d + 1))
    t1 = jnp.sort(t1, axis=-1)

    distortions = stepfun.interval_distortion(t0[..., :-1], t0[..., 1:],
                                              t1[..., :-1], t1[..., 1:])

    distortions_brute = np.array(jnp.zeros_like(distortions))
    for i in range(n):
      for j in range(d):
        distortions_brute[i, j] = jnp.mean(
            jnp.abs(
                jnp.linspace(t0[i, j], t0[i, j + 1], 5001)[:, None] -
                jnp.linspace(t1[i, j], t1[i, j + 1], 5001)[None, :]))
    np.testing.assert_allclose(
        distortions, distortions_brute, atol=1e-6, rtol=1e-3)

  def test_distortion_loss_against_interval_distortion(self):
    """Test that the distortion loss matches a brute-force alternative."""
    # Construct a random step function that defines a weight distribution.
    n, d = 3, 8
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    t = random.uniform(key, minval=-3, maxval=3, shape=(n, d + 1))
    t = jnp.sort(t, axis=-1)
    key, rng = random.split(rng)
    logits = 2 * random.normal(key, shape=(n, d))

    # Compute the distortion loss.
    w = jax.nn.softmax(logits, axis=-1)
    losses = stepfun.lossfun_distortion(t, w)

    # Compute it again in a more brute-force way, but computing the weighted
    # distortion of all pairs of intervals.
    d = stepfun.interval_distortion(t[..., :-1, None], t[..., 1:, None],
                                    t[..., None, :-1], t[..., None, 1:])
    losses_alt = jnp.sum(w[:, None, :] * w[:, :, None] * d, axis=[-1, -2])

    np.testing.assert_allclose(losses, losses_alt, atol=1e-6, rtol=1e-4)

  def test_max_dilate(self):
    """Compare max_dilate to a brute force test on queries of step functions."""
    n, d, dilation = 20, 8, 0.53

    # Construct a non-negative step function.
    key0, key1 = random.split(random.PRNGKey(0))
    t = jnp.cumsum(
        random.randint(key0, minval=1, maxval=10, shape=(n, d + 1)),
        axis=-1) / 10
    w = jax.nn.softmax(random.normal(key1, shape=(n, d)), axis=-1)

    # Dilate it.
    td, wd = stepfun.max_dilate(t, w, dilation)

    # Construct queries at the midpoint of each interval.
    tq = (jnp.arange((d + 4) * 10) - 2.5) / 10

    # Query the step function and its dilation.
    wq = stepfun.query(tq[None], t, w)
    wdq = stepfun.query(tq[None], td, wd)

    # The queries of the dilation must be the max of the non-dilated queries.
    mask = jnp.abs(tq[None, :] - tq[:, None]) <= dilation
    for i in range(n):
      wdq_i = jnp.max(mask * wq[i], axis=-1)
      np.testing.assert_array_equal(wdq[i], wdq_i)

  @parameterized.named_parameters(('deterministic', False, None),
                                  ('random_multiple_jitters', True, False),
                                  ('random_single_jitter', True, True))
  def test_sample_train_mode(self, randomized, single_jitter):
    """Test that piecewise-constant sampling reproduces its distribution."""
    rng = random.PRNGKey(0)
    batch_size = 4
    num_bins = 16
    num_samples = 1000000
    precision = 1e5

    # Generate a series of random PDFs to sample from.
    data = []
    for _ in range(batch_size):
      rng, key = random.split(rng)
      # Randomly initialize the distances between bins.
      # We're rolling our own fixed precision here to make cumsum exact.
      bins_delta = jnp.round(precision * jnp.exp(
          random.uniform(key, shape=(num_bins + 1,), minval=-3, maxval=3)))

      # Set some of the bin distances to 0.
      rng, key = random.split(rng)
      bins_delta *= random.uniform(key, shape=bins_delta.shape) < 0.9

      # Integrate the bins.
      bins = jnp.cumsum(bins_delta) / precision
      rng, key = random.split(rng)
      bins += random.normal(key) * num_bins / 2
      rng, key = random.split(rng)

      # Randomly generate weights, allowing some to be zero.
      weights = jnp.maximum(
          0, random.uniform(key, shape=(num_bins,), minval=-0.5, maxval=1.))
      gt_hist = weights / weights.sum()
      data.append((bins, weights, gt_hist))

    bins, weights, gt_hist = [jnp.stack(x) for x in zip(*data)]

    rng = random.PRNGKey(0) if randomized else None
    # Draw samples from the batch of PDFs.
    samples = stepfun.sample(
        key,
        bins,
        jnp.log(weights) + 0.7,
        num_samples,
        single_jitter=single_jitter,
    )
    self.assertEqual(samples.shape[-1], num_samples)

    # Check that samples are sorted.
    self.assertTrue(jnp.all(samples[..., 1:] >= samples[..., :-1]))

    # Verify that each set of samples resembles the target distribution.
    for i_samples, i_bins, i_gt_hist in zip(samples, bins, gt_hist):
      i_hist = jnp.float32(jnp.histogram(i_samples, i_bins)[0]) / num_samples
      i_gt_hist = jnp.array(i_gt_hist)

      # Merge any of the zero-span bins until there aren't any left.
      while jnp.any(i_bins[:-1] == i_bins[1:]):
        j = int(jnp.where(i_bins[:-1] == i_bins[1:])[0][0])
        i_hist = jnp.concatenate([
            i_hist[:j],
            jnp.array([i_hist[j] + i_hist[j + 1]]), i_hist[j + 2:]
        ])
        i_gt_hist = jnp.concatenate([
            i_gt_hist[:j],
            jnp.array([i_gt_hist[j] + i_gt_hist[j + 1]]), i_gt_hist[j + 2:]
        ])
        i_bins = jnp.concatenate([i_bins[:j], i_bins[j + 1:]])

      # Angle between the two histograms in degrees.
      angle = 180 / jnp.pi * jnp.arccos(
          jnp.minimum(
              1.,
              jnp.mean((i_hist * i_gt_hist) /
                       jnp.sqrt(jnp.mean(i_hist**2) * jnp.mean(i_gt_hist**2)))))
      # Jensen-Shannon divergence.
      m = (i_hist + i_gt_hist) / 2
      js_div = jnp.sum(
          sp.special.kl_div(i_hist, m) + sp.special.kl_div(i_gt_hist, m)) / 2
      self.assertLessEqual(angle, 0.5)
      self.assertLessEqual(js_div, 1e-5)

  @parameterized.named_parameters(('deterministic', False, None),
                                  ('random_multiple_jitters', True, False),
                                  ('random_single_jitter', True, True))
  def test_sample_large_flat(self, randomized, single_jitter):
    """Test sampling when given a large flat distribution."""
    key = random.PRNGKey(0) if randomized else None
    num_samples = 100
    num_bins = 100000
    bins = jnp.arange(num_bins)
    weights = np.ones(len(bins) - 1)
    samples = stepfun.sample(
        key,
        bins[None],
        jnp.log(jnp.maximum(1e-15, weights[None])),
        num_samples,
        single_jitter=single_jitter,
    )[0]
    # All samples should be within the range of the bins.
    self.assertTrue(jnp.all(samples >= bins[0]))
    self.assertTrue(jnp.all(samples <= bins[-1]))

    # Samples modded by their bin index should resemble a uniform distribution.
    samples_mod = jnp.mod(samples, 1)
    self.assertLessEqual(
        sp.stats.kstest(samples_mod, 'uniform', (0, 1)).statistic, 0.2)

    # All samples should collectively resemble a uniform distribution.
    self.assertLessEqual(
        sp.stats.kstest(samples, 'uniform', (bins[0], bins[-1])).statistic, 0.2)

  def test_gpu_vs_tpu_resampling(self):
    """Test that  gather-based resampling matches the search-based resampler."""
    key = random.PRNGKey(0)
    num_samples = 100
    num_bins = 100000
    bins = jnp.arange(num_bins)
    weights = np.ones(len(bins) - 1)
    samples_search_tpu = stepfun.sample(
        key,
        bins[None],
        jnp.log(jnp.maximum(1e-15, weights[None])),
        num_samples,
        single_jitter=False,
        use_gpu_resampling=False,
    )[0]
    samples_search_gpu = stepfun.sample(
        key,
        bins[None],
        jnp.log(jnp.maximum(1e-15, weights[None])),
        num_samples,
        single_jitter=False,
        use_gpu_resampling=True,
    )[0]
    np.testing.assert_allclose(
        samples_search_tpu, samples_search_gpu, atol=1E-5, rtol=1E-5)

  @parameterized.named_parameters(('deterministic', False, None),
                                  ('random_multiple_jitters', True, False),
                                  ('random_single_jitter', True, True))
  def test_sample_sparse_delta(self, randomized, single_jitter):
    """Test sampling when given a large distribution with a big delta in it."""
    key = random.PRNGKey(0) if randomized else None
    num_samples = 100
    num_bins = 100000
    bins = jnp.arange(num_bins)
    weights = np.ones(len(bins) - 1)
    delta_idx = len(weights) // 2
    weights[delta_idx] = len(weights) - 1
    samples = stepfun.sample(
        key,
        bins[None],
        jnp.log(jnp.maximum(1e-15, weights[None])),
        num_samples,
        single_jitter=single_jitter,
    )[0]

    # All samples should be within the range of the bins.
    self.assertTrue(jnp.all(samples >= bins[0]))
    self.assertTrue(jnp.all(samples <= bins[-1]))

    # Samples modded by their bin index should resemble a uniform distribution.
    samples_mod = jnp.mod(samples, 1)
    self.assertLessEqual(
        sp.stats.kstest(samples_mod, 'uniform', (0, 1)).statistic, 0.2)

    # The delta function bin should contain ~half of the samples.
    in_delta = (samples >= bins[delta_idx]) & (samples <= bins[delta_idx + 1])
    np.testing.assert_allclose(jnp.mean(in_delta), 0.5, atol=0.05)

  @parameterized.named_parameters(('deterministic', False, None),
                                  ('random_multiple_jitters', True, False),
                                  ('random_single_jitter', True, True))
  def test_sample_single_bin(self, randomized, single_jitter):
    """Test sampling when given a small `one hot' distribution."""
    key = random.PRNGKey(0) if randomized else None
    num_samples = 625
    bins = jnp.array([0, 1, 3, 6, 10], jnp.float32)
    for i in range(len(bins) - 1):
      weights = np.zeros(len(bins) - 1, jnp.float32)
      weights[i] = 1.
      samples = stepfun.sample(
          key,
          bins[None],
          jnp.log(weights[None]),
          num_samples,
          single_jitter=single_jitter,
      )[0]

      # All samples should be within [bins[i], bins[i+1]].
      self.assertTrue(jnp.all(samples >= bins[i]))
      self.assertTrue(jnp.all(samples <= bins[i + 1]))

  @parameterized.named_parameters(('deterministic', False, 0.1),
                                  ('random', True, 0.1))
  def test_sample_intervals_accuracy(self, randomized, tolerance):
    """Test that resampled intervals resemble their original distribution."""
    n, d = 50, 32
    d_resample = 2 * d
    domain = -3, 3

    # Generate some step functions.
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    t = random.uniform(
        key, minval=domain[0], maxval=domain[1], shape=(n, d + 1))
    t = jnp.sort(t, axis=-1)
    key, rng = random.split(rng)
    logits = 2 * random.normal(key, shape=(n, d))

    # Resample the step functions.
    key = random.PRNGKey(999) if randomized else None
    t_sampled = stepfun.sample_intervals(
        key, t, logits, d_resample, single_jitter=True, domain=domain)

    # Precompute the accumulated weights of the original intervals.
    weights = jax.nn.softmax(logits, axis=-1)
    acc_weights = stepfun.integrate_weights(weights)

    errors = []
    for i in range(t_sampled.shape[0]):
      # Resample into the original accumulated weights.
      acc_resampled = jnp.interp(t_sampled[i], t[i], acc_weights[i])
      # Differentiate the accumulation to get resampled weights (that do not
      # necessarily sum to 1 because some of the ends might get missed).
      weights_resampled = jnp.diff(acc_resampled, axis=-1)
      # Check that the resampled weights resemble a uniform distribution.
      u = 1 / len(weights_resampled)
      errors.append(float(jnp.sum(jnp.abs(weights_resampled - u))))
    errors = jnp.array(errors)
    mean_error = jnp.mean(errors)
    print(f'Mean Error = {mean_error}, Tolerance = {tolerance}')
    self.assertLess(mean_error, tolerance)

  @parameterized.named_parameters(('deterministic_unbounded', False, False),
                                  ('random_unbounded', True, False),
                                  ('deterministic_bounded', False, True),
                                  ('random_bounded', True, True))
  def test_sample_intervals_unbiased(self, randomized, bound_domain):
    """Test that resampled intervals are unbiased."""
    n, d_resample = 1000, 64
    domain = (-0.5, 0.5) if bound_domain else (-jnp.inf, jnp.inf)

    # A single interval from [-0.5, 0.5].
    t = jnp.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    logits = jnp.array([0, 0, 100., 0, 0])

    ts = t[None].tile([n, 1])
    logits = logits[None].tile([n, 1])

    # Resample the step functions.
    rng = random.PRNGKey(0) if randomized else None
    t_sampled = stepfun.sample_intervals(
        rng, ts, logits, d_resample, single_jitter=True, domain=domain)

    # The average sample should be close to zero.
    if randomized:
      self.assertLess(
          jnp.max(jnp.abs(jnp.mean(t_sampled, axis=-1))), 0.5 / d_resample)
    else:
      np.testing.assert_allclose(
          jnp.mean(t_sampled, axis=-1), jnp.zeros(n), atol=1E-5, rtol=1E-5)

    # The extents of the samples should be near -0.5 and 0.5.
    if bound_domain and randomized:
      np.testing.assert_allclose(jnp.median(t_sampled[:, 0]), -0.5, atol=1e-4)
      np.testing.assert_allclose(jnp.median(t_sampled[:, -1]), 0.5, atol=1e-4)

    # The interval edge near the extent should be centered around +/-0.5.
    if randomized:
      np.testing.assert_allclose(
          jnp.mean(t_sampled[:, 0] > -0.5), 0.5, atol=1 / d_resample)
      np.testing.assert_allclose(
          jnp.mean(t_sampled[:, -1] < 0.5), 0.5, atol=1 / d_resample)

  def test_sample_single_interval(self):
    """Resample a single interval and check that it's a linspace."""
    t = jnp.array([1, 2, 3, 4, 5, 6])
    logits = jnp.array([0, 0, 100, 0, 0])
    key = None
    t_sampled = stepfun.sample_intervals(key, t, logits, 10, single_jitter=True)
    np.testing.assert_allclose(
        t_sampled, jnp.linspace(3, 4, 11), atol=1E-5, rtol=1E-5)

  @parameterized.named_parameters(('sameset', 0, True), ('diffset', 2, False))
  def test_lossfun_outer(self, num_ablate, is_all_zero):
    """Two histograms of the same/diff points have a loss of zero/non-zero."""
    rng = random.PRNGKey(0)
    eps = 1e-12  # Need a little slack because of cumsum's numerical weirdness.
    all_zero = True
    for _ in range(10):
      key, rng = random.split(rng)
      num_pts, d0, d1 = random.randint(key, [3], minval=10, maxval=20)

      key, rng = random.split(rng)
      t0 = jnp.sort(random.uniform(key, [d0 + 1]), axis=-1)

      key, rng = random.split(rng)
      t1 = jnp.sort(random.uniform(key, [d1 + 1]), axis=-1)

      lo = jnp.maximum(jnp.min(t0), jnp.min(t1)) + 0.1
      hi = jnp.minimum(jnp.max(t0), jnp.max(t1)) - 0.1
      rand = random.uniform(key, [num_pts], minval=lo, maxval=hi)

      pts = rand
      pts_ablate = rand[:-num_ablate] if num_ablate > 0 else pts

      w0 = []
      for i in range(len(t0) - 1):
        w0.append(jnp.mean((pts_ablate >= t0[i]) & (pts_ablate < t0[i + 1])))
      w0 = jnp.array(w0)

      w1 = []
      for i in range(len(t1) - 1):
        w1.append(jnp.mean((pts >= t1[i]) & (pts < t1[i + 1])))
      w1 = jnp.array(w1)

      all_zero &= jnp.all(stepfun.lossfun_outer(t0, w0, t1, w1) < eps)
    self.assertEqual(is_all_zero, all_zero)

  def test_inner_outer(self):
    """Two histograms of the same points will be bounds on each other."""
    rng = random.PRNGKey(4)
    for _ in range(10):
      key, rng = random.split(rng)
      d0, d1, num_pts = random.randint(key, [3], minval=10, maxval=20)

      key, rng = random.split(rng)
      t0 = jnp.sort(random.uniform(key, [d0 + 1]), axis=-1)

      key, rng = random.split(rng)
      t1 = jnp.sort(random.uniform(key, [d1 + 1]), axis=-1)

      lo = jnp.maximum(jnp.min(t0), jnp.min(t1)) + 0.1
      hi = jnp.minimum(jnp.max(t0), jnp.max(t1)) - 0.1
      pts = random.uniform(key, [num_pts], minval=lo, maxval=hi)

      w0 = []
      for i in range(len(t0) - 1):
        w0.append(jnp.sum((pts >= t0[i]) & (pts < t0[i + 1])))
      w0 = jnp.array(w0)

      w1 = []
      for i in range(len(t1) - 1):
        w1.append(jnp.sum((pts >= t1[i]) & (pts < t1[i + 1])))
      w1 = jnp.array(w1)

      w0_inner, w0_outer = stepfun.inner_outer(t0, t1, w1)
      w1_inner, w1_outer = stepfun.inner_outer(t1, t0, w0)

      self.assertTrue(jnp.all(w0_inner <= w0) and jnp.all(w0 <= w0_outer))
      self.assertTrue(jnp.all(w1_inner <= w1) and jnp.all(w1 <= w1_outer))

  def test_lossfun_outer_monotonic(self):
    """The loss is invariant to monotonic transformations on `t`."""
    rng = random.PRNGKey(0)

    curve_fn = lambda x: 1 + x**3  # Some monotonic transformation.

    for _ in range(10):
      key, rng = random.split(rng)
      d0, d1 = random.randint(key, [2], minval=10, maxval=20)

      key, rng = random.split(rng)
      t0 = jnp.sort(random.uniform(key, [d0 + 1]), axis=-1)

      key, rng = random.split(rng)
      t1 = jnp.sort(random.uniform(key, [d1 + 1]), axis=-1)

      key, rng = random.split(rng)
      w0 = jnp.exp(random.normal(key, [d0]))

      key, rng = random.split(rng)
      w1 = jnp.exp(random.normal(key, [d1]))

      excess = stepfun.lossfun_outer(t0, w0, t1, w1)
      curve_excess = stepfun.lossfun_outer(curve_fn(t0), w0, curve_fn(t1), w1)
      self.assertTrue(jnp.all(excess == curve_excess))

  def test_lossfun_outer_self_zero(self):
    """The loss is ~zero for the same (t, w) step function."""
    rng = random.PRNGKey(0)

    for _ in range(10):
      key, rng = random.split(rng)
      d = random.randint(key, (), minval=10, maxval=20)

      key, rng = random.split(rng)
      t = jnp.sort(random.uniform(key, [d + 1]), axis=-1)

      key, rng = random.split(rng)
      w = jnp.exp(random.normal(key, [d]))

      self.assertTrue(jnp.all(stepfun.lossfun_outer(t, w, t, w) < 1e-10))

  def test_outer_measure_reference(self):
    """Test that outer measures match a reference implementation."""
    rng = random.PRNGKey(0)
    for _ in range(10):
      key, rng = random.split(rng)
      d0, d1 = random.randint(key, [2], minval=10, maxval=20)

      key, rng = random.split(rng)
      t0 = jnp.sort(random.uniform(key, [d0 + 1]), axis=-1)

      key, rng = random.split(rng)
      t1 = jnp.sort(random.uniform(key, [d1 + 1]), axis=-1)

      key, rng = random.split(rng)
      w0 = jnp.exp(random.normal(key, [d0]))

      _, w1_outer = stepfun.inner_outer(t1, t0, w0)
      w1_outer_ref = outer(t1, t0, w0)
      np.testing.assert_allclose(w1_outer, w1_outer_ref, atol=1E-5, rtol=1E-5)

  def test_inner_measure_reference(self):
    """Test that inner measures match a reference implementation."""
    rng = random.PRNGKey(0)
    for _ in range(10):
      key, rng = random.split(rng)
      d0, d1 = random.randint(key, [2], minval=10, maxval=20)

      key, rng = random.split(rng)
      t0 = jnp.sort(random.uniform(key, [d0 + 1]), axis=-1)

      key, rng = random.split(rng)
      t1 = jnp.sort(random.uniform(key, [d1 + 1]), axis=-1)

      key, rng = random.split(rng)
      w0 = jnp.exp(random.normal(key, [d0]))

      w1_inner, _ = stepfun.inner_outer(t1, t0, w0)
      w1_inner_ref = inner(t1, t0, w0)
      np.testing.assert_allclose(w1_inner, w1_inner_ref, rtol=1e-5, atol=1e-5)

  def test_weighted_percentile(self):
    """Test that step function percentiles match the empirical percentile."""
    num_samples = 1000000
    rng = random.PRNGKey(0)
    for _ in range(10):
      rng, key = random.split(rng)
      d = random.randint(key, (), minval=10, maxval=20)

      rng, key = random.split(rng)
      ps = 100 * random.uniform(key, [3])

      key, rng = random.split(rng)
      t = jnp.sort(random.normal(key, [d + 1]), axis=-1)

      key, rng = random.split(rng)
      w = jax.nn.softmax(random.normal(key, [d]))

      key, rng = random.split(rng)
      samples = stepfun.sample(
          key, t, jnp.log(w), num_samples, single_jitter=False)
      true_percentiles = jnp.percentile(samples, ps)

      our_percentiles = stepfun.weighted_percentile(t, w, ps)
      np.testing.assert_allclose(
          our_percentiles, true_percentiles, rtol=1e-4, atol=1e-4)

  def test_weighted_percentile_vectorized(self):
    rng = random.PRNGKey(0)
    shape = (3, 4)
    d = 128

    rng, key = random.split(rng)
    ps = 100 * random.uniform(key, (5,))

    key, rng = random.split(rng)
    t = jnp.sort(random.normal(key, shape + (d + 1,)), axis=-1)

    key, rng = random.split(rng)
    w = jax.nn.softmax(random.normal(key, shape + (d,)))

    percentiles_vec = stepfun.weighted_percentile(t, w, ps)

    percentiles = []
    for i in range(shape[0]):
      percentiles.append([])
      for j in range(shape[1]):
        percentiles[i].append(stepfun.weighted_percentile(t[i, j], w[i, j], ps))
      percentiles[i] = jnp.stack(percentiles[i])
    percentiles = jnp.stack(percentiles)

    np.testing.assert_allclose(
        percentiles_vec, percentiles, rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(('', False), ('_avg', True))
  def test_resample_self_noop(self, use_avg):
    """Resampling a step function into itself should be a no-op."""
    d = 32
    rng = random.PRNGKey(0)

    key, rng = random.split(rng)
    tp = random.normal(rng, shape=(d + 1,))
    tp = jnp.sort(tp)

    key, rng = random.split(rng)
    vp = random.normal(key, shape=(d,))

    vp_recon = stepfun.resample(tp, tp, vp, use_avg=use_avg)
    np.testing.assert_allclose(vp, vp_recon, atol=1e-4)

  @parameterized.named_parameters(('', False), ('_avg', True))
  def test_resample_2x_downsample(self, use_avg):
    """Check resampling for a 2d downsample."""
    d = 32
    rng = random.PRNGKey(0)

    key, rng = random.split(rng)
    tp = random.normal(rng, shape=(d + 1,))
    tp = jnp.sort(tp)

    key, rng = random.split(rng)
    vp = random.normal(key, shape=(d,))

    t = tp[::2]

    v = stepfun.resample(t, tp, vp, use_avg=use_avg)

    vp2 = vp.reshape([-1, 2])
    dtp2 = jnp.diff(tp).reshape([-1, 2])
    if use_avg:
      v_true = jnp.sum(vp2 * dtp2, axis=-1) / jnp.sum(dtp2, axis=-1)
    else:
      v_true = jnp.sum(vp2, axis=-1)

    np.testing.assert_allclose(v, v_true, atol=1e-4)

  @parameterized.named_parameters(('', False), ('_avg', True))
  def test_resample_entire_interval(self, use_avg):
    """Check the sum (or weighted mean) of an entire interval."""
    d = 32
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    tp = random.normal(rng, shape=(d + 1,))
    tp = jnp.sort(tp)

    key, rng = random.split(rng)
    vp = random.normal(key, shape=(d,))

    t = jnp.array([jnp.min(tp), jnp.max(tp)])

    v = stepfun.resample(t, tp, vp, use_avg=use_avg)[0]
    if use_avg:
      v_true = jnp.sum(vp * jnp.diff(tp)) / sum(jnp.diff(tp))
    else:
      v_true = jnp.sum(vp)

    np.testing.assert_allclose(v, v_true, atol=1e-4)

  def test_resample_entire_domain(self):
    """Check the sum of the entire input domain."""
    d = 32
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    tp = random.normal(rng, shape=(d + 1,))
    tp = jnp.sort(tp)

    key, rng = random.split(rng)
    vp = random.normal(key, shape=(d,))

    t = jnp.array([-1e6, 1e6])

    v = stepfun.resample(t, tp, vp)[0]
    v_true = jnp.sum(vp)

    np.testing.assert_allclose(v, v_true, atol=1e-4)

  @parameterized.named_parameters(('', False), ('_avg', True))
  def test_resample_single_span(self, use_avg):
    """Check the sum (or weighted mean) of a single span."""
    d = 32
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    tp = random.normal(rng, shape=(d + 1,))
    tp = jnp.sort(tp)

    key, rng = random.split(rng)
    vp = random.normal(key, shape=(d,))

    pad = (tp[d // 2 + 1] - tp[d // 2]) / 4
    t = jnp.array([tp[d // 2] + pad, tp[d // 2 + 1] - pad])

    v = stepfun.resample(t, tp, vp, use_avg=use_avg)[0]
    if use_avg:
      v_true = vp[d // 2]
    else:
      v_true = vp[d // 2] * 0.5

    np.testing.assert_allclose(v, v_true, atol=1e-4)

  @parameterized.named_parameters(('', False), ('_avg', True))
  def test_resample_vectorized(self, use_avg):
    """Check that resample works with vectorized inputs."""
    shape = (3, 4)
    dp = 32
    d = 16
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    tp = random.normal(rng, shape=shape + (dp + 1,))
    tp = jnp.sort(tp)

    key, rng = random.split(rng)
    vp = random.normal(key, shape=shape + (dp,))

    key, rng = random.split(rng)
    t = random.normal(rng, shape=shape + (d + 1,))
    t = jnp.sort(t)

    v_batch = stepfun.resample(t, tp, vp, use_avg=use_avg)

    v_indiv = []
    for i in range(t.shape[0]):
      v_indiv.append(
          jnp.array([
              stepfun.resample(t[i, j], tp[i, j], vp[i, j], use_avg=use_avg)
              for j in range(t.shape[1])
          ]))
    v_indiv = jnp.array(v_indiv)

    np.testing.assert_allclose(v_batch, v_indiv, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
