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

"""Unit tests for math."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from internal import math
import jax
from jax import random
import jax.numpy as jnp
import numpy as np


def safe_trig_harness(fn, max_exp):
  x = 10**np.linspace(-30, max_exp, 10000)
  x = np.concatenate([-x[::-1], np.array([0]), x])
  y_true = getattr(np, fn)(x)
  y = getattr(math, 'safe_' + fn)(x)
  return y_true, y


class MathTest(parameterized.TestCase):

  def test_sin(self):
    """In [-1e10, 1e10] safe_sin and safe_cos are accurate."""
    for fn in ['sin', 'cos']:
      y_true, y = safe_trig_harness(fn, 10)
      self.assertLess(jnp.max(jnp.abs(y - y_true)), 1e-4)
      self.assertFalse(jnp.any(jnp.isnan(y)))
    # Beyond that range it's less accurate but we just don't want it to be NaN.
    for fn in ['sin', 'cos']:
      y_true, y = safe_trig_harness(fn, 60)
      self.assertFalse(jnp.any(jnp.isnan(y)))

  def test_safe_exp_correct(self):
    """math.safe_exp() should match np.exp() for not-huge values."""
    x = jnp.linspace(-80, 80, 10001)
    y = math.safe_exp(x)
    g = jax.vmap(jax.grad(math.safe_exp))(x)
    yg_true = jnp.exp(x)
    np.testing.assert_allclose(y, yg_true)
    np.testing.assert_allclose(g, yg_true)

  def test_safe_exp_finite(self):
    """math.safe_exp() behaves reasonably for huge values."""
    x = jnp.linspace(-100000, 100000, 10001)
    y = math.safe_exp(x)
    g = jax.vmap(jax.grad(math.safe_exp))(x)
    # `y` and `g` should both always be finite.
    self.assertTrue(jnp.all(jnp.isfinite(y)))
    self.assertTrue(jnp.all(jnp.isfinite(g)))
    # The derivative of exp() should be exp().
    np.testing.assert_allclose(y, g)
    # safe_exp()'s output and gradient should be monotonic.
    self.assertTrue(jnp.all(y[1:] >= y[:-1]))
    self.assertTrue(jnp.all(g[1:] >= g[:-1]))

  def test_learning_rate_decay(self):
    rng = random.PRNGKey(0)
    for _ in range(10):
      key, rng = random.split(rng)
      lr_init = jnp.exp(random.normal(key) - 3)
      key, rng = random.split(rng)
      lr_final = lr_init * jnp.exp(random.normal(key) - 5)
      key, rng = random.split(rng)
      max_steps = int(jnp.ceil(100 + 100 * jnp.exp(random.normal(key))))

      lr_fn = functools.partial(
          math.learning_rate_decay,
          lr_init=lr_init,
          lr_final=lr_final,
          max_steps=max_steps)

      # Test that the rate at the beginning is the initial rate.
      np.testing.assert_allclose(lr_fn(0), lr_init, atol=1E-5, rtol=1E-5)

      # Test that the rate at the end is the final rate.
      np.testing.assert_allclose(
          lr_fn(max_steps), lr_final, atol=1E-5, rtol=1E-5)

      # Test that the rate at the middle is the geometric mean of the two rates.
      np.testing.assert_allclose(
          lr_fn(max_steps / 2),
          jnp.sqrt(lr_init * lr_final),
          atol=1E-5,
          rtol=1E-5)

      # Test that the rate past the end is the final rate
      np.testing.assert_allclose(
          lr_fn(max_steps + 100), lr_final, atol=1E-5, rtol=1E-5)

  def test_delayed_learning_rate_decay(self):
    rng = random.PRNGKey(0)
    for _ in range(10):
      key, rng = random.split(rng)
      lr_init = jnp.exp(random.normal(key) - 3)
      key, rng = random.split(rng)
      lr_final = lr_init * jnp.exp(random.normal(key) - 5)
      key, rng = random.split(rng)
      max_steps = int(jnp.ceil(100 + 100 * jnp.exp(random.normal(key))))
      key, rng = random.split(rng)
      lr_delay_steps = int(
          random.uniform(key, minval=0.1, maxval=0.4) * max_steps)
      key, rng = random.split(rng)
      lr_delay_mult = jnp.exp(random.normal(key) - 3)

      lr_fn = functools.partial(
          math.learning_rate_decay,
          lr_init=lr_init,
          lr_final=lr_final,
          max_steps=max_steps,
          lr_delay_steps=lr_delay_steps,
          lr_delay_mult=lr_delay_mult)

      # Test that the rate at the beginning is the delayed initial rate.
      np.testing.assert_allclose(
          lr_fn(0), lr_delay_mult * lr_init, atol=1E-5, rtol=1E-5)

      # Test that the rate at the end is the final rate.
      np.testing.assert_allclose(
          lr_fn(max_steps), lr_final, atol=1E-5, rtol=1E-5)

      # Test that the rate at after the delay is over is the usual rate.
      np.testing.assert_allclose(
          lr_fn(lr_delay_steps),
          math.learning_rate_decay(lr_delay_steps, lr_init, lr_final,
                                   max_steps),
          atol=1E-5,
          rtol=1E-5)

      # Test that the rate at the middle is the geometric mean of the two rates.
      np.testing.assert_allclose(
          lr_fn(max_steps / 2),
          jnp.sqrt(lr_init * lr_final),
          atol=1E-5,
          rtol=1E-5)

      # Test that the rate past the end is the final rate
      np.testing.assert_allclose(
          lr_fn(max_steps + 100), lr_final, atol=1E-5, rtol=1E-5)

  @parameterized.named_parameters(('', False), ('sort', True))
  def test_interp(self, sort):
    n, d0, d1 = 100, 10, 20
    rng = random.PRNGKey(0)

    key, rng = random.split(rng)
    x = random.normal(key, [n, d0])

    key, rng = random.split(rng)
    xp = random.normal(key, [n, d1])

    key, rng = random.split(rng)
    fp = random.normal(key, [n, d1])

    if sort:
      xp = jnp.sort(xp, axis=-1)
      fp = jnp.sort(fp, axis=-1)
      z = math.sorted_interp(x, xp, fp)
    else:
      z = math.interp(x, xp, fp)

    z_true = jnp.stack([jnp.interp(x[i], xp[i], fp[i]) for i in range(n)])
    np.testing.assert_allclose(z, z_true, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
