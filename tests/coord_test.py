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

"""Unit tests for coord."""

from absl.testing import absltest
from absl.testing import parameterized
from internal import coord
from internal import math
import jax
from jax import random
import jax.numpy as jnp
import numpy as np


def sample_covariance(rng, batch_size, num_dims):
  """Sample a random covariance matrix."""
  half_cov = jax.random.normal(rng, [batch_size] + [num_dims] * 2)
  cov = math.matmul(half_cov, jnp.moveaxis(half_cov, -1, -2))
  return cov


def stable_pos_enc(x, n):
  """A stable pos_enc for very high degrees, courtesy of Sameer Agarwal."""
  sin_x = np.sin(x)
  cos_x = np.cos(x)
  output = []
  rotmat = np.array([[cos_x, -sin_x], [sin_x, cos_x]], dtype='double')
  for _ in range(n):
    output.append(rotmat[::-1, 0, :])
    rotmat = np.einsum('ijn,jkn->ikn', rotmat, rotmat)
  return np.reshape(np.transpose(np.stack(output, 0), [2, 1, 0]), [-1, 2 * n])


class CoordTest(parameterized.TestCase):

  def test_stable_pos_enc(self):
    """Test that the stable posenc implementation works on multiples of pi/2."""
    n = 10
    x = np.linspace(-np.pi, np.pi, 5)
    z = stable_pos_enc(x, n).reshape([-1, 2, n])
    z0_true = np.zeros_like(z[:, 0, :])
    z1_true = np.ones_like(z[:, 1, :])
    z0_true[:, 0] = [0, -1, 0, 1, 0]
    z1_true[:, 0] = [-1, 0, 1, 0, -1]
    z1_true[:, 1] = [1, -1, 1, -1, 1]
    z_true = np.stack([z0_true, z1_true], axis=1)
    np.testing.assert_allclose(z, z_true, atol=1e-10)

  def test_contract_matches_special_case(self):
    """Test the math for Figure 2 of https://arxiv.org/abs/2111.12077."""
    n = 10
    _, s_to_t = coord.construct_ray_warps(jnp.reciprocal, 1, jnp.inf)
    s = jnp.linspace(0, 1 - jnp.finfo(jnp.float32).eps, n + 1)
    tc = coord.contract(s_to_t(s)[:, None])[:, 0]
    delta_tc = tc[1:] - tc[:-1]
    np.testing.assert_allclose(
        delta_tc, np.full_like(delta_tc, 1 / n), atol=1E-5, rtol=1E-5)

  def test_contract_is_bounded(self):
    n, d = 10000, 3
    rng = random.PRNGKey(0)
    key0, key1, rng = random.split(rng, 3)
    x = jnp.where(random.bernoulli(key0, shape=[n, d]), 1, -1) * jnp.exp(
        random.uniform(key1, [n, d], minval=-10, maxval=10))
    y = coord.contract(x)
    self.assertLessEqual(jnp.max(y), 2)

  def test_contract_is_noop_when_norm_is_leq_one(self):
    n, d = 10000, 3
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    x = random.normal(key, shape=[n, d])
    xc = x / jnp.maximum(1, jnp.linalg.norm(x, axis=-1, keepdims=True))

    # Sanity check on the test itself.
    assert jnp.abs(jnp.max(jnp.linalg.norm(xc, axis=-1)) - 1) < 1e-6

    yc = coord.contract(xc)
    np.testing.assert_allclose(xc, yc, atol=1E-5, rtol=1E-5)

  def test_contract_gradients_are_finite(self):
    # Construct x such that we probe x == 0, where things are unstable.
    x = jnp.stack(jnp.meshgrid(*[jnp.linspace(-4, 4, 11)] * 2), axis=-1)
    grad = jax.grad(lambda x: jnp.sum(coord.contract(x)))(x)
    self.assertTrue(jnp.all(jnp.isfinite(grad)))

  def test_inv_contract_gradients_are_finite(self):
    z = jnp.stack(jnp.meshgrid(*[jnp.linspace(-2, 2, 21)] * 2), axis=-1)
    z = z.reshape([-1, 2])
    z = z[jnp.sum(z**2, axis=-1) < 2, :]
    grad = jax.grad(lambda z: jnp.sum(coord.inv_contract(z)))(z)
    self.assertTrue(jnp.all(jnp.isfinite(grad)))

  def test_inv_contract_inverts_contract(self):
    """Do a round-trip from metric space to contracted space and back."""
    x = jnp.stack(jnp.meshgrid(*[jnp.linspace(-4, 4, 11)] * 2), axis=-1)
    x_recon = coord.inv_contract(coord.contract(x))
    np.testing.assert_allclose(x, x_recon, atol=1E-5, rtol=1E-5)

  @parameterized.named_parameters(
      ('05_1e-5', 5, 1e-5),
      ('10_1e-4', 10, 1e-4),
      ('15_0.005', 15, 0.005),
      ('20_0.2', 20, 0.2),  # At high degrees, our implementation is unstable.
      ('25_2', 25, 2),  # 2 is the maximum possible error.
      ('30_2', 30, 2),
  )
  def test_pos_enc(self, n, tol):
    """test pos_enc against a stable recursive implementation."""
    x = np.linspace(-np.pi, np.pi, 10001)
    z = coord.pos_enc(x[:, None], 0, n, append_identity=False)
    z_stable = stable_pos_enc(x, n)
    max_err = np.max(np.abs(z - z_stable))
    print(f'PE of degree {n} has a maximum error of {max_err}')
    self.assertLess(max_err, tol)

  def test_pos_enc_matches_integrated(self):
    """Integrated positional encoding with a variance of zero must be pos_enc."""
    min_deg = 0
    max_deg = 10
    np.linspace(-jnp.pi, jnp.pi, 10)
    x = jnp.stack(
        jnp.meshgrid(*[np.linspace(-jnp.pi, jnp.pi, 10)] * 2), axis=-1)
    x = np.linspace(-jnp.pi, jnp.pi, 10000)
    z_ipe = coord.integrated_pos_enc(x, jnp.zeros_like(x), min_deg, max_deg)
    z_pe = coord.pos_enc(x, min_deg, max_deg, append_identity=False)
    # We're using a pretty wide tolerance because IPE uses safe_sin().
    np.testing.assert_allclose(z_pe, z_ipe, atol=1e-4)

  def test_track_linearize(self):
    rng = random.PRNGKey(0)
    batch_size = 20
    for _ in range(30):
      # Construct some random Gaussians with dimensionalities in [1, 10].
      key, rng = random.split(rng)
      in_dims = random.randint(key, (), 1, 10)
      key, rng = random.split(rng)
      mean = jax.random.normal(key, [batch_size, in_dims])
      key, rng = random.split(rng)
      cov = sample_covariance(key, batch_size, in_dims)
      key, rng = random.split(rng)
      out_dims = random.randint(key, (), 1, 10)

      # Construct a random affine transformation.
      key, rng = random.split(rng)
      a_mat = jax.random.normal(key, [out_dims, in_dims])
      key, rng = random.split(rng)
      b = jax.random.normal(key, [out_dims])

      def fn(x):
        x_vec = x.reshape([-1, x.shape[-1]])
        y_vec = jax.vmap(lambda z: math.matmul(a_mat, z))(x_vec) + b  # pylint:disable=cell-var-from-loop
        y = y_vec.reshape(list(x.shape[:-1]) + [y_vec.shape[-1]])
        return y

      # Apply the affine function to the Gaussians.
      fn_mean_true = fn(mean)
      fn_cov_true = math.matmul(math.matmul(a_mat, cov), a_mat.T)

      # Tracking the Gaussians through a linearized function of a linear
      # operator should be the same.
      fn_mean, fn_cov = coord.track_linearize(fn, mean, cov)
      np.testing.assert_allclose(fn_mean, fn_mean_true, atol=1E-5, rtol=1E-5)
      np.testing.assert_allclose(fn_cov, fn_cov_true, atol=1e-5, rtol=1e-5)

  @parameterized.named_parameters(('reciprocal', jnp.reciprocal),
                                  ('log', jnp.log), ('sqrt', jnp.sqrt))
  def test_construct_ray_warps_extents(self, fn):
    n = 100
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    t_near = jnp.exp(jax.random.normal(key, [n]))
    key, rng = random.split(rng)
    t_far = t_near + jnp.exp(jax.random.normal(key, [n]))

    t_to_s, s_to_t = coord.construct_ray_warps(fn, t_near, t_far)

    np.testing.assert_allclose(
        t_to_s(t_near), jnp.zeros_like(t_near), atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(
        t_to_s(t_far), jnp.ones_like(t_far), atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(
        s_to_t(jnp.zeros_like(t_near)), t_near, atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(
        s_to_t(jnp.ones_like(t_near)), t_far, atol=1E-5, rtol=1E-5)

  def test_construct_ray_warps_special_reciprocal(self):
    """Test fn=1/x against its closed form."""
    n = 100
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    t_near = jnp.exp(jax.random.normal(key, [n]))
    key, rng = random.split(rng)
    t_far = t_near + jnp.exp(jax.random.normal(key, [n]))

    key, rng = random.split(rng)
    u = jax.random.uniform(key, [n])
    t = t_near * (1 - u) + t_far * u
    key, rng = random.split(rng)
    s = jax.random.uniform(key, [n])

    t_to_s, s_to_t = coord.construct_ray_warps(jnp.reciprocal, t_near, t_far)

    # Special cases for fn=reciprocal.
    s_to_t_ref = lambda s: 1 / (s / t_far + (1 - s) / t_near)
    t_to_s_ref = lambda t: (t_far * (t - t_near)) / (t * (t_far - t_near))

    np.testing.assert_allclose(t_to_s(t), t_to_s_ref(t), atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(s_to_t(s), s_to_t_ref(s), atol=1E-5, rtol=1E-5)

  def test_expected_sin(self):
    normal_samples = random.normal(random.PRNGKey(0), (10000,))
    for mu, var in [(0, 1), (1, 3), (-2, .2), (10, 10)]:
      sin_mu = coord.expected_sin(mu, var)
      x = jnp.sin(jnp.sqrt(var) * normal_samples + mu)
      np.testing.assert_allclose(sin_mu, jnp.mean(x), atol=1e-2)

  def test_integrated_pos_enc(self):
    num_dims = 2  # The number of input dimensions.
    min_deg = 0  # Must be 0 for this test to work.
    max_deg = 4
    num_samples = 100000
    rng = random.PRNGKey(0)
    for _ in range(5):
      # Generate a coordinate's mean and covariance matrix.
      key, rng = random.split(rng)
      mean = random.normal(key, (2,))
      key, rng = random.split(rng)
      half_cov = jax.random.normal(key, [num_dims] * 2)
      cov = half_cov @ half_cov.T
      var = jnp.diag(cov)
      # Generate an IPE.
      enc = coord.integrated_pos_enc(
          mean,
          var,
          min_deg,
          max_deg,
      )

      # Draw samples, encode them, and take their mean.
      key, rng = random.split(rng)
      samples = random.multivariate_normal(key, mean, cov, [num_samples])
      assert min_deg == 0
      enc_samples = np.concatenate(
          [stable_pos_enc(x, max_deg) for x in tuple(samples.T)], axis=-1)
      # Correct for a different dimension ordering in stable_pos_enc.
      enc_gt = jnp.mean(enc_samples, 0)
      enc_gt = enc_gt.reshape([num_dims, max_deg * 2]).T.reshape([-1])
      np.testing.assert_allclose(enc, enc_gt, rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
  absltest.main()
