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

"""Unit tests for render."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from internal import math
from internal import render
import jax
from jax import random
import jax.numpy as jnp
import numpy as np


def surface_stats(points):
  """Get the sample mean and covariance matrix of a set of matrices [..., d]."""
  means = jnp.mean(points, -1)
  centered = points - means[..., None]
  covs = jnp.mean(centered[..., None, :, :] * centered[..., :, None, :], -1)
  return means, covs


def sqrtm(mat):
  """Take the matrix square root of a PSD matrix [..., d, d]."""
  eigval, eigvec = jax.scipy.linalg.eigh(mat)
  scaling = jnp.sqrt(jnp.maximum(0., eigval))[..., None, :]
  return math.matmul(eigvec * scaling, jnp.moveaxis(eigvec, -2, -1))


def control_points(mean, cov):
  """Construct "sigma points" using a matrix sqrt (Cholesky or SVD are fine)."""
  sqrtm_cov = sqrtm(cov)  # or could be jax.scipy.linalg.cholesky(cov)
  offsets = jnp.sqrt(mean.shape[-1] + 0.5) * jnp.concatenate(
      [jnp.zeros_like(mean[..., None]), sqrtm_cov, -sqrtm_cov], -1)
  return mean[..., None] + offsets


def inside_conical_frustum(x, d, t0, t1, r, ttol=1e-6, rtol=1e-6):
  """Test if `x` is inside the conical frustum specified by the other inputs."""
  d_normsq = jnp.sum(d**2)
  d_norm = jnp.sqrt(d_normsq)
  x_normsq = jnp.sum(x**2, -1)
  x_norm = jnp.sqrt(x_normsq)
  xd = math.matmul(x, d)
  is_inside = (
      (t0 - ttol) <= xd / d_normsq) & (xd / d_normsq <= (t1 + ttol)) & (
          (xd / (d_norm * x_norm)) >=
          (1 / jnp.sqrt(1 + r**2 / d_normsq) - rtol))
  return is_inside


def sample_conical_frustum(rng, num_samples, d, t0, t1, base_radius):
  """Draw random samples from a conical frustum.

  Args:
    rng: The RNG seed.
    num_samples: int, the number of samples to draw.
    d: jnp.float32 3-vector, the axis of the cone.
    t0: float, the starting distance of the frustum.
    t1: float, the ending distance of the frustum.
    base_radius: float, the scale of the radius as a function of distance.

  Returns:
    A matrix of samples.
  """
  key, rng = random.split(rng)
  u = random.uniform(key, shape=[num_samples])
  t = (t0**3 * (1 - u) + t1**3 * u)**(1 / 3)
  key, rng = random.split(rng)
  theta = random.uniform(key, shape=[num_samples], minval=0, maxval=jnp.pi * 2)
  key, rng = random.split(rng)
  r = base_radius * t * jnp.sqrt(random.uniform(key, shape=[num_samples]))

  d_norm = d / jnp.linalg.norm(d)
  null = jnp.eye(3) - d_norm[:, None] * d_norm[None, :]
  basis = jnp.linalg.svd(null)[0][:, :2]
  rot_samples = ((basis[:, 0:1] * r * jnp.cos(theta)) +
                 (basis[:, 1:2] * r * jnp.sin(theta)) + d[:, None] * t).T
  return rot_samples


def generate_random_cylinder(rng, num_zs=4):
  t0, t1 = [], []
  for _ in range(num_zs):
    rng, key = random.split(rng)
    z_mean = random.uniform(key, minval=1.5, maxval=3)
    rng, key = random.split(rng)
    z_delta = random.uniform(key, minval=0.1, maxval=.3)
    t0.append(z_mean - z_delta)
    t1.append(z_mean + z_delta)
  t0 = jnp.array(t0)
  t1 = jnp.array(t1)

  rng, key = random.split(rng)
  radius = random.uniform(key, minval=0.1, maxval=.2)

  rng, key = random.split(rng)
  raydir = random.normal(key, [3])
  raydir = raydir / jnp.sqrt(jnp.sum(raydir**2, -1))

  rng, key = random.split(rng)
  scale = random.uniform(key, minval=0.4, maxval=1.2)
  raydir = scale * raydir

  return raydir, t0, t1, radius


def generate_random_conical_frustum(rng, num_zs=4):
  t0, t1 = [], []
  for _ in range(num_zs):
    rng, key = random.split(rng)
    z_mean = random.uniform(key, minval=1.5, maxval=3)
    rng, key = random.split(rng)
    z_delta = random.uniform(key, minval=0.1, maxval=.3)
    t0.append(z_mean - z_delta)
    t1.append(z_mean + z_delta)
  t0 = jnp.array(t0)
  t1 = jnp.array(t1)

  rng, key = random.split(rng)
  r = random.uniform(key, minval=0.01, maxval=.05)

  rng, key = random.split(rng)
  raydir = random.normal(key, [3])
  raydir = raydir / jnp.sqrt(jnp.sum(raydir**2, -1))

  rng, key = random.split(rng)
  scale = random.uniform(key, minval=0.8, maxval=1.2)
  raydir = scale * raydir

  return raydir, t0, t1, r


def cylinder_to_gaussian_sample(key,
                                raydir,
                                t0,
                                t1,
                                radius,
                                padding=1,
                                num_samples=1000000):
  # Sample uniformly from a cube that surrounds the entire conical frustom.
  z_max = max(t0, t1)
  samples = random.uniform(
      key, [num_samples, 3],
      minval=jnp.min(raydir) * z_max - padding,
      maxval=jnp.max(raydir) * z_max + padding)

  # Grab only the points within the cylinder.
  raydir_magsq = jnp.sum(raydir**2, -1, keepdims=True)
  proj = (raydir * (samples @ raydir)[:, None]) / raydir_magsq
  dist = samples @ raydir
  mask = (dist >= raydir_magsq * t0) & (dist <= raydir_magsq * t1) & (
      jnp.sum((proj - samples)**2, -1) < radius**2)
  samples = samples[mask, :]

  # Compute their mean and covariance.
  mean = jnp.mean(samples, 0)
  cov = jnp.cov(samples.T, bias=False)
  return mean, cov


def conical_frustum_to_gaussian_sample(key, raydir, t0, t1, r):
  """A brute-force numerical approximation to conical_frustum_to_gaussian()."""
  # Sample uniformly from a cube that surrounds the entire conical frustum.
  samples = sample_conical_frustum(key, 100000, raydir, t0, t1, r)
  # Compute their mean and covariance.
  return surface_stats(samples.T)


def finite_output_and_gradient(fn, args):
  """True if fn(*args) and all of its gradients are finite."""
  vals = fn(*args)
  is_finite = True
  for do, v in enumerate(vals):
    is_finite &= jnp.all(jnp.isfinite(v))
    # pylint: disable=cell-var-from-loop
    grads = jax.grad(
        lambda *x: jnp.sum(fn(*x)[do]), argnums=range(len(args)))(*args)
    for g in grads:
      is_finite &= jnp.all(jnp.isfinite(g))
  return is_finite


class RenderTest(parameterized.TestCase):

  def test_cylinder_scaling(self):
    d = jnp.array([0., 0., 1.])
    t0 = jnp.array([0.3])
    t1 = jnp.array([0.7])
    radius = jnp.array([0.4])
    mean, cov = render.cylinder_to_gaussian(
        d,
        t0,
        t1,
        radius,
        False,
    )
    scale = 2.7
    scaled_mean, scaled_cov = render.cylinder_to_gaussian(
        scale * d,
        t0,
        t1,
        radius,
        False,
    )
    np.testing.assert_allclose(scale * mean, scaled_mean, atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(
        scale**2 * cov[2, 2], scaled_cov[2, 2], atol=1E-5, rtol=1E-5)
    control = control_points(mean, cov)[0]
    control_scaled = control_points(scaled_mean, scaled_cov)[0]
    np.testing.assert_allclose(
        control[:2, :], control_scaled[:2, :], atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(
        control[2, :] * scale, control_scaled[2, :], atol=1E-5, rtol=1E-5)

  def test_conical_frustum_scaling(self):
    d = jnp.array([0., 0., 1.])
    t0 = jnp.array([0.3])
    t1 = jnp.array([0.7])
    radius = jnp.array([0.4])
    mean, cov = render.conical_frustum_to_gaussian(
        d,
        t0,
        t1,
        radius,
        False,
    )
    scale = 2.7
    scaled_mean, scaled_cov = render.conical_frustum_to_gaussian(
        scale * d,
        t0,
        t1,
        radius,
        False,
    )
    np.testing.assert_allclose(scale * mean, scaled_mean, atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(
        scale**2 * cov[2, 2], scaled_cov[2, 2], atol=1E-5, rtol=1E-5)
    control = control_points(mean, cov)[0]
    control_scaled = control_points(scaled_mean, scaled_cov)[0]
    np.testing.assert_allclose(
        control[:2, :], control_scaled[:2, :], atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(
        control[2, :] * scale, control_scaled[2, :], atol=1E-5, rtol=1E-5)

  def test_control_points(self):
    rng = random.PRNGKey(0)
    batch_size = 10
    for num_dims in [1, 2, 3]:
      key, rng = random.split(rng)
      mean = jax.random.normal(key, [batch_size, num_dims])
      key, rng = random.split(rng)
      half_cov = jax.random.normal(key, [batch_size] + [num_dims] * 2)
      cov = half_cov @ jnp.moveaxis(half_cov, -1, -2)

      sqrtm_cov = sqrtm(cov)
      np.testing.assert_allclose(
          sqrtm_cov @ sqrtm_cov, cov, atol=1e-5, rtol=1E-5)

      points = control_points(mean, cov)
      mean_recon, cov_recon = surface_stats(points)
      np.testing.assert_allclose(mean, mean_recon, atol=1E-5, rtol=1E-5)
      np.testing.assert_allclose(cov, cov_recon, atol=1e-5, rtol=1E-5)

  def test_conical_frustum(self):
    rng = random.PRNGKey(0)
    data = []
    for _ in range(10):
      key, rng = random.split(rng)
      raydir, t0, t1, r = generate_random_conical_frustum(key)
      i_results = []
      for i_t0, i_t1 in zip(t0, t1):
        key, rng = random.split(rng)
        i_results.append(
            conical_frustum_to_gaussian_sample(key, raydir, i_t0, i_t1, r))
      mean_gt, cov_gt = [jnp.stack(x, 0) for x in zip(*i_results)]
      data.append((raydir, t0, t1, r, mean_gt, cov_gt))
    raydir, t0, t1, r, mean_gt, cov_gt = [jnp.stack(x, 0) for x in zip(*data)]
    diag_cov_gt = jax.vmap(jax.vmap(jnp.diag))(cov_gt)
    for diag in [False, True]:
      for stable in [False, True]:
        mean, cov = render.conical_frustum_to_gaussian(
            raydir, t0, t1, r[..., None], diag, stable=stable)
        np.testing.assert_allclose(mean, mean_gt, atol=0.001)
        if diag:
          np.testing.assert_allclose(cov, diag_cov_gt, atol=0.0002)
        else:
          np.testing.assert_allclose(cov, cov_gt, atol=0.0002)

  def test_inside_conical_frustum(self):
    """This test only tests helper functions used by other tests."""
    rng = random.PRNGKey(0)
    for _ in range(20):
      key, rng = random.split(rng)
      d, t0, t1, r = generate_random_conical_frustum(key, num_zs=1)
      key, rng = random.split(rng)
      # Sample some points.
      samples = sample_conical_frustum(key, 1000000, d, t0, t1, r)
      # Check that they're all inside.
      check = lambda x: inside_conical_frustum(x, d, t0, t1, r)  # pylint: disable=cell-var-from-loop
      self.assertTrue(jnp.all(check(samples)))
      # Check that wiggling them a little puts some outside (potentially flaky).
      self.assertFalse(jnp.all(check(samples + 1e-3)))
      self.assertFalse(jnp.all(check(samples - 1e-3)))

  def test_conical_frustum_stable(self):
    rng = random.PRNGKey(0)
    for _ in range(10):
      key, rng = random.split(rng)
      d, t0, t1, r = generate_random_conical_frustum(key)
      for diag in [False, True]:
        mean, cov = render.conical_frustum_to_gaussian(
            d, t0, t1, r, diag, stable=False)
        mean_stable, cov_stable = render.conical_frustum_to_gaussian(
            d, t0, t1, r, diag, stable=True)
        np.testing.assert_allclose(mean, mean_stable, atol=1e-7, rtol=1E-5)
        np.testing.assert_allclose(cov, cov_stable, atol=1e-5, rtol=1E-5)

  def test_cylinder(self):
    rng = random.PRNGKey(0)
    data = []
    for _ in range(10):
      key, rng = random.split(rng)
      raydir, t0, t1, radius = generate_random_cylinder(rng)
      key, rng = random.split(rng)
      i_results = []
      for i_t0, i_t1 in zip(t0, t1):
        i_results.append(
            cylinder_to_gaussian_sample(key, raydir, i_t0, i_t1, radius))
      mean_gt, cov_gt = [jnp.stack(x, 0) for x in zip(*i_results)]
      data.append((raydir, t0, t1, radius, mean_gt, cov_gt))
    raydir, t0, t1, radius, mean_gt, cov_gt = [
        jnp.stack(x, 0) for x in zip(*data)
    ]
    mean, cov = (
        render.cylinder_to_gaussian(raydir, t0, t1, radius[..., None], False))
    np.testing.assert_allclose(mean, mean_gt, atol=0.1)
    np.testing.assert_allclose(cov, cov_gt, atol=0.01)

  def test_lift_gaussian_diag(self):
    dims, n, m = 3, 10, 4
    rng = random.PRNGKey(0)
    key, rng = random.split(rng)
    d = random.normal(key, [n, dims])
    key, rng = random.split(rng)
    z_mean = random.normal(key, [n, m])
    key, rng = random.split(rng)
    z_var = jnp.exp(random.normal(key, [n, m]))
    key, rng = random.split(rng)
    xy_var = jnp.exp(random.normal(key, [n, m]))
    mean, cov = render.lift_gaussian(d, z_mean, z_var, xy_var, diag=False)
    mean_diag, cov_diag = render.lift_gaussian(
        d, z_mean, z_var, xy_var, diag=True)
    np.testing.assert_allclose(mean, mean_diag, atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(
        jax.vmap(jax.vmap(jnp.diag))(cov), cov_diag, atol=1E-5, rtol=1E-5)

  def test_rotated_conic_frustums(self):
    # Test that conic frustum Gaussians are closed under rotation.
    diag = False
    rng = random.PRNGKey(0)
    for _ in range(10):
      rng, key = random.split(rng)
      z_mean = random.uniform(key, minval=1.5, maxval=3)
      rng, key = random.split(rng)
      z_delta = random.uniform(key, minval=0.1, maxval=.3)
      t0 = jnp.array(z_mean - z_delta)
      t1 = jnp.array(z_mean + z_delta)

      rng, key = random.split(rng)
      r = random.uniform(key, minval=0.1, maxval=.2)

      rng, key = random.split(rng)
      d = random.normal(key, [3])

      mean, cov = render.conical_frustum_to_gaussian(d, t0, t1, r, diag)

      # Make a random rotation matrix.
      rng, key = random.split(rng)
      x = random.normal(key, [10, 3])
      rot_mat = x.T @ x
      u, _, v = jnp.linalg.svd(rot_mat)
      rot_mat = u @ v.T

      mean, cov = render.conical_frustum_to_gaussian(d, t0, t1, r, diag)
      rot_mean, rot_cov = render.conical_frustum_to_gaussian(
          rot_mat @ d, t0, t1, r, diag)
      gt_rot_mean, gt_rot_cov = surface_stats(
          rot_mat @ control_points(mean, cov))

      np.testing.assert_allclose(rot_mean, gt_rot_mean, atol=1E-5, rtol=1E-5)
      np.testing.assert_allclose(rot_cov, gt_rot_cov, atol=1E-5, rtol=1E-5)

  @parameterized.named_parameters(
      ('-100 -100', -100, -100),
      ('-100 -10', -100, -10),
      ('-100  0', -100, 0),
      ('-100  10', -100, 10),
      ('-10  -100', -10, -100),
      ('-10  -10', -10, -10),
      ('-10   0', -10, 0),
      ('-10   10', -10, 10),
      (' 0   -100', 0, -100),
      (' 0   -10', 0, -10),
      (' 0    0', 0, 0),
      (' 0    10', 0, 10),
      (' 10  -10', 10, -10),
      (' 10   0', 10, 0),
      (' 10   10', 10, 10),
      (' 10  -100', 10, -100),
  )
  def test_alpha_weights_finite(self, log_density_log_mult, tvals_log_mult):
    rng = random.PRNGKey(0)
    n, d = 100, 128

    key, rng = random.split(rng)
    density = jnp.exp(log_density_log_mult + random.normal(key, [n, d]))
    key, rng = random.split(rng)
    tvals_unsorted = 2 * random.uniform(key, [n, d + 1]) - 1
    tvals = jnp.exp(tvals_log_mult) * jnp.sort(tvals_unsorted, axis=-1)
    key, rng = random.split(rng)
    dirs = random.normal(key, [n, 3])

    fn = jax.jit(render.compute_alpha_weights)
    args = density, tvals, dirs

    self.assertTrue(finite_output_and_gradient(fn, args))

  def test_alpha_weights_delta_correct(self):
    """A single interval with a huge density should produce a spikey weight."""
    max_density = 1e10
    rng = random.PRNGKey(0)
    n, d = 100, 128

    key, rng = random.split(rng)
    r = random.normal(key, [n, d])
    mask = (r == jnp.max(r, axis=-1, keepdims=True))
    density = max_density * mask

    key, rng = random.split(rng)
    tvals_unsorted = 2 * random.uniform(key, [n, d + 1]) - 1
    tvals = jnp.sort(tvals_unsorted, axis=-1)

    key, rng = random.split(rng)
    dirs = random.normal(key, [n, 3])

    weights, alpha, _ = render.compute_alpha_weights(density, tvals, dirs)
    np.testing.assert_allclose(jnp.float32(mask), weights, atol=1E-5, rtol=1E-5)
    np.testing.assert_allclose(jnp.float32(mask), alpha, atol=1E-5, rtol=1E-5)

  @parameterized.named_parameters(
      ('-100_-100', -100, -100),
      ('-100_-10', -100, -10),
      ('-100__0', -100, 0),
      ('-100__10', -100, 10),
      ('-10__-100', -10, -100),
      ('-10__-10', -10, -10),
      ('-10___0', -10, 0),
      ('-10___10', -10, 10),
      ('_0___-100', 0, -100),
      ('_0___-10', 0, -10),
      ('_0____0', 0, 0),
      ('_0____10', 0, 10),
      ('_10__-10', 10, -10),
      ('_10___0', 10, 0),
      ('_10___10', 10, 10),
      ('_10__-100', 10, -100),
  )
  def test_conical_frustum_to_gaussian_finite(
      self,
      tvals_log_mult,
      radius_log_mult,
  ):
    n, d = 10, 128
    rng = random.PRNGKey(0)

    key, rng = random.split(rng)
    rad = jnp.exp(radius_log_mult) * jnp.exp(random.normal(key, [n, d]))

    key, rng = random.split(rng)
    tvals_unsorted = random.uniform(key, [n, d + 1], minval=-1, maxval=1)
    tvals = jnp.exp(tvals_log_mult) * jnp.sort(tvals_unsorted, axis=-1)

    key, rng = random.split(rng)
    dirs = random.normal(key, [n, 3])

    t0, t1 = tvals[..., :-1], tvals[..., 1:]

    fn = jax.jit(
        functools.partial(render.conical_frustum_to_gaussian, diag=True))
    args = dirs, t0, t1, rad

    self.assertTrue(finite_output_and_gradient(fn, args))


if __name__ == '__main__':
  absltest.main()
