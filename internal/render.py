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

"""Helper functions for shooting and rendering rays."""

from internal import stepfun
import jax.numpy as jnp


def lift_gaussian(d, t_mean, t_var, r_var, diag):
  """Lift a Gaussian defined along a ray to 3D coordinates."""
  mean = d[..., None, :] * t_mean[..., None]

  d_mag_sq = jnp.maximum(1e-10, jnp.sum(d**2, axis=-1, keepdims=True))

  if diag:
    d_outer_diag = d**2
    null_outer_diag = 1 - d_outer_diag / d_mag_sq
    t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
    xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
    cov_diag = t_cov_diag + xy_cov_diag
    return mean, cov_diag
  else:
    d_outer = d[..., :, None] * d[..., None, :]
    eye = jnp.eye(d.shape[-1])
    null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
    t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
    xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
    cov = t_cov + xy_cov
    return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
  """Approximate a conical frustum as a Gaussian distribution (mean+cov).

  Assumes the ray is originating from the origin, and base_radius is the
  radius at dist=1. Doesn't assume `d` is normalized.

  Args:
    d: jnp.float32 3-vector, the axis of the cone
    t0: float, the starting distance of the frustum.
    t1: float, the ending distance of the frustum.
    base_radius: float, the scale of the radius as a function of distance.
    diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
    stable: boolean, whether or not to use the stable computation described in
      the paper (setting this to False will cause catastrophic failure).

  Returns:
    a Gaussian (mean and covariance).
  """
  if stable:
    # Equation 7 in the paper (https://arxiv.org/abs/2103.13415).
    mu = (t0 + t1) / 2  # The average of the two `t` values.
    hw = (t1 - t0) / 2  # The half-width of the two `t` values.
    eps = jnp.finfo(jnp.float32).eps
    t_mean = mu + (2 * mu * hw**2) / jnp.maximum(eps, 3 * mu**2 + hw**2)
    denom = jnp.maximum(eps, 3 * mu**2 + hw**2)
    t_var = (hw**2) / 3 - (4 / 15) * hw**4 * (12 * mu**2 - hw**2) / denom**2
    r_var = (mu**2) / 4 + (5 / 12) * hw**2 - (4 / 15) * (hw**4) / denom
  else:
    # Equations 37-39 in the paper.
    t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
    r_var = 3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3)
    t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
    t_var = t_mosq - t_mean**2
  r_var *= base_radius**2
  return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cylinder_to_gaussian(d, t0, t1, radius, diag):
  """Approximate a cylinder as a Gaussian distribution (mean+cov).

  Assumes the ray is originating from the origin, and radius is the
  radius. Does not renormalize `d`.

  Args:
    d: jnp.float32 3-vector, the axis of the cylinder
    t0: float, the starting distance of the cylinder.
    t1: float, the ending distance of the cylinder.
    radius: float, the radius of the cylinder
    diag: boolean, whether or the Gaussian will be diagonal or full-covariance.

  Returns:
    a Gaussian (mean and covariance).
  """
  t_mean = (t0 + t1) / 2
  r_var = radius**2 / 4
  t_var = (t1 - t0)**2 / 12
  return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cast_rays(tdist, origins, directions, radii, ray_shape, diag=True):
  """Cast rays (cone- or cylinder-shaped) and featurize sections of it.

  Args:
    tdist: float array, the "fencepost" distances along the ray.
    origins: float array, the ray origin coordinates.
    directions: float array, the ray direction vectors.
    radii: float array, the radii (base radii for cones) of the rays.
    ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
    diag: boolean, whether or not the covariance matrices should be diagonal.

  Returns:
    a tuple of arrays of means and covariances.
  """
  t0 = tdist[..., :-1]
  t1 = tdist[..., 1:]
  if ray_shape == 'cone':
    gaussian_fn = conical_frustum_to_gaussian
  elif ray_shape == 'cylinder':
    gaussian_fn = cylinder_to_gaussian
  else:
    raise ValueError('ray_shape must be \'cone\' or \'cylinder\'')
  means, covs = gaussian_fn(directions, t0, t1, radii, diag)
  means = means + origins[..., None, :]
  return means, covs


def compute_alpha_weights(density, tdist, dirs, opaque_background=False):
  """Helper function for computing alpha compositing weights."""
  t_delta = tdist[..., 1:] - tdist[..., :-1]
  delta = t_delta * jnp.linalg.norm(dirs[..., None, :], axis=-1)
  density_delta = density * delta

  if opaque_background:
    # Equivalent to making the final t-interval infinitely wide.
    density_delta = jnp.concatenate([
        density_delta[..., :-1],
        jnp.full_like(density_delta[..., -1:], jnp.inf)
    ],
                                    axis=-1)

  alpha = 1 - jnp.exp(-density_delta)
  trans = jnp.exp(-jnp.concatenate([
      jnp.zeros_like(density_delta[..., :1]),
      jnp.cumsum(density_delta[..., :-1], axis=-1)
  ],
                                   axis=-1))
  weights = alpha * trans
  return weights, alpha, trans


def volumetric_rendering(rgbs,
                         weights,
                         tdist,
                         bg_rgbs,
                         t_far,
                         compute_extras,
                         extras=None):
  """Volumetric Rendering Function.

  Args:
    rgbs: jnp.ndarray(float32), color, [batch_size, num_samples, 3]
    weights: jnp.ndarray(float32), weights, [batch_size, num_samples].
    tdist: jnp.ndarray(float32), [batch_size, num_samples].
    bg_rgbs: jnp.ndarray(float32), the color(s) to use for the background.
    t_far: jnp.ndarray(float32), [batch_size, 1], the distance of the far plane.
    compute_extras: bool, if True, compute extra quantities besides color.
    extras: dict, a set of values along rays to render by alpha compositing.

  Returns:
    rendering: a dict containing an rgb image of size [batch_size, 3], and other
      visualizations if compute_extras=True.
  """
  eps = jnp.finfo(jnp.float32).eps
  rendering = {}

  acc = weights.sum(axis=-1)
  bg_w = jnp.maximum(0, 1 - acc[..., None])  # The weight of the background.
  rgb = (weights[..., None] * rgbs).sum(axis=-2) + bg_w * bg_rgbs
  rendering['rgb'] = rgb

  if compute_extras:
    rendering['acc'] = acc

    if extras is not None:
      for k, v in extras.items():
        if v is not None:
          rendering[k] = (weights[..., None] * v).sum(axis=-2)

    expectation = lambda x: (weights * x).sum(axis=-1) / jnp.maximum(eps, acc)
    t_mids = 0.5 * (tdist[..., :-1] + tdist[..., 1:])
    # For numerical stability this expectation is computing using log-distance.
    rendering['distance_mean'] = (
        jnp.clip(
            jnp.nan_to_num(jnp.exp(expectation(jnp.log(t_mids))), jnp.inf),
            tdist[..., 0], tdist[..., -1]))

    # Add an extra fencepost with the far distance at the end of each ray, with
    # whatever weight is needed to make the new weight vector sum to exactly 1
    # (`weights` is only guaranteed to sum to <= 1, not == 1).
    t_aug = jnp.concatenate([tdist, t_far], axis=-1)
    weights_aug = jnp.concatenate([weights, bg_w], axis=-1)

    ps = [5, 50, 95]
    distance_percentiles = stepfun.weighted_percentile(t_aug, weights_aug, ps)

    for i, p in enumerate(ps):
      s = 'median' if p == 50 else 'percentile_' + str(p)
      rendering['distance_' + s] = distance_percentiles[..., i]

  return rendering
