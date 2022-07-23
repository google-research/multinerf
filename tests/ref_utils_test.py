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

"""Tests for ref_utils."""

from absl.testing import absltest
from internal import ref_utils
from jax import random
import jax.numpy as jnp
import numpy as np
import scipy


def generate_dir_enc_fn_scipy(deg_view):
  """Return spherical harmonics using scipy.special.sph_harm."""
  ml_array = ref_utils.get_ml_array(deg_view)

  def dir_enc_fn(theta, phi):
    de = [scipy.special.sph_harm(m, l, phi, theta) for m, l in ml_array.T]
    de = np.stack(de, axis=-1)
    # Split into real and imaginary parts.
    return np.concatenate([np.real(de), np.imag(de)], axis=-1)

  return dir_enc_fn


class RefUtilsTest(absltest.TestCase):

  def test_reflection(self):
    """Make sure reflected vectors have the same angle from normals as input."""
    rng = random.PRNGKey(0)
    for shape in [(45, 3), (4, 7, 3)]:
      key, rng = random.split(rng)
      normals = random.normal(key, shape)
      key, rng = random.split(rng)
      directions = random.normal(key, shape)

      # Normalize normal vectors.
      normals = normals / (
          jnp.linalg.norm(normals, axis=-1, keepdims=True) + 1e-10)

      reflected_directions = ref_utils.reflect(directions, normals)

      cos_angle_original = jnp.sum(directions * normals, axis=-1)
      cos_angle_reflected = jnp.sum(reflected_directions * normals, axis=-1)

      np.testing.assert_allclose(
          cos_angle_original, cos_angle_reflected, atol=1E-5, rtol=1E-5)

  def test_spherical_harmonics(self):
    """Make sure the fast spherical harmonics are accurate."""
    shape = (12, 11, 13)

    # Generate random points on sphere.
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    theta = random.uniform(key1, shape, minval=0.0, maxval=jnp.pi)
    phi = random.uniform(key2, shape, minval=0.0, maxval=2.0*jnp.pi)

    # Convert to Cartesian coordinates.
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)
    xyz = jnp.stack([x, y, z], axis=-1)

    deg_view = 5
    de = ref_utils.generate_dir_enc_fn(deg_view)(xyz)
    de_scipy = generate_dir_enc_fn_scipy(deg_view)(theta, phi)

    np.testing.assert_allclose(
        de, de_scipy, atol=0.02, rtol=1e6)  # Only use atol.
    self.assertFalse(jnp.any(jnp.isnan(de)))


if __name__ == '__main__':
  absltest.main()
