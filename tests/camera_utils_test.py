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

"""Tests for camera_utils."""

from absl.testing import absltest
from absl.testing import parameterized
from internal import camera_utils
from jax import random
import jax.numpy as jnp
import numpy as np


class CameraUtilsTest(parameterized.TestCase):

  def test_convert_to_ndc(self):
    rng = random.PRNGKey(0)
    for _ in range(10):
      # Random pinhole camera intrinsics.
      key, rng = random.split(rng)
      focal, width, height = random.uniform(key, (3,), minval=100., maxval=200.)
      camtopix = camera_utils.intrinsic_matrix(focal, focal, width / 2.,
                                               height / 2.)
      pixtocam = np.linalg.inv(camtopix)
      near = 1.

      # Random rays, pointing forward (negative z direction).
      num_rays = 1000
      key, rng = random.split(rng)
      origins = jnp.array([0., 0., 1.])
      origins += random.uniform(key, (num_rays, 3), minval=-1., maxval=1.)
      directions = jnp.array([0., 0., -1.])
      directions += random.uniform(key, (num_rays, 3), minval=-.5, maxval=.5)

      # Project world-space points along each ray into NDC space.
      t = jnp.linspace(0., 1., 10)
      pts_world = origins + t[:, None, None] * directions
      pts_ndc = jnp.stack([
          -focal / (.5 * width) * pts_world[..., 0] / pts_world[..., 2],
          -focal / (.5 * height) * pts_world[..., 1] / pts_world[..., 2],
          1. + 2. * near / pts_world[..., 2],
      ],
                          axis=-1)

      # Get NDC space rays.
      origins_ndc, directions_ndc = camera_utils.convert_to_ndc(
          origins, directions, pixtocam, near)

      # Ensure that the NDC space points lie on the calculated rays.
      directions_ndc_norm = jnp.linalg.norm(
          directions_ndc, axis=-1, keepdims=True)
      directions_ndc_unit = directions_ndc / directions_ndc_norm
      projection = ((pts_ndc - origins_ndc) * directions_ndc_unit).sum(axis=-1)
      pts_ndc_proj = origins_ndc + directions_ndc_unit * projection[..., None]

      # pts_ndc should be close to their projections pts_ndc_proj onto the rays.
      np.testing.assert_allclose(pts_ndc, pts_ndc_proj, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
