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

"""Tests for utils."""

from absl.testing import absltest

from internal import utils


class UtilsTest(absltest.TestCase):

  def test_dummy_rays(self):
    """Ensures that the dummy Rays object is correctly initialized."""
    rays = utils.dummy_rays()
    self.assertEqual(rays.origins.shape[-1], 3)


if __name__ == '__main__':
  absltest.main()
