# MultiNeRF: A Code Release for Mip-NeRF 360, Ref-NeRF, and RawNeRF

*This is not an officially supported Google product.*

This repository contains the code release for three CVPR 2022 papers:
[Mip-NeRF 360](https://jonbarron.info/mipnerf360/),
[Ref-NeRF](https://dorverbin.github.io/refnerf/), and
[RawNeRF](https://bmild.github.io/rawnerf/).
This codebase was written by
integrating our internal implementions of Ref-NeRF and RawNeRF into our
mip-NeRF 360 implementation. As such, this codebase should exactly
reproduce the results shown in mip-NeRF 360, but may differ slightly when
reproducing Ref-NeRF or RawNeRF results.

This implementation is written in [JAX](https://github.com/google/jax), and
is a fork of [mip-NeRF](https://github.com/google/mipnerf).
This is research code, and should be treated accordingly.

## Setup

```
# Clone the repo.
git clone https://github.com/google-research/multinerf.git
cd multinerf

# Make a conda environment.
conda create --name multinerf python=3.9
conda activate multinerf

# Prepare pip.
conda install pip
pip install --upgrade pip

# Install requirements.
pip install -r requirements.txt

# Manually install rmbrualla's `pycolmap` (don't use pip's! It's different).
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap

# Confirm that all the unit tests pass.
./scripts/run_all_unit_tests.sh
```
You'll probably also need to update your JAX installation to support GPUs or TPUs.

## Running

Example scripts for training, evaluating, and rendering can be found in
`scripts/`. You'll need to change the paths to point to wherever the datasets
are located. [Gin](https://github.com/google/gin-config) configuration files
for our model and some ablations can be found in `configs/`.
After evaluating on the test set of each scene in one of the datasets, you can
use `scripts/generate_tables.ipynb` to produce error metrics across all scenes
in the same format as was used in tables in the paper.

### OOM errors

You may need to reduce the batch size (`Config.batch_size`) to avoid out of memory
errors. If you do this, but want to preserve quality, be sure to increase the number
of training iterations and decrease the learning rate by whatever scale factor you
decrease batch size by.

## Using your own data

### Running COLMAP to get camera poses

In order to run MultiNeRF on your own captured images of a scene, you must first run [COLMAP](https://colmap.github.io/install.html) to calculate camera poses. You can do this using our provided script `scripts/local_colmap_and_resize.sh`. Just make a directory `my_dataset_dir/` and copy your input images into a folder `my_dataset_dir/images/`, then run:
```
bash scripts/local_colmap_and_resize.sh my_dataset_dir
```
This will run COLMAP and create 2x, 4x, and 8x downsampled versions of your images. These lower resolution images can be used in NeRF by setting, e.g., the `Config.factor = 4` gin flag.

By default, `local_colmap_and_resize.sh` uses the OPENCV camera model, which is a perspective pinhole camera with k1, k2 radial and t1, t2 tangential distortion coefficients. To switch to another COLMAP camera model, for example OPENCV_FISHEYE, you can run
```
bash scripts/local_colmap_and_resize.sh my_dataset_dir OPENCV_FISHEYE
```

If you have a very large capture of more than around 500 images, we recommend switching from the exhaustive matcher to the vocabulary tree matcher in COLMAP (see the script for a commented-out example).

Our script is simply a thin wrapper for COLMAP--if you have run COLMAP yourself, all you need to do to load your scene in NeRF is ensure it has the following format:
```
my_dataset_dir/images/    <--- all input images
my_dataset_dir/sparse/0/  <--- COLMAP sparse reconstruction files (cameras, images, points)
```

### Writing a custom dataloader

If you already have poses for your own data, you may prefer to write your own custom dataloader.

MultiNeRF includes a variety of dataloaders, all of which inherit from the
base
[Dataset class](https://github.com/google-research/multinerf/blob/main/internal/datasets.py#L152).

The job of this class is to load all image and pose information from disk, then
create batches of ray and color data for training or rendering a NeRF model.

Any inherited subclass is responsible for loading images and camera poses from
disk by implementing the `_load_renderings` method (which is marked as
abstract by the decorator `@abc.abstractmethod`). This data is then used to
generate train and test batches of ray + color data for feeding through the NeRF
model. The ray parameters are calculated in `_make_ray_batch`.

#### Existing data loaders

To work from an example, you can see how this function is overloaded for the
different dataloaders we have already implemented:

-   [Blender](https://github.com/google-research/multinerf/blob/main/internal/datasets.py#L470)
-   [DTU dataset](https://github.com/google-research/multinerf/blob/main/internal/datasets.py#L793)
-   [Tanks and Temples](https://github.com/google-research/multinerf/blob/main/internal/datasets.py#L680),
    as processed by the NeRF++ paper
-   [Tanks and Temples](https://github.com/google-research/multinerf/blob/main/internal/datasets.py#L728),
    as processed by the Free View Synthesis paper

The main data loader we rely on is
[LLFF](https://github.com/google-research/multinerf/blob/main/internal/datasets.py#L526)
(named for historical reasons), which is the loader for a dataset that has been
posed by COLMAP.

#### Making your own loader by implementing `_load_renderings`

To make a new dataset, make a class inheriting from `Dataset` and overload the
`_load_renderings` method:

```
class MyNewDataset(Dataset):
  def _load_renderings(self, config):
    ...
```

In this function, you **must** set the following public attributes:

-   images
-   camtoworlds
-   pixtocams
-   height, width

Many of our dataset loaders also set other useful attributes, but these are the
critical ones for generating rays. You can see how they are used (along with a batch of pixel coordinates) to create rays in [`camera_utils.pixels_to_rays`](https://github.com/google-research/multinerf/blob/main/internal/camera_utils.py#L520).

**Images**

`images` = [N, height, width, 3] numpy array of RGB images. Currently we
require all images to have the same resolution.

**Extrinsic camera poses**

`camtoworlds` = [N, 3, 4] numpy array of extrinsic pose matrices.
`camtoworlds[i]` should be in **camera-to-world** format, such that we can run

```
pose = camtoworlds[i]
x_world = pose[:3, :3] @ x_camera + pose[:3, 3:4]
```

to convert a 3D camera space point `x_camera` into a world space point `x_world`.

These matrices must be stored in the **OpenGL** coordinate system convention for camera rotation:
x-axis to the right, y-axis upward, and z-axis backward along the camera's focal
axis.

The most common conventions are

-   `[right, up, backwards]`: OpenGL, NeRF, most graphics code.
-   `[right, down, forwards]`: OpenCV, COLMAP, most computer vision code.

Fortunately switching from OpenCV/COLMAP to NeRF is
[simple](https://github.com/google-research/multinerf/blob/main/internal/datasets.py#L108):
you just need to right-multiply the OpenCV pose matrices by `np.diag([1, -1, -1, 1])`,
which will flip the sign of the y-axis (from down to up) and z-axis (from
forwards to backwards):
```
camtoworlds_opengl = camtoworlds_opencv @ np.diag([1, -1, -1, 1])
```

You may also want to **scale** your camera pose translations such that they all
lie within the `[-1, 1]^3` cube for best performance with the default mipnerf360
config files.

We provide a useful helper function [`camera_utils.transform_poses_pca`](https://github.com/google-research/multinerf/blob/main/internal/camera_utils.py#L191) that computes a translation/rotation/scaling transform for the input poses that aligns the world space x-y plane with the ground (based on PCA) and scales the scene so that all input pose positions lie within `[-1, 1]^3`. (This function is applied by default when loading mip-NeRF 360 scenes with the LLFF data loader.) For a scene where this transformation has been applied, [`camera_utils.generate_ellipse_path`](https://github.com/google-research/multinerf/blob/main/internal/camera_utils.py#L230) can be used to generate a nice elliptical camera path for rendering videos.

**Intrinsic camera poses**

`pixtocams`= [N, 3, 4] numpy array of inverse intrinsic matrices, OR [3, 4]
numpy array of a single shared inverse intrinsic matrix. These should be in
**OpenCV** format, e.g.

```
camtopix = np.array([
  [focal,     0,  width/2],
  [    0, focal, height/2],
  [    0,     0,        1],
])
pixtocam = np.linalg.inv(camtopix)
```

Given a focal length and image size (and assuming a centered principal point,
this matrix can be created using
[`camera_utils.get_pixtocam`](https://github.com/google-research/multinerf/blob/main/internal/camera_utils.py#L411).

Alternatively, it can be created by using
[`camera_utils.intrinsic_matrix`](https://github.com/google-research/multinerf/blob/main/internal/camera_utils.py#L398)
and inverting the resulting matrix.

**Resolution**

`height` = int, height of images.

`width` = int, width of images.

**Distortion parameters (optional)**

`distortion_params` = dict, camera lens distortion model parameters. This
dictionary must map from strings -> floats, and the allowed keys are `['k1',
'k2', 'k3', 'k4', 'p1', 'p2']` (up to four radial coefficients and up to two
tangential coefficients). By default, this is set to the empty dictionary `{}`,
in which case undistortion is not run.

### Details of the inner workings of Dataset

The public interface mimics the behavior of a standard machine learning pipeline
dataset provider that can provide infinite batches of data to the
training/testing pipelines without exposing any details of how the batches are
loaded/created or how this is parallelized. Therefore, the initializer runs all
setup, including data loading from disk using `_load_renderings`, and begins
the thread using its parent start() method. After the initializer returns, the
caller can request batches of data straight away.

The internal `self._queue` is initialized as `queue.Queue(3)`, so the infinite
loop in `run()` will block on the call `self._queue.put(self._next_fn())` once
there are 3 elements. The main thread training job runs in a loop that pops 1
element at a time off the front of the queue. The Dataset thread's `run()` loop
will populate the queue with 3 elements, then wait until a batch has been
removed and push one more onto the end.

This repeats indefinitely until the main thread's training loop completes
(typically hundreds of thousands of iterations), then the main thread will exit
and the Dataset thread will automatically be killed since it is a daemon.


## Citation
If you use this software package, please cite whichever constituent paper(s)
you build upon, or feel free to cite this entire codebase as:

```
@misc{multinerf2022,
      title={MultiNeRF: A Code Release for Mip-NeRF 360, Ref-NeRF, and RawNeRF},
      author={Ben Mildenhall and Dor Verbin and Pratul P. Srinivasan and Peter Hedman and Ricardo Martin-Brualla and Jonathan T. Barron},
      year={2022},
}
```
