# MultiNeRF: A Code Release for Mip-NeRF 360, Ref-NeRF, and Raw NeRF.

*This is not an officially supported Google product.*

This repository contains the code release for three CVPR 2022 papers:
[Mip-NeRF 360](https://jonbarron.info/mipnerf360/),
[Ref-NeRF](https://dorverbin.github.io/refnerf/index.html), and
[Raw NeRF](https://bmild.github.io/rawnerf/index.html).
This codebase was written by
integrating our internal implementions of Ref-NeRF and Raw NeRF into our
mip-NeRF 360 implementation. As such, this codebase should exactly
reproduce the results shown in mip-NeRF 360, but may differ slightly when
reproducing Ref-NeRF or Raw NeRF results.

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
of training iterations by whatever scale factor you decreased batch size.

## Citation
If you use this software package, please cite whichever constituent paper(s)
you build upon, or feel free to cite this entire codebase as:

```
@misc{multinerf2022,
      title={MultiNeRF: A Code Release for Mip-NeRF 360, Ref-NeRF, and Raw NeRF},
      author={Ben Mildenhall and Dor Verbin and Pratul P. Srinivasan and Peter Hedman and Ricardo Martin-Brualla and Jonathan T. Barron},
      year={2022},
}
```
