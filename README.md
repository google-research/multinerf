# MultiNeRF: A Code Release for Mip-NeRF 360, Ref-NeRF, and Raw-NeRF.

*This is not an officially supported Google product.*

This repository contains the code release for three CVPR 2022 papers:
[Mip-NeRF 360](https://bmild.github.io/rawnerf/index.html),
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
Contact [Jon Barron](https://jonbarron.info/) if you encounter any issues.

## Setup

```
# Clone the repo
git clone https://github.com/google-research/mipnerf.git; cd multinerf
conda create --name multinerf python=3.9
conda activate multinerf

# Prepare pip
conda install pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Manually install rmbrualla's `pycolmap` (don't use pip's! It's different)
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap
```

## Citation
If you use this software package, please cite whichever constituent paper(s)
you build upon, or feel free to cite this entire codebase as:

```
@misc{multinerf2022,
      title={MultiNeRF: A Code Release for Mip-NeRF 360, Ref-NeRF, and Raw-NeRF},
      author={Ben Mildenhall and Dor Verbin and Pratul P. Srinivasan and Peter Hedman and Ricardo Martin-Brualla and Jonathan T. Barron},
      year={2022},
}
```
