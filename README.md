[![PyPI](https://img.shields.io/pypi/v/seg2link)](https://pypi.org/project/seg2link/) [![GitHub](https://img.shields.io/github/license/WenChentao/3DeeCellTracker)](https://github.com/WenChentao/3DeeCellTracker/blob/master/LICENSE)

## Language

- [English](README.md) | [中文](README_zh.md)

# ![icon](docs/pics/icon.svg)

**Seg2Link** is a [napari](https://napari.org
)-based software program, designed to semi-automatically segment cells in 3D image stacks, especially for the brain 
images obtained by electron microscopy. 

You can read the [documentation](https://wenchentao.github.io/Seg2Link/) to learn how to use it and our [paper](https://doi.org/10.1038/s41598-023-34232-6) for more details.

Seg2Link requires predictions of cell/non-cell regions as inputs, which can be created with [Seg2linkUnet2d](https://github.com/WenChentao/seg2link_unet2d) ([Documentation](https://wenchentao.github.io/Seg2Link/seg2link-unet2d.html))

## Features
- **Utilize deep learning predictions** -- Seg2Link assist users to create to accurate segmentation results of individual cells from inaccurate cell/non-cell predictions .
- **Simplicity** -- Seg2Link generates segmentation automatically and allows for easy inspection and manual corrections.
- **Efficient** -- Seg2Link is designed for the fast processing of medium-sized 3D images with billions of voxels.
  
## Introduction
![Introduction](docs/pics/Introduction.png)

## Install
- Install [Anaconda](https://www.anaconda.com/products/individual) 
  or [Miniconda](https://conda.io/miniconda.html)
- Create a new conda environment and activate it by:
```console
$ conda create -n seg2link-env python=3.8 pip
$ conda activate seg2link-env
```
- Install seg2link:
```console
$ pip install seg2link
```
- Update to the latest version:
```console
$ pip install --upgrade seg2link
```

## Use the software
- Activate the created environment by:
```console
$ conda activate seg2link-env
```
- Start the software
```console
$ seg2link
```

## Citation
If you used this package in your research please cite it:

- Wen, C., Matsumoto, M., Sawada, M. et al. Seg2Link: an efficient and versatile solution for semi-automatic cell segmentation in 3D image stacks. _Sci Rep_ **13**, 7109 (2023). https://doi.org/10.1038/s41598-023-34232-6
