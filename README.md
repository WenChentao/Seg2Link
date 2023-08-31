[![PyPI](https://img.shields.io/pypi/v/seg2link)](https://pypi.org/project/seg2link/) [![GitHub](https://img.shields.io/github/license/WenChentao/3DeeCellTracker)](https://github.com/WenChentao/3DeeCellTracker/blob/master/LICENSE)

## Language

- [English](README.md) | [中文](README_zh.md) | [日本語](README_jp.md)

# ![icon](docs/pics/icon.svg)

**Seg2Link** is a [napari](https://napari.org
)-based software specifically designed for scientific research. 
The software aims to tackle a focused problem: offering an efficient toolbox for quick manual refinement of automated segmentation in large-scale 3D cellular images, particularly useful for brain images obtained through electron microscopy."

Our extensive [documentation](https://wenchentao.github.io/Seg2Link/) 
offers step-by-step tutorials, and our [academic paper](https://doi.org/10.1038/s41598-023-34232-6) delves into the scientific methodology and validation behind the software.

Unlike other segmentation solutions, Seg2Link requires pre-processed predictions of cell/non-cell regions as inputs. 
These predictions can conveniently be generated using [Seg2linkUnet2d](https://github.com/WenChentao/seg2link_unet2d) ([Documentation](https://wenchentao.github.io/Seg2Link/seg2link-unet2d.html)). This integrated approach makes the segmentation process both accurate and efficient.

## Features
- **Utilizing Deep Learning Predictions** -- Seg2Link takes deep learning predictions as input and refines initial inaccurate predictions into highly accurate results through semi-automatic user operations.
  
- **User-Friendly** -- Seg2Link not only auto-generates segmentation results but also allows for easy inspection and manual corrections through minimal mouse and keyboard interactions. It supports features like cell ordering, multiple-step undo and redo.

- **Efficiency** -- Seg2Link is engineered for the rapid processing of large 3D images with billions of voxels.
  
## Graphic Overview
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
