## Install Seg2Link
- Install [Anaconda](https://www.anaconda.com/products/individual) 
  or [Miniconda](https://conda.io/miniconda.html)
- Create a new conda environment with a custom name, such as **seg2link-env**, and activate it by
running following commands in Anaconda PowerShell Prompt (Windows) or in terminal (macOS/Linux):
```console
$ conda create -n seg2link-env python=3.8 pip
$ conda activate seg2link-env
```
- Install Seg2Link (hosted on [PyPI]()):
```console
(seg2link-env) $ pip install seg2link
```

## Use Seg2Link
### Data preparation
Before performing segmentation, you must have at least the following data:

1. A 3D cell image saved in a folder as a set of 2D Tiff images.
2. A cellular/non-cellular prediction based on 1, saved in a different folder as 2D Tiff images.

### Launch Seg2Link
- Activate the created environment:
```console
$ conda activate seg2link-env
```
- Launch Seg2Link
```console
(seg2link-env) $ seg2link
```
![start_seg2link](./Round1/pictures/launch.png)
* Here we created and activated a custom environment called *seg2link* rather than *seg2link-env*.

### Choose a module
- Choose a proper module to perform:
    - the semi-automatic segmentation (in Module [Seg2D+Link](./Round1/start_r1.md))
    - or the comprehensive 3D inspection and corrections (in Module [3D correction](./Round2/start_r2.md))

![choose_round](./Round1/pictures/select_round.png)