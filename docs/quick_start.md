## Install Seg2Link
- Install [Anaconda](https://www.anaconda.com/products/individual) 
  or [Miniconda](https://conda.io/miniconda.html)
- Launch the Anaconda PowerShell Prompt in Windows or the terminal in macOS/Linux. Create a new conda environment with a custom name, such as seg2link-env, and activate it by running following commands:
```console
$ conda create -n seg2link-env python=3.8 pip
$ conda activate seg2link-env
```
- Install Seg2Link:
```console
(seg2link-env) $ pip install seg2link
```

## Use Seg2Link
### Data preparation
Before performing segmentation, you must have at least the following data:

1. A 3D cell image saved in a folder as a set of 2D Tiff images.
2. A cellular/non-cellular prediction based on 1, saved in a different folder as 2D Tiff images.

### Launch Seg2Link
- Activate the created environment and launch Seg2link:
```console
$ conda activate seg2link-env
(seg2link-env) $ seg2link
```

- Screenshots

    ![start_seg2link](./Round1/pictures/launch.png)
    
    Note I used an environment name *seg2link* rather than *seg2link-env*.

    ![choose_round](./Round1/pictures/select_round.png){: style="width:350px"}

    An initial interface is displayed.

### Choose a module
- Choose a proper module from the initial interface to perform:
    - the semi-automatic segmentation (in Module [Seg2D+Link](./Round1/start_r1.md))
    - or the comprehensive 3D inspection and corrections (in Module [3D correction](./Round2/start_r2.md))

