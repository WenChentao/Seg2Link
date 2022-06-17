# ![icon](pic/icon.svg)

**Seg2Link** is a software designed to semi-automatically segment 3D cell images, especially EM images. 

Check out this [**Online Guide**]() to learn how to install and use it.

## Features
- Accept *imperfect* **cell/non-cell prediction** by machine learning as input.
- Generate **automatic segmentation** in each slice.
- Allows *easy* **inspection and manual corrections**. 
- Allows *multiple-steps* of **undo/redo**
- Can *efficiently* process **3D images with billions of voxels**.
  
## Workflow
```mermaid
  flowchart TB
  A[Raw Image]-->|DNN prediction|B1[Cell/NonCell]

  subgraph ide1 [Round #1: Seg2D+Link]
    B1-->|Segment|B2[2D SEG. in layer 1]
    B2-->|Segment+Link|C[Linked 2D SEG. in layer i]
    B2-->|Manual correction|B2
    C-->|Manual correction + <br> Seg+Link next layer|C
    C-->D[Linked 2D SEG. in All layers]
  end

  subgraph ide2 [Round #2: 3D correction]
    D-->|Export/Import|E[3D SEG.]
    E-->|Manual <br/> correction|F[Completed SEG]
    F-->|Export|G[TIFF image sequence]
  end
  A-->|Other software|E
```

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
$ pip install --update seg2link
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
