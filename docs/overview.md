## Segment cells with Seg2Link and Deep learning

### Problem
Assume you have this 3D EM image dataset and want to segment it into individual cells.

![Raw-image](./pics/raw_0020.png){: style="width:350px"}

It is possible to manually annotate the cells in one or more slices one by one. However, when there are thousands of slices, manual annotation becomes impractical.

Modern machine learning and deep learning methods could produce automatic segmentation, but the results could contain a large number of errors that must be corrected manually.

### Solution
By using Seg2Link, you can quickly convert inaccurate deep learning or machine learning predictions to accurate segmentation results.

1. [Annotate a few subregions as cell/non cell manually](./seg2link-unet2d.md#2-annotate-cells):

    ![annotation](./pics/train_0020.png){: style="width:103px"} ![annotation](./pics/train_annotation_0020.png){: style="width:103px"} ![annotation](./pics/ellipsis.png){: style="width:103px"}

    *Typically, 20-30 subregions may be sufficient.*

2. With the annotated data, [train a deep neural network](./seg2link-unet2d.md#3-install-the-deep-learning-environment-in-local-pc) or other machine learning models.
3. [Predict the cell/non-cell regions](./seg2link-unet2d.md#6-predict-cell-regions) in the entire 3D image using the trained network:

    ![prediction](./pics/prediction_0020.png){: style="width:350px"}

4. Input your prediction into [Seg2Link](./quick_start.md). It will generate segmentations automatically and allow you to easily correct any errors:

    ![seg2link-gui](./pics/round2_0020.png)
