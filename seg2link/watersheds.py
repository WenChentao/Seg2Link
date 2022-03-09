from typing import Tuple, List, Optional

import numpy as np
from numpy import ndarray
from scipy import ndimage as ndi
from scipy.ndimage import measurements, grey_dilation
from skimage import measure
from skimage.filters import gaussian
from skimage.morphology import h_maxima, local_maxima
from skimage.segmentation import watershed, find_boundaries

from seg2link import config
if config.DEBUG:
    from seg2link.config import lprofile


def dist_watershed(cell_img2d: ndarray, h: int) -> ndarray:
    distance: ndarray = ndi.distance_transform_edt(cell_img2d)
    # Without this gaussian filtering, the h_maxima will generate multiple neighbouring maximum with the same distance,
    # and leading to over-segmentation
    distance_f = gaussian(distance, 1, preserve_range=True)
    maxima_filtered: ndarray = h_maxima(distance_f, h=h)
    seg_connectivity: ndarray = measure.label(cell_img2d, connectivity=1)
    maxima_final: ndarray = maxima_combine(distance_f, seg_connectivity, maxima_filtered)

    # use this structure so that neighbouring pixels with the same distance is assigned as the same marker
    markers: ndarray = ndi.label(maxima_final)[0]
    return watershed(-distance, markers=markers, mask=cell_img2d)


def _dist_watershed_3d(cell_img3d: ndarray):
    distance: ndarray = ndi.distance_transform_edt(cell_img3d, sampling = config.pars.scale_xyz)
    distance_f = gaussian(distance, 1, preserve_range=True)
    maxima_filtered: ndarray = h_maxima(distance_f, h=config.pars.h_watershed)
    seg_connectivity: ndarray = measure.label(cell_img3d, connectivity=1)
    maxima_final: ndarray = maxima_combine_3d(distance_f, seg_connectivity, maxima_filtered)
    markers: ndarray = ndi.label(maxima_final)[0]
    return watershed(-distance_f, markers=markers, mask=cell_img3d)


def maxima_combine(distance: ndarray, seg_connectivity: ndarray, maxima_filtered: ndarray) -> ndarray:
    """Combine following maxima points to avoid over-segmentation and loss of tiny regions:
    1. Maxima in each subregions based on connectivity (at least one point in each subregion will be kept)
    2. Maxima filtered by e.g. h_maxima (all kept)"""
    centers: List[Tuple[int, int]] = \
        measurements.maximum_position(distance, seg_connectivity, range(1, np.max(seg_connectivity)+1))
    maxima_h1: ndarray = local_maxima(distance)
    for pos_x, pos_y in centers:
        if maxima_h1[pos_x, pos_y]:
            maxima_filtered[pos_x, pos_y] = 1
    return maxima_filtered


def maxima_combine_3d(distance: ndarray, seg_connectivity: ndarray, maxima_filtered: ndarray) -> ndarray:
    centers: List[Tuple[int, int, int]] = \
        measurements.maximum_position(distance, seg_connectivity, range(1, np.max(seg_connectivity)+1))
    maxima_h1: ndarray = local_maxima(distance)
    for pos_x, pos_y, pos_z in centers:
        if maxima_h1[pos_x, pos_y, pos_z]:
            maxima_filtered[pos_x, pos_y, pos_z] = 1
    return maxima_filtered


def labels_with_boundary(labels: ndarray) -> ndarray:
    if config.pars.add_boundary_mode != "2D":
        result = find_boundaries(labels, mode="outer", connectivity=3)
    else:
        result = np.zeros_like(labels)
        for z in range(result.shape[2]):
            result[..., z] = find_boundaries(labels[..., z], mode="outer", connectivity=2)
    result = np.logical_not(result)
    return result * labels


def remove_boundary_scipy(labels: ndarray) -> ndarray:
    """Faster than using skimage"""
    labels_dilate = grey_dilation(labels, config.pars.labels_dilate_kernel_r2)
    labels_dilate *= (labels == 0)
    labels += labels_dilate
    return labels


