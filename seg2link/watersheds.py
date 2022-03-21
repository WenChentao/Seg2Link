from typing import Tuple, List

import numpy as np
from numpy import ndarray
from scipy import ndimage as ndi
from scipy.ndimage import measurements
from skimage import measure
from skimage.filters import gaussian
from skimage.morphology import h_maxima, local_maxima
from skimage.segmentation import watershed

from seg2link import parameters
if parameters.DEBUG:
    pass


def dist_watershed(cell_img2d: ndarray, h: int) -> ndarray:
    distance_map: ndarray = ndi.distance_transform_edt(cell_img2d)
    # Without this gaussian filtering, the h_maxima will generate multiple neighbouring maximum with the same distance,
    # leading to over-segmentation
    distance_map_smooth = gaussian(distance_map, 1, preserve_range=True)
    h_maxima_of_distance_map: ndarray = h_maxima(distance_map_smooth, h=h)
    labels_by_connectivity: ndarray = measure.label(cell_img2d, connectivity=1)
    maxima_combined: ndarray = maxima_combine(h_maxima_of_distance_map, labels_by_connectivity, distance_map_smooth)
    markers_of_maxima: ndarray = ndi.label(maxima_combined)[0]
    return watershed(-distance_map, markers=markers_of_maxima, mask=cell_img2d)


def maxima_combine(h_maxima_of_distance: ndarray, labels_by_connectivity: ndarray, distance_map: ndarray) -> ndarray:
    """Combine following maxima points to avoid over-segmentation and loss of tiny regions:
    1. Maxima in each subregions based on connectivity (at least one point in each subregion will be kept)
    2. Maxima filtered by e.g. h_maxima (all kept)"""
    center_positions_of_labels: List[Tuple[int, int]] = measurements.maximum_position(
            distance_map, labels_by_connectivity, range(1, np.max(labels_by_connectivity) + 1)
        )
    local_maxima_of_distance: ndarray = local_maxima(distance_map)
    for pos_x, pos_y in center_positions_of_labels:
        if local_maxima_of_distance[pos_x, pos_y]:
            h_maxima_of_distance[pos_x, pos_y] = 1
    return h_maxima_of_distance


def maxima_combine_3d(distance: ndarray, seg_connectivity: ndarray, maxima_filtered: ndarray) -> ndarray:
    centers: List[Tuple[int, int, int]] = \
        measurements.maximum_position(distance, seg_connectivity, range(1, np.max(seg_connectivity)+1))
    maxima_h1: ndarray = local_maxima(distance)
    for pos_x, pos_y, pos_z in centers:
        if maxima_h1[pos_x, pos_y, pos_z]:
            maxima_filtered[pos_x, pos_y, pos_z] = 1
    return maxima_filtered


