from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Union, Dict, Optional, Set

import numpy as np
import skimage as ski
from numpy import ndarray
from skimage.measure import regionprops
from sklearn.neighbors import NearestNeighbors

from seg2link import config
from seg2link.link_by_overlap import link2slices_return_seg
from seg2link.watersheds import dist_watershed

if config.debug:
    pass

@dataclass
class DivideMode:
    _2D: str = "2D"
    _2D_Link: str = "2D Link"
    _3D: str = "3D"


class NoDivisionError(Exception):
    pass


class NoLabelError(Exception):
    pass


def separate_one_label(seg_img3d: ndarray, label: int, threshold_area: int, mode: str, layer_from0: int) \
        -> Tuple[ndarray, ndarray, Tuple[slice, slice, slice], List[int]]:
    layer_num = layer_from0 if mode != DivideMode._3D else None
    sub_region, slice_subregion, pre_region = get_subregion(seg_img3d, label, layer_num)

    max_label = np.max(seg_img3d)
    segmented_subregion = segment_subregion(max_label, mode, pre_region, sub_region, threshold_area)

    divided_labels = np.unique(segmented_subregion)
    divided_labels = divided_labels[divided_labels > 0]
    if len(divided_labels) == 1:
        raise NoDivisionError

    subregion_old = seg_img3d[slice_subregion].copy()
    subregion_new = seg_img3d[slice_subregion].copy()

    subregion_new, labels = assign_new_labels(
        divided_labels, label, max_label, mode, pre_region, segmented_subregion, subregion_new)

    return subregion_old, subregion_new, slice_subregion, labels


def assign_new_labels(divided_labels, label, max_label, mode, pre_region, segmented_subregion, subregion_new):
    if mode == DivideMode._3D:
        segmented_subregion, smaller_labels = _suppress_largest_label(segmented_subregion)
        updated_regions = segmented_subregion > 0
        subregion_new[updated_regions] = segmented_subregion[updated_regions] + max_label
        labels = [label] + [label_ + max_label for label_ in smaller_labels]
    elif pre_region is None:
        mode = DivideMode._2D
    else:
        pass

    if mode == DivideMode._2D:
        updated_regions = segmented_subregion > 0
        subregion_new[updated_regions] = segmented_subregion[updated_regions] + max_label
        labels = [label_ + max_label for label_ in divided_labels]
    elif mode == DivideMode._2D_Link:
        updated_regions = segmented_subregion > 0
        subregion_new[updated_regions] = segmented_subregion[updated_regions]
        labels = np.unique(segmented_subregion)
        labels = labels[labels != 0].tolist()
        if label in labels:
            labels.remove(label)
            labels = [label] + labels
    else:
        pass

    return subregion_new, labels


def segment_subregion(max_label, mode, pre_region, sub_region, min_area):
    if mode == DivideMode._3D:
        segmented_subregion = separate_one_cell_3d(sub_region)
    elif mode == DivideMode._2D_Link:
        segmented_subregion = segment_link(sub_region, min_area, pre_region, max_label)
    elif mode == DivideMode._2D:
        segmented_subregion = segment_one_cell_2d_watershed(sub_region, min_area)
    else:
        raise ValueError
    return segmented_subregion


def segment_link(label_subregion: ndarray, min_area: int, pre_region: Optional[ndarray], max_label: int):
    seg = segment_one_cell_2d_watershed(label_subregion, min_area)
    if pre_region is None:
        return seg
    else:
        return link2slices_return_seg(pre_region, seg[..., 0], max_label, ratio_overlap=0.5)[..., np.newaxis]


def segment_one_cell_2d_watershed(labels_img3d: ndarray, min_area: int = 0) -> ndarray:
    """Input: a 3D array with shape (x,x,1). Segmentation is based on watershed"""
    result = np.zeros_like(labels_img3d, dtype=np.uint16)
    seg2d = dist_watershed(labels_img3d[..., 0], h=2)
    result[..., 0] = suppress_small_regions(seg2d, min_area)
    return result


def separate_one_label_r1(seg_img2d: ndarray, label: int, max_label: int) -> ndarray:
    sub_region, slice_subregion, _ = get_subregion_2d(seg_img2d, label)

    seg2d = dist_watershed(sub_region, h=2)
    segmented_subregion = suppress_small_regions(seg2d, 0)
    segmented_subregion, smaller_labels = _suppress_largest_label(segmented_subregion)

    smaller_regions = segmented_subregion > 0
    seg_img2d[slice_subregion][smaller_regions] = segmented_subregion[smaller_regions] + max_label
    return seg_img2d


def separate_one_cell_3d(sub_region: ndarray) -> ndarray:
    return ski.measure.label(sub_region, connectivity=3)

def _suppress_largest_label(seg: ndarray) -> Tuple[ndarray, List[int]]:
    """Revise the _labels: largest label (of area) -> 0; _labels>larest label: -= 1

    Examples
    --------
    >>> _suppress_largest_label(np.array([[1, 1, 0],
    ...                                   [1, 1, 2],
    ...                                   [0, 3, 2]]))
    (array([[0, 0, 0],
           [0, 0, 1],
           [0, 2, 1]]), [1, 2])
    """
    regions = regionprops(seg)
    max_idx = max(range(len(regions)), key=lambda k: regions[k].area)
    max_label = regions[max_idx].label
    seg[seg == max_label] = 0
    seg[seg > max_label] -= 1
    other_labels = [region.label for region in regions if region.label != max_label]
    other_labels_ = np.array(other_labels)
    other_labels_[other_labels_ > max_label] -= 1
    return seg, other_labels_.tolist()


class SortedRegion():
    def __init__(self, coords: Dict[int, ndarray], seg2d: ndarray):
        self.coords = coords  # OrderedDict: sorted coords from small region to big region
        self.seg2d = seg2d

    def removetiny(self):
        coords = iter(self.coords.items())
        label_ini, smallest_coord = next(coords)
        label_tgt = label_ini
        dist_min = np.inf
        for label, coord in coords:
            dist = self.min_dist_knn(smallest_coord, coord)
            if dist < dist_min:
                dist_min = dist
                label_tgt = label
        self.coords[label_tgt] = np.vstack((self.coords[label_tgt], self.coords.pop(label_ini)))
        self.seg2d[self.seg2d == label_ini] = label_tgt

    def min_dist_knn(self, coord1, coord2):
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(coord2)
        return neigh.kneighbors(X=coord1, return_distance=True)[0].min()


def suppress_small_regions(seg2d: ndarray, min_area_percentile: int) -> ndarray:
    """Merge regions with area < min_area with other regions
    """
    total_areas = np.count_nonzero(seg2d)
    threshold_area_ = total_areas * min_area_percentile / 100
    regions = regionprops(seg2d)
    num_tinyregions = int(np.sum([1 for r in regions if r.area < threshold_area_]))
    if num_tinyregions == 0:
        return seg2d
    sorted_idxes: List[int] = sorted(range(len(regions)), key=lambda k: regions[k].area)
    sorted_coords = OrderedDict({regions[k].label: regions[k].coords for k in sorted_idxes})
    sorted_regions = SortedRegion(sorted_coords, seg2d)
    for i in range(num_tinyregions):
        sorted_regions.removetiny()
    return sorted_regions.seg2d


def get_subregion(labels_img3d: ndarray, label: Union[int, Set[int]], layer_from0: Optional[int] = None) -> Tuple[
    ndarray, Tuple[slice, slice, slice], Optional[ndarray]]:
    """Get the subregion (bbox), the corresponding slice for the subregion, and the pre-slice subregion before
    the layer number
    """
    if layer_from0 is None:
        return get_subregion_3d(labels_img3d, label)
    else:
        return get_subregion2d_and_preslice(labels_img3d, label, layer_from0)

def get_subregion2d_and_preslice(labels_img3d: ndarray, label: Union[int, List[int]], layer_from0: int) \
        -> Tuple[ndarray, Tuple[slice, slice, slice], Optional[ndarray]]:
    subregion_2d, _, (x_max, x_min, y_max, y_min) = get_subregion_2d(labels_img3d[..., layer_from0], label)
    slice_subregion = np.s_[x_min:x_max + 1, y_min:y_max + 1, layer_from0:layer_from0 + 1]
    if layer_from0 == 0:
        pre_region_2d = None
    else:
        pre_region_2d = labels_img3d[x_min:x_max + 1, y_min:y_max + 1, layer_from0 - 1].copy()

    return subregion_2d[:, :, np.newaxis], slice_subregion, pre_region_2d


def get_subregion_3d(labels_img3d: ndarray, label: Union[int, Set[int]]) \
        -> Optional[Tuple[ndarray, Tuple[slice, slice, slice], None]]:
    labels = list(label) if isinstance(label, set) else label
    subregion = array_isin_labels_quick(labels, labels_img3d)
    x_max, x_min, y_max, y_min, z_max, z_min = bbox_3D_quick_v4(subregion)
    slice_subregion = np.s_[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
    return subregion[slice_subregion], slice_subregion, None


def array_isin_labels_quick(labels: Union[int, List[int]], labels_img: ndarray):
    # find the label 4582 (for delete): 0.7s (uniem_test1_600)
    # find the label 2864, 4785, 5184, 4582 (for merge): 5.1s (uniem_test1_600)
    if isinstance(labels, list):
        return np.isin(labels_img, labels).view(np.int8)
    else:
        return (labels_img == labels).view(np.int8)


def bbox_2D_quick_v2(img):
    r = np.any(img, axis=1)
    if not np.any(r):
        raise NoLabelError
    rmin, rmax = np.where(r)[0][[0, -1]]

    c = np.any(img[rmin:rmax + 1, :], axis=0)
    cmin, cmax = np.where(c)[0][[0, -1]]

    return rmax, rmin, cmax, cmin


def bbox_3D_quick_v4(img):
    """first compute along z axis"""
    z = np.any(img, axis=(0, 1))
    if not np.any(z):
        raise NoLabelError
    zmin, zmax = np.where(z)[0][[0, -1]]

    c = np.any(img[:, :, zmin:zmax + 1], axis=(0, 2))
    cmin, cmax = np.where(c)[0][[0, -1]]

    r = np.any(img[:, cmin:cmax + 1, zmin:zmax + 1], axis=(1, 2))
    rmin, rmax = np.where(r)[0][[0, -1]]
    return rmax, rmin, cmax, cmin, zmax, zmin


def get_subregion_2d(labels_img2d: ndarray, label: Union[int, List[int]]) -> Tuple[
    ndarray, Tuple[slice, slice], Tuple[int, int, int, int]]:
    """Get the subregion (bbox) and the corresponding slice for the subregion in a 2d label image
    """
    subregion = array_isin_labels_quick(label, labels_img2d)
    x_max, x_min, y_max, y_min = bbox_2D_quick_v2(subregion)
    slice_subregion = np.s_[x_min:x_max + 1, y_min:y_max + 1]
    return subregion[slice_subregion], slice_subregion, (x_max, x_min, y_max, y_min)


