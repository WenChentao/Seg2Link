from collections import OrderedDict
from enum import Enum
from typing import Tuple, List, Union, Dict, Optional, Set

import numpy as np
import skimage as ski
from numpy import ndarray
from skimage.measure import regionprops
from sklearn.neighbors import NearestNeighbors

import config
from link_by_overlap import link2slices_return_seg
from config import qprofile

if config.debug:
    from config import qprofile, lprofile
    from memory_profiler import profile as mprofile

class DivideMode(Enum):
    _2D: str = "2D 1slice"
    _2D_Link: str = "2D multi-slices"
    _3D: str = "3D"


class NoDivisionError(Exception):
    pass


class NoLabelError(Exception):
    pass


@lprofile
def separate_one_label(seg_img3d: ndarray, label: int, threshold_area: int, mode: str, layer_from0: int) \
        -> Tuple[ndarray, ndarray, Tuple[slice, slice, slice], List[int]]:
    layer_num = layer_from0 if mode != DivideMode._3D else None
    sub_region, slice_subregion, pre_region = get_subregion(seg_img3d, label, layer_num)

    max_label = np.max(seg_img3d)
    if mode == DivideMode._3D:
        segmented_subregion = separate_subregion_3d(sub_region)
    elif mode == DivideMode._2D_Link:
        segmented_subregion = segment_link(sub_region, threshold_area, pre_region, max_label)
    elif mode == DivideMode._2D:
        segmented_subregion = segment_stack(sub_region, threshold_area)
    else:
        raise ValueError

    divided_labels = np.unique(segmented_subregion)
    divided_labels = divided_labels[divided_labels > 0]
    if len(divided_labels) == 1:
        raise NoDivisionError

    subregion_old = seg_img3d[slice_subregion].copy()
    subregion_new = seg_img3d[slice_subregion].copy()

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
        segmented_subregion, smaller_labels = _suppress_largest_label(segmented_subregion)
        updated_regions = segmented_subregion > 0
        subregion_new[updated_regions] = segmented_subregion[updated_regions] + max_label
        labels = [label] + [label_ + max_label for label_ in smaller_labels]

    return subregion_old, subregion_new, slice_subregion, labels


def segment_link(label_subregion: ndarray, threshold: int, pre_region: Optional[ndarray], max_label: int):
    seg = segment_stack(label_subregion, threshold)
    print("seg_shape", seg.shape)
    if pre_region is None:
        return seg
    else:
        return link2slices_return_seg(pre_region, seg[..., 0], max_label, ratio_overlap=0.5)[..., np.newaxis]


def segment_stack(labels_img3d: ndarray, threshold: int = 0) -> ndarray:
    """

    Examples
    --------
    >>> a = np.array([[[1, 1, 0, 0],
    ...               [1, 1, 0, 0],
    ...               [0, 0, 0, 1],
    ...               [0, 0, 1, 1]],
    ...              [[1, 1, 0, 0],
    ...               [1, 0, 0, 0],
    ...               [0, 0, 1, 1],
    ...               [0, 0, 1, 1]]]).transpose((1, 2, 0))
    >>> segment_stack(a).transpose((2,0,1))
    array([[[1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 2],
            [0, 0, 2, 2]],
    <BLANKLINE>
           [[1, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 2, 2],
            [0, 0, 2, 2]]], dtype=uint16)
    """
    result = np.zeros_like(labels_img3d, dtype=np.uint16)
    for z in range(labels_img3d.shape[2]):
        seg2d = ski.measure.label(labels_img3d[..., z], connectivity=1)
        result[..., z] = suppress_small_regions(seg2d, threshold)
    return result


def separate_one_slice_one_label(seg_img2d: ndarray, label: int, max_label: int) -> ndarray:
    sub_region, slice_subregion = get_subregion_from_2d(seg_img2d, label)

    seg2d = ski.measure.label(sub_region, connectivity=1)
    segmented_subregion = suppress_small_regions(seg2d, 0)
    segmented_subregion, smaller_labels = _suppress_largest_label(segmented_subregion)

    smaller_regions = segmented_subregion > 0
    seg_img2d[slice_subregion][smaller_regions] = segmented_subregion[smaller_regions] + max_label
    return seg_img2d


def separate_subregion_3d(sub_region: ndarray) -> ndarray:
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


def suppress_small_regions(seg2d: ndarray, threshold_area_percentile: int) -> ndarray:
    """Merge regions with area < threshold with other regions
    """
    total_areas = np.count_nonzero(seg2d)
    threshold_area_ = total_areas * threshold_area_percentile / 100
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
        return get_subregion_2d(labels_img3d, label, layer_from0)

def get_subregion_2d(labels_img3d: ndarray, label: Union[int, List[int]], layer_from0: int) \
        -> Tuple[ndarray, Tuple[slice, slice, slice], Optional[ndarray]]:
    subregion = array_isin_labels_quick(label, labels_img3d[..., layer_from0])
    x_max, x_min, y_max, y_min = bbox_2D_quick_v2(subregion)
    z_min = z_max = layer_from0
    slice_subregion = np.s_[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
    if layer_from0 == 0:
        pre_region_2d = None
    else:
        pre_region_2d = labels_img3d[x_min:x_max + 1, y_min:y_max + 1, layer_from0 - 1].copy()

    return subregion[x_min:x_max + 1, y_min:y_max + 1, np.newaxis], slice_subregion, pre_region_2d


def get_subregion_3d(labels_img3d: ndarray, label: Union[int, Set[int]]) \
        -> Optional[Tuple[ndarray, Tuple[slice, slice, slice], None]]:
    labels = list(label) if isinstance(label, set) else label
    subregion = array_isin_labels_quick(labels, labels_img3d)
    x_max, x_min, y_max, y_min, z_max, z_min = bbox_3D_quick_v2(subregion)
    slice_subregion = np.s_[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
    return subregion[slice_subregion], slice_subregion, None


def array_isin_labels(labels: Union[int, List[int]], labels_img: ndarray):
    """Deprecated as slow"""
    # find the label 4582 (for delete): 3.4s (uniem_test1_600)
    # find the label 2864, 4785, 5184, 4582 (for merge): 6.4s (uniem_test1_600)
    return np.isin(labels_img, labels).astype(np.int8)


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

def bbox_3D_quick_v2(img):
    """More efficient when the target cell is small"""
    r = np.any(img, axis=(1, 2))
    if not np.any(r):
        raise NoLabelError
    rmin, rmax = np.where(r)[0][[0, -1]]

    c = np.any(img[rmin:rmax + 1, :, :], axis=(0, 2))
    cmin, cmax = np.where(c)[0][[0, -1]]

    z = np.any(img[rmin:rmax + 1, cmin:cmax + 1, :], axis=(0, 1))
    zmin, zmax = np.where(z)[0][[0, -1]]
    return rmax, rmin, cmax, cmin, zmax, zmin


def bbox_3D_quick_v3(img):
    """More efficient when the target cell is large"""
    rmax, rmin = nonzero_min_max_1d(img, axis=0)
    cmax, cmin = nonzero_min_max_1d(img[rmin:rmax + 1, :, :], axis=1)
    zmax, zmin = nonzero_min_max_1d(img[rmin:rmax + 1, cmin:cmax + 1, :], axis=2)
    return rmax, rmin, cmax, cmin, zmax, zmin


def nonzero_min_max_1d(img, axis):
    min_ = None
    max_ = None
    for i in range(img.shape[axis]):
        if np.any(img.take(i, axis)):
            min_ = i
            break
    if min_ is None:
        raise NoLabelError
    for _i in range(img.shape[axis] - 1, -1, -1):
        if np.any(img.take(_i, axis)):
            max_ = _i
            break
    return max_, min_


def get_subregion_from_2d(labels_img2d: ndarray, label: Union[int, List[int]]) -> Tuple[
    ndarray, Tuple[slice, slice]]:
    """Get the subregion (bbox) and the corresponding slice for the subregion in a 2d label image
    """
    subregion = np.isin(labels_img2d, label).astype(np.int8)
    coordinates = np.where(subregion)
    x_max, x_min = np.max(coordinates[0]), np.min(coordinates[0])
    y_max, y_min = np.max(coordinates[1]), np.min(coordinates[1])
    slice_subregion = np.s_[x_min:x_max + 1, y_min:y_max + 1]
    return subregion[slice_subregion], slice_subregion


if __name__ == "__main__":
    import doctest

    doctest.testmod()
