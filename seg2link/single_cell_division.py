from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, List, Union, Dict, Optional

import numpy as np
import skimage as ski
from numpy import ndarray
from scipy.spatial import KDTree
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential

from seg2link import parameters
from seg2link.cache_bbox import array_isin_labels_quick, NoLabelError
from seg2link.link_by_overlap import link_round2
from seg2link.misc import get_unused_labels_quick
from seg2link.watersheds import dist_watershed

if parameters.DEBUG:
    pass


@dataclass
class DivideMode:
    _2D: str = "2D"
    _2D_Link: str = "2D Link"
    _3D: str = "3D"


class NoDivisionError(Exception):
    pass


BBox2D = Tuple[slice, slice]
BBox = Tuple[slice, slice, slice]


def divide_link(label_subregion: ndarray, max_division: int, pre_region: Optional[ndarray], max_label: int):
    seg = segment_one_cell_2d_watershed(label_subregion, max_division)
    if pre_region is None:
        return seg
    else:
        return link_round2(pre_region, seg[..., 0], max_label, minimum_ratio_overlap=0.5)[..., np.newaxis]


def segment_one_cell_2d_watershed(labels_img3d: ndarray, max_division: int = 2) -> ndarray:
    """Input: a 3D array with shape (x,x,1). Segmentation is based on watershed"""
    result = np.zeros_like(labels_img3d, dtype=np.uint16)
    seg2d = dist_watershed(labels_img3d[..., 0], h=2)
    result[..., 0] = merge_tiny_labels(seg2d, max_division)
    return result


def separate_one_label_r1(seg_img2d: ndarray, selected_label: int, used_labels: List[int]) -> Tuple[ndarray, List[int]]:
    subarray_2d_bool, bbox_2d = get_subregion_2d(seg_img2d, selected_label)
    seg2d = dist_watershed(subarray_2d_bool, h=2)
    labels = np.unique(seg2d)
    labels_ = labels[labels != 0]
    if labels_.size == 1:
        raise NoDivisionError

    expected_labels = get_unused_labels_quick(used_labels, len(labels_))
    for label_ori, label_tgt in zip(labels_, expected_labels):
        seg_img2d[bbox_2d][seg2d == label_ori] = label_tgt
    return seg_img2d, expected_labels


def separate_one_cell_3d(sub_region: ndarray) -> ndarray:
    return ski.measure.label(sub_region, connectivity=3)


def suppress_largest_label(seg: ndarray) -> Tuple[ndarray, List[int]]:
    """Revise the _labels: largest label (of area): l = 0; _labels > largest label: l = l - 1"""
    regions = regionprops(seg)
    max_idx = max(range(len(regions)), key=lambda k: regions[k].area)
    max_label = regions[max_idx].label
    seg[seg == max_label] = 0
    seg[seg > max_label] -= 1
    other_labels = [region.label for region in regions if region.label != max_label]
    other_labels_ = np.array(other_labels)
    other_labels_[other_labels_ > max_label] -= 1
    return seg, other_labels_.tolist()


def merge_tiny_labels(seg2d: ndarray, max_division: int) -> ndarray:
    """Merge tiny regions with other regions
    """
    if max_division == "Inf":
        max_division = float("inf")
    regions = regionprops(seg2d)
    num_labels_remove = len(regions) - max_division if len(regions) > max_division else 0
    if num_labels_remove == 0:
        return seg2d

    sorted_idxes: List[int] = sorted(range(len(regions)), key=lambda k: regions[k].area)
    sorted_coords = OrderedDict({regions[k].label: regions[k].coords for k in sorted_idxes})
    sorted_regions = SortedRegion(sorted_coords, seg2d)
    for i in range(num_labels_remove):
        sorted_regions.remove_tiny()
    return relabel_sequential(sorted_regions.seg2d)[0]


class SortedRegion():
    def __init__(self, sorted_coords: Dict[int, ndarray], seg2d: ndarray):
        self.sorted_coords = sorted_coords  # OrderedDict: sorted coords from small region to big region
        self.seg2d = seg2d

    def remove_tiny(self):
        coords = iter(self.sorted_coords.items())
        label_ori, smallest_coord = next(coords)
        label_tgt = label_ori
        dist_min = np.inf
        for label, coord in coords:
            dist = min_dist_knn_scipy(smallest_coord, coord)
            if dist < dist_min:
                dist_min = dist
                label_tgt = label
        self.sorted_coords[label_tgt] = np.vstack((self.sorted_coords[label_tgt],
                                                   self.sorted_coords.pop(label_ori)))
        self.seg2d[self.seg2d == label_ori] = label_tgt


def min_dist_knn_scipy(coord1, coord2):
    tree = KDTree(coord2)
    return np.min(tree.query(coord1, k=1)[0])


def get_subregion2d_and_preslice(labels_img3d: ndarray, label: Union[int, List[int]], layer_from0: int) \
        -> Tuple[ndarray, BBox, Optional[ndarray]]:
    subregion_2d, bbox = get_subregion_2d(labels_img3d[..., layer_from0], label)
    slice_subregion = bbox[0], bbox[1], slice(layer_from0, layer_from0 + 1)
    if layer_from0 == 0:
        pre_region_2d = None
    else:
        pre_region_2d = labels_img3d[bbox[0], bbox[1], layer_from0 - 1].copy()

    return subregion_2d[:, :, np.newaxis], slice_subregion, pre_region_2d


def bbox_2D_quick(img_2d: ndarray) -> BBox2D:
    r = np.any(img_2d, axis=1)
    if not np.any(r):
        raise NoLabelError
    rmin, rmax = np.where(r)[0][[0, -1]]

    c = np.any(img_2d[rmin:rmax + 1, :], axis=0)
    cmin, cmax = np.where(c)[0][[0, -1]]

    return slice(rmin, rmax + 1), slice(cmin, cmax + 1)


def get_subregion_2d(labels_img2d: ndarray, label: Union[int, List[int]]) -> Tuple[
    ndarray, BBox2D]:
    """Get the subregion (bbox) and the corresponding slice for the subregion in a 2d label image
    """
    subregion_bool = array_isin_labels_quick(label, labels_img2d)
    bbox = bbox_2D_quick(subregion_bool)
    return subregion_bool[bbox], bbox
