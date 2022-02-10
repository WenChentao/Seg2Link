from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, List, Union, Dict, Optional, Set

import numpy as np
import skimage as ski
from numpy import ndarray
from scipy.spatial import KDTree
from skimage.measure import regionprops

from seg2link.cache_bbox import array_isin_labels_quick, NoLabelError
from seg2link import config
from seg2link.link_by_overlap import match_return_seg_img
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


def segment_link(label_subregion: ndarray, max_division: int, pre_region: Optional[ndarray], max_label: int):
    seg = segment_one_cell_2d_watershed(label_subregion, max_division)
    if pre_region is None:
        return seg
    else:
        return match_return_seg_img(pre_region, seg[..., 0], max_label, ratio_overlap=0.5)[..., np.newaxis]


def segment_one_cell_2d_watershed(labels_img3d: ndarray, max_division: int = 2) -> ndarray:
    """Input: a 3D array with shape (x,x,1). Segmentation is based on watershed"""
    result = np.zeros_like(labels_img3d, dtype=np.uint16)
    seg2d = dist_watershed(labels_img3d[..., 0], h=2)
    result[..., 0] = suppress_small_regions(seg2d, max_division)
    return result


def separate_one_label_r1(seg_img2d: ndarray, label: int, max_label: int) -> ndarray:
    sub_region, slice_subregion, _ = get_subregion_2d(seg_img2d, label)

    seg2d = dist_watershed(sub_region, h=2)
    segmented_subregion, smaller_labels = _suppress_largest_label(seg2d)

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

    def remove_tiny(self):
        coords = iter(self.coords.items())
        label_ini, smallest_coord = next(coords)
        label_tgt = label_ini
        dist_min = np.inf
        for label, coord in coords:
            dist = min_dist_knn_scipy(smallest_coord, coord)
            if dist < dist_min:
                dist_min = dist
                label_tgt = label
        self.coords[label_tgt] = np.vstack((self.coords[label_tgt], self.coords.pop(label_ini)))
        self.seg2d[self.seg2d == label_ini] = label_tgt


def min_dist_knn_scipy(coord1, coord2):
    tree = KDTree(coord2)
    return np.min(tree.query(coord1, k=1)[0])


def suppress_small_regions(seg2d: ndarray, max_division: int) -> ndarray:
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
    return sorted_regions.seg2d


def get_subregion2d_and_preslice(labels_img3d: ndarray, label: Union[int, List[int]], layer_from0: int) \
        -> Tuple[ndarray, Tuple[slice, slice, slice], Optional[ndarray]]:
    subregion_2d, _, (x_max, x_min, y_max, y_min) = get_subregion_2d(labels_img3d[..., layer_from0], label)
    slice_subregion = np.s_[x_min:x_max + 1, y_min:y_max + 1, layer_from0:layer_from0 + 1]
    if layer_from0 == 0:
        pre_region_2d = None
    else:
        pre_region_2d = labels_img3d[x_min:x_max + 1, y_min:y_max + 1, layer_from0 - 1].copy()

    return subregion_2d[:, :, np.newaxis], slice_subregion, pre_region_2d


def bbox_2D_quick(img_2d: ndarray) -> Tuple[int, int, int, int]:
    r = np.any(img_2d, axis=1)
    if not np.any(r):
        raise NoLabelError
    rmin, rmax = np.where(r)[0][[0, -1]]

    c = np.any(img_2d[rmin:rmax + 1, :], axis=0)
    cmin, cmax = np.where(c)[0][[0, -1]]

    return rmax, rmin, cmax, cmin


def get_subregion_2d(labels_img2d: ndarray, label: Union[int, List[int]]) -> Tuple[
    ndarray, Tuple[slice, slice], Tuple[int, int, int, int]]:
    """Get the subregion (bbox) and the corresponding slice for the subregion in a 2d label image
    """
    subregion = array_isin_labels_quick(label, labels_img2d)
    x_max, x_min, y_max, y_min = bbox_2D_quick(subregion)
    slice_subregion = np.s_[x_min:x_max + 1, y_min:y_max + 1]
    return subregion[slice_subregion], slice_subregion, (x_max, x_min, y_max, y_min)




