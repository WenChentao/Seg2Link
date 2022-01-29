from typing import Tuple, List, Optional, Dict

import numpy as np
from numpy import ndarray

from seg2link.misc import qprofile


def link2slices(seg_s1: ndarray, seg_s2: ndarray, labels_pre: ndarray, labels_post: ndarray,
                ratio_overlap: float) -> Tuple[List[int], List[int]]:
    """EmSeg two images and return the modified list of label for the two images

    Notes
    -----
    This function is for the 1st round. seg_s2's and labels_post's values should start from max_labels+1 in previous slices
    """
    areas_dict = {label: area for label, area in zip(*_labels_areas(seg_s1))}
    max_s1 = max(areas_dict)
    _overlap_match(seg_s1, seg_s2, ratio_overlap, areas_dict, max_s1, labels_pre, labels_post)
    _overlap_match(seg_s2, seg_s1, ratio_overlap, areas_dict, max_s1, labels_pre, labels_post)

    return labels_pre.tolist(), labels_post.tolist()

def link2slices_return_seg(seg_s1: ndarray, seg_s2: ndarray, max_label: int, ratio_overlap: float) -> ndarray:
    """EmSeg two images and return the modified list of label for the two images

    Notes
    -----
    This function is for the 2nd round. seg_s2's values should start from 1
    """
    areas_dict = {label: area for label, area in zip(*_labels_areas(seg_s1))}
    print("seg_s1",np.unique(seg_s1))
    print("seg_s2",np.unique(seg_s2))
    seg_s2[seg_s2!=0] += max_label
    max_s1 = max(areas_dict)
    _overlap_match(seg_s1, seg_s2, ratio_overlap, areas_dict, max_s1)
    print("seg_s2", np.unique(seg_s2))
    return seg_s2


def _labels_areas(label_img: ndarray) -> Tuple[ndarray, ndarray]:
    labels, areas = np.unique(label_img, return_counts=True)
    return labels[1:], areas[1:]

def _overlap_match(seg_s1: ndarray, seg_s2: ndarray, ratio_overlap: float, areas_dict: Dict[int, int], max_pre: int,
                   labels_pre: Optional[ndarray] = None, labels_post: Optional[ndarray] = None):
    labels, areas = _labels_areas(seg_s1)
    label_s1: int
    area_s1: int
    for label_s1, area_s1 in zip(labels, areas):
        labels_ins2, areas = _overlapped_labels(label_s1, seg_s1, seg_s2)

        if labels_ins2.size == 0:
            continue
        area_match, label_match, area_match_overlap = _best_match(areas, labels_ins2, seg_s2)

        if area_match_overlap < min(area_s1, area_match) * ratio_overlap:
            continue
        original, target = _merge_labels(label_match, label_s1, max_pre, areas_dict)

        if (labels_pre is None) or (labels_post is None):
            _replace(original, target, (seg_s1, seg_s2))
        else:
            _replace(original, target, (seg_s1, seg_s2, labels_pre, labels_post))


def _merge_labels(v1: int, v2: int, max_label_pre: int, areas_dict: Dict[int, int]) \
        -> Tuple[int, int]:
    """Return the label to be changed and the target label

    Notes
    -----
    To keep the previous labels not change as possible, when both the v1 and v2 are from the previous slice,
    the one with smaller areas (in the previous slice) is set to be target.
    On the other hand, if at least one of the them is from the posterior slice, the label with smaller value is
    set to be the target.
    """
    if v1 <= max_label_pre and v2 <= max_label_pre:
        if areas_dict[v1] < areas_dict[v2]:
            return v1, v2
        else:
            return v2, v1
    else:
        return max(v1, v2), min(v1, v2)


def _replace(original: int, target: int, arrays: Tuple[ndarray, ...]):
    for array in arrays:
        array[array == original] = target


def _best_match(areas: ndarray, labels_ins2: ndarray, seg_s2: ndarray) -> Tuple[int, int, int]:
    best_match_idx = np.argmax(areas)
    best_match_label = labels_ins2[best_match_idx]
    area_total = np.count_nonzero(seg_s2 == best_match_label)
    area_overlap = areas[best_match_idx]
    return area_total, best_match_label, area_overlap


def _overlapped_labels(label_i: int, seg_1: ndarray, seg_2: ndarray) -> Tuple[ndarray, ndarray]:
    labels_seg1_inseg2, counts = np.unique(seg_2[seg_1 == label_i], return_counts=True)
    labels_seg1_inseg2, counts = _del0(labels_seg1_inseg2, counts)
    return labels_seg1_inseg2, counts


def _del0(labels: ndarray, counts: ndarray) -> Tuple[ndarray, ndarray]:
    """Remove label with zero value and its count"""
    if len(labels) == 0:
        return labels, counts
    if labels[0] == 0:
        return labels[1:], counts[1:]
    else:
        return labels, counts
