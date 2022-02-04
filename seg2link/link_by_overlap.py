from typing import Tuple, List, Optional, Dict

import numpy as np
from numpy import ndarray

from seg2link import config

if config.debug:
    from seg2link.misc import qprofile
    from seg2link.config import lprofile


def link2slices_return_seg(seg_s1: ndarray, seg_s2: ndarray, max_label: int, ratio_overlap: float) -> ndarray:
    """EmSeg two images and return the modified list of label for the two images

    Notes
    -----
    This function is for the 2nd round. seg_s2's values should start from 1
    """
    seg_s2[seg_s2!=0] += max_label
    seg_s1, seg_s2, _, _ = match_by_overlap(seg_s1, seg_s2, ratio_overlap)
    return seg_s2


def _labels_areas(label_img: ndarray) -> Tuple[ndarray, ndarray]:
    labels, areas = np.unique(label_img, return_counts=True)
    return labels[labels!=0], areas[labels!=0]


def match_for_r1(seg_s1: ndarray, seg_s2: ndarray, ratio_overlap: float,
                 labels_pre: ndarray, labels_post: ndarray):
    seg_s1, seg_s2, targets_post, targets_pre = match_by_overlap(seg_s1, seg_s2, ratio_overlap)

    labels_pre = targets_pre[labels_pre]
    labels_post = targets_post[labels_post]
    return labels_pre.tolist(), labels_post.tolist()


def match_by_overlap(seg_s1, seg_s2, ratio_overlap: float):
    """Note: Any value of seg_s2 should be higher than values in seg_s1"""
    labels_pre_area = {label: area for label, area in zip(*_labels_areas(seg_s1))}
    labels_post_area = {label: area for label, area in zip(*_labels_areas(seg_s2))}
    labels_pre_target = {label: label for label in labels_pre_area.keys()}
    labels_post_target = {label: label for label in labels_post_area.keys()}
    for label_i_pre, area_i_pre in labels_pre_area.items():
        overlapped_areas = {label: area for label, area in zip(
            *_overlapped_labels(label_i_pre, seg_s1, seg_s2))}

        if not overlapped_areas:
            continue
        update_targets(labels_pre_target, labels_post_target, overlapped_areas, label_i_pre,
                       labels_pre_area, labels_post_area, ratio_overlap)
    targets_pre = np.arange(np.max(seg_s1) + 1)
    for label, target in labels_pre_target.items():
        targets_pre[label] = target
    seg_s1 = targets_pre[seg_s1]
    targets_post = np.arange(np.max(seg_s2) + 1)
    for label, target in labels_post_target.items():
        targets_post[label] = target
    seg_s2 = targets_post[seg_s2]
    return seg_s1, seg_s2, targets_post, targets_pre


def update_targets(labels_pre_target: Dict[int, int], labels_post_target: Dict[int, int],
                   overlapped_areas: Dict[int, int], label_i_pre: int, labels_pre_area: Dict[int, int],
                   labels_post_area: Dict[int, int], ratio_overlap: float):
    label_i_area = labels_pre_area[label_i_pre]
    for label_post, area_overlap in overlapped_areas.items():
        if area_overlap > ratio_overlap * min(label_i_area, labels_post_area[label_post]):
            target_post_ori = labels_post_target[label_post]
            target_new = update_target(target_post_ori, label_i_pre, labels_pre_area)
            labels_pre_target[label_i_pre] = target_new
            labels_post_target[label_post] = target_new


def update_target(target_post_ori: int, label_pre: int, labels_pre_area: Dict[int, int]):
    """choose a label with large area as target in case both pre and post label point to a pre label"""
    if target_post_ori in labels_pre_area and \
            labels_pre_area[target_post_ori] > labels_pre_area[label_pre]:
        return target_post_ori
    else:
        return label_pre


def _overlapped_labels(label_s1_i: int, seg_1: ndarray, seg_2: ndarray) -> Tuple[ndarray, ndarray]:
    labels_s2_overlap, areas_overlap = np.unique(seg_2[seg_1 == label_s1_i], return_counts=True)
    return _del0(labels_s2_overlap, areas_overlap)


def _del0(labels: ndarray, counts: ndarray) -> Tuple[ndarray, ndarray]:
    """Remove label with zero value and its count"""
    if len(labels) == 0:
        return labels, counts
    else:
        return labels[labels!=0], counts[labels!=0]
