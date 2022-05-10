import copy
from typing import Tuple, Dict, List

import numpy as np
from numpy import ndarray
from skimage.segmentation import relabel_sequential

from seg2link import parameters

if parameters.DEBUG:
    pass


def link_round2(seg_s1: ndarray, seg_s2: ndarray, max_label: int, minimum_ratio_overlap: float) -> ndarray:
    """Match the segmentation in slice 2 with slice 1 and only return the modified slice 2

    Notes
    -----
    The segmentation of slice 1 will not be modified.
    The labels in s1 not corresponding to slice 1 will be assigned with values > max_label
    """
    seg_s2[seg_s2!=0] += max_label

    labels_and_area_s1 = {label1: area for label1, area in zip(*labels_and_areas(seg_s1))}
    labels_and_area_s2 = {label1: area for label1, area in zip(*labels_and_areas(seg_s2))}
    links_between_s1_and_s2 = extract_links_from_matching(seg_s1, seg_s2, labels_and_area_s1, labels_and_area_s2, minimum_ratio_overlap)

    original_and_transformed_labels_s2 = {label1: label1 for label1 in labels_and_area_s2.keys()}
    for label_i_in_s1, label_in_s2 in links_between_s1_and_s2:
        target_post_ori = original_and_transformed_labels_s2[label_in_s2]
        target_new = update_target(target_post_ori, label_i_in_s1, labels_and_area_s1)
        original_and_transformed_labels_s2[label_in_s2] = target_new

    targets_s2 = np.arange(np.max(seg_s2) + 1)
    for label, target in original_and_transformed_labels_s2.items():
        targets_s2[label] = target
    return targets_s2[seg_s2]


def link_previous_slices_round1(seg_s1: ndarray, seg_s2: ndarray, labels_s1: ndarray, labels_s2: ndarray,
                                minimum_ratio_overlap: float) -> Tuple[List[int], List[int]]:
    """Match the segmentation in slice 2 with slice 1 and return the modified label list in s1 and s2

    Notes
    -----
    labels_s1 is the flatten label list of all previous slices.
    The labels in s2 should have been modified to values higher than all labels in previous slices
    Note: Any value of seg_s2 should be higher than values in seg_s1
    """
    labels_and_area_s1 = {label1: area for label1, area in zip(*labels_and_areas(seg_s1))}
    labels_and_area_s2 = {label1: area for label1, area in zip(*labels_and_areas(seg_s2))}
    links_between_s1_and_s2 = extract_links_from_matching(seg_s1, seg_s2, labels_and_area_s1, labels_and_area_s2,
                                                          minimum_ratio_overlap)

    original_and_transformed_labels_s1 = {label1: label1 for label1 in labels_and_area_s1.keys()}
    original_and_transformed_labels_s2 = {label1: label1 for label1 in labels_and_area_s2.keys()}
    for label_i_in_s1, label_in_s2 in links_between_s1_and_s2:
        target_post_ori = original_and_transformed_labels_s2[label_in_s2]
        target_new = update_target(target_post_ori, label_i_in_s1, labels_and_area_s1)
        original_and_transformed_labels_s1[label_i_in_s1] = target_new
        original_and_transformed_labels_s2[label_in_s2] = target_new

    targets_s1 = np.arange(np.max(labels_s1) + 1)
    for label, target in original_and_transformed_labels_s1.items():
        targets_s1[label] = target
    targets_s2 = np.arange(np.max(seg_s2) + 1)
    for label, target in original_and_transformed_labels_s2.items():
        targets_s2[label] = target

    labels_s1 = targets_s1[labels_s1]
    labels_s2 = targets_s2[labels_s2]
    return labels_s1.tolist(), labels_s2.tolist()


def link_a_divided_label_round1(labels_pre_now: List[int], labels_s2_now: List[int], labels_pre_past: List[int],
                                seg_s1_past: ndarray, seg_s2_now: ndarray, labels_divided: List[int],
                                minimum_ratio_overlap: float) -> Tuple[List[int], List[int]]:
    seg_s2, _, _ = relabel_sequential(seg_s2_now)
    seg_s2[seg_s2 != 0] += max(labels_pre_past)
    labels_s2 = np.unique(seg_s2)
    labels_s2 = labels_s2[labels_s2 != 0]
    seg_s1 = seg_s1_past.view()
    labels_pre = labels_pre_past
    labels_and_area_s1 = {label1: area for label1, area in zip(*labels_and_areas(seg_s1))}
    labels_and_area_s2 = {label1: area for label1, area in zip(*labels_and_areas(seg_s2))}

    links_between_s1_and_s2 = link_by_overlapping(seg_s1,
                                                  seg_s2,
                                                  labels_s2,
                                                  labels_divided,
                                                  labels_s2_now,
                                                  labels_and_area_s1,
                                                  labels_and_area_s2,
                                                  minimum_ratio_overlap)
    links_between_s1_and_s2 = link_by_old_matching(labels_pre_now,
                                                   labels_s2_now,
                                                   labels_divided,
                                                   labels_pre_past,
                                                   labels_s2,
                                                   links_between_s1_and_s2)

    original_and_transformed_labels_s1 = {label1: label1 for label1 in labels_and_area_s1.keys()}
    original_and_transformed_labels_s2 = {label1: label1 for label1 in labels_and_area_s2.keys()}
    for label_i_in_s1, label_in_s2 in links_between_s1_and_s2:
        target_post_ori = original_and_transformed_labels_s2[label_in_s2]
        target_new = update_target(target_post_ori, label_i_in_s1, labels_and_area_s1)
        original_and_transformed_labels_s1[label_i_in_s1] = target_new
        original_and_transformed_labels_s2[label_in_s2] = target_new

    targets_s1 = np.arange(np.max(seg_s1) + 1)
    for label, target in original_and_transformed_labels_s1.items():
        targets_s1[label] = target
    targets_s2 = np.arange(np.max(seg_s2) + 1)
    for label, target in original_and_transformed_labels_s2.items():
        targets_s2[label] = target

    labels_pre = targets_s1[labels_pre]
    labels_s2 = targets_s2[labels_s2]
    return labels_pre.tolist(), labels_s2.tolist()


def link_by_overlapping(seg_s1: ndarray,
                        seg_s2: ndarray,
                        labels_s2: List[int],
                        labels_divided: List[int],
                        labels_s2_now: List[int],
                        labels_and_area_s1: Dict[int, int],
                        labels_and_area_s2: Dict[int, int],
                        minimum_ratio_overlap: float) -> List[Tuple[int, int]]:
    links_between_s1_and_s2: List[Tuple[int, int]] = []
    for label_i_in_s2_now in labels_divided:
        label_i_in_s2 = labels_s2[labels_s2_now.index(label_i_in_s2_now)]
        labels_s1_overlap_with_i_and_area = \
            {label: area for label, area in zip(*_overlapped_labels(label_i_in_s2, seg_s2, seg_s1))}
        if not labels_s1_overlap_with_i_and_area:
            continue
        label_i_area = labels_and_area_s2[label_i_in_s2]
        for label_in_s1, area_overlap_with_i in labels_s1_overlap_with_i_and_area.items():
            if area_overlap_with_i > minimum_ratio_overlap * min(label_i_area, labels_and_area_s1[label_in_s1]):
                links_between_s1_and_s2.append((label_in_s1, label_i_in_s2))
    return links_between_s1_and_s2


def link_by_old_matching(labels_pre_now: List[int], labels_s2_now: List[int], labels_divided: List[int],
                         labels_pre_past: List[int], labels_s2: List[int], links_between_s1_and_s2: List[Tuple[int, int]]):
    labels_s2_now_set = set(labels_s2_now)
    for label_in_s2 in labels_s2_now_set:
        if label_in_s2 not in labels_divided:
            labels_in_s1 = np.unique(np.array(labels_pre_past)[np.array(labels_pre_now) == label_in_s2])
            labels_in_s1 = labels_in_s1[labels_in_s1 != 0]
            labels_in_s2 = np.unique(np.array(labels_s2)[np.array(labels_s2_now) == label_in_s2])
            labels_in_s2 = labels_in_s2[labels_in_s2 != 0]
            if len(labels_in_s1) + len(labels_in_s2) <= 1:
                continue
            for l1 in labels_in_s1:
                links_between_s1_and_s2.append((l1, labels_in_s2[0]))
            for l2 in labels_in_s2:
                links_between_s1_and_s2.append((labels_in_s1[0], l2))
    return links_between_s1_and_s2


def labels_and_areas(label_img: ndarray) -> Tuple[ndarray, ndarray]:
    labels, areas = np.unique(label_img, return_counts=True)
    return labels[labels!=0], areas[labels!=0]


def _del0(labels: ndarray, counts: ndarray) -> Tuple[ndarray, ndarray]:
    """Remove label with zero value and its count"""
    if len(labels) == 0:
        return labels, counts
    else:
        return labels[labels != 0], counts[labels != 0]


def _overlapped_labels(label_s1_i: int, seg_1: ndarray, seg_2: ndarray) -> Tuple[ndarray, ndarray]:
    labels_s2_overlap, areas_overlap = np.unique(seg_2[seg_1 == label_s1_i], return_counts=True)
    return _del0(labels_s2_overlap, areas_overlap)


def extract_links_from_matching(seg_s1, seg_s2, labels_and_area_s1: Dict[int, int],
                                labels_and_area_s2: Dict[int, int], minimum_ratio_overlap: float) -> List[Tuple[int, int]]:
    links_between_s1_and_s2: List[Tuple[int, int]] = []
    for label_i_in_s1 in labels_and_area_s1.keys():
        labels_s2_overlap_with_i_and_area = {label: area for label, area in
                                             zip(*_overlapped_labels(label_i_in_s1, seg_s1, seg_s2))}
        if not labels_s2_overlap_with_i_and_area:
            continue
        label_i_area = labels_and_area_s1[label_i_in_s1]
        for label_in_s2, area_overlap_with_i in labels_s2_overlap_with_i_and_area.items():
            if area_overlap_with_i > minimum_ratio_overlap * min(label_i_area, labels_and_area_s2[label_in_s2]):
                links_between_s1_and_s2.append((label_i_in_s1, label_in_s2))
    return links_between_s1_and_s2


def update_target(target_post_ori: int, label_pre: int, labels_pre_area: Dict[int, int]):
    """choose a label with large area as target in case both pre and post label point to a pre label"""
    if target_post_ori in labels_pre_area and \
            labels_pre_area[target_post_ori] > labels_pre_area[label_pre]:
        return target_post_ori
    else:
        return label_pre

