from __future__ import annotations

import copy
import itertools
import os
import pickle
import re
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Optional, Union, Set, TYPE_CHECKING, Iterable

import numpy as np
import skimage as ski
from numpy import ndarray
from skimage.segmentation import relabel_sequential

from seg2link import parameters
from seg2link.link_by_overlap import link_previous_slices_round1, link_a_divided_label_round1
from seg2link.misc import make_folder, replace, mask_cells, flatten_2d_list, get_unused_labels_quick
from seg2link.watersheds import dist_watershed

if TYPE_CHECKING:
    from seg2link.seg2link_round1 import Seg2LinkR1

if parameters.DEBUG:
    from seg2link.parameters import lprofile

class Labels:
    """Labels are stored as a python list corresponding to the values in the segmented images"""

    __slots__ = ['emseg1', '_labels', 'ratio_overlap', '_label_nums']

    _labels: List[List[int]]
    _label_nums: List[int]

    def __init__(self, emseg1: "Seg2LinkR1", ratio_overlap: float = 0.5):
        self._labels = []
        self.emseg1 = emseg1
        self.ratio_overlap = ratio_overlap
        self._label_nums = []

    def __repr__(self):
        return "Current slice number: " + str(self.emseg1.current_slice) + ";   Current label number: " + str(self._cell_num)

    def reset(self):
        self._labels.clear()

    def cal_unused_labels(self) -> Set[int]:
        return set(get_unused_labels_quick(self.flatten()[0]))

    @property
    def unused_labels(self) -> str:
        return str(f"{self.cal_unused_labels()}, and {np.max(self.flatten()[0]) + 1}...")

    def rollback(self):
        if len(self._labels) > 0:
            labels = self.emseg1.archive.read_labels(self.emseg1.current_slice - 1)
            if isinstance(labels, Labels):
                self._labels = labels._labels
            else:
                self._labels = labels
            self.emseg1.current_slice -= 1

    def flatten(self) -> Tuple[List[int], List[int]]:
        return flatten_2d_list(self._labels)

    def delete(self, delete_list: Union[int, Set[int]]):
        """Delete a label (modify the value in self._labels to 0)"""
        labels1d, label_nums = self.flatten()
        labels1d_array = np.asarray(labels1d, dtype=int)
        labels1d_array = replace(delete_list, 0, labels1d_array)
        self._labels = self._to_labels2d(labels1d_array.tolist(), label_nums)

    def merge(self):
        """Merge the cells in the label_list and modify the transformation list"""
        labels1d, label_nums = self.flatten()
        labels1d_array = np.asarray(labels1d, dtype=int)

        target = min(self.emseg1.label_list)
        if not np.isin(target, labels1d_array):
            raise ValueError("Label ", target, " not exist")
        for label in self.emseg1.label_list:
            if label != target:
                labels1d_array = replace(label, target, labels1d_array)
        self._labels = self._to_labels2d(labels1d_array.tolist(), label_nums)


    @staticmethod
    def _to_labels2d(labels1d: List[int], label_nums: List[int]) -> List[List[int]]:
        cum_nums = list(itertools.accumulate([0] + label_nums))
        return [labels1d[cum_nums[i]:cum_nums[i + 1]] for i in range(len(label_nums))]

    @property
    def _cell_num(self) -> int:
        return len(set([item for sublist in self._labels for item in sublist]))

    @property
    def max_label(self) -> int:
        if len(self._labels) == 0:
            return 0
        else:
            return max([item for sublist in self._labels for item in sublist])

    def append_labels(self, initial_seg: Segmentation):
        _labels = np.unique(initial_seg.current_seg)
        labels = _labels[_labels != 0].tolist()
        self._labels.append(labels)

    def to_labels_img(self, layer: int, seg_img_cache: OrderedDict) -> ndarray:
        try:
            labels_pre_slice = np.asarray([0] + self._labels[layer - 1])
            seg_img = self.emseg1.archive.read_seg_img(seg_img_cache, layer)
            return labels_pre_slice[seg_img]
        except IndexError:
            raise IndexError(f"{labels_pre_slice.max()=}, {seg_img.max()=}, {layer=}")

    def get_seg_and_labels_tolink(self) \
            -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """Prepare the segmentations and label for linking"""
        seg_pre = self.to_labels_img(self.emseg1.current_slice - 1, self.emseg1.seg_img_cache)

        labels1d, self._label_nums = self.flatten()
        labels_pre_1d = np.asarray(labels1d)

        seg_post = self.emseg1.seg.current_seg.copy()
        seg_post[seg_post != 0] += max(labels1d)

        labels_post = np.unique(seg_post)
        labels_post = labels_post[labels_post != 0]

        return seg_pre, seg_post, labels_post, labels_pre_1d

    def to_multiple_labels(self, layers: slice) -> ndarray:
        """Get segmentation results (images) around current slice"""
        layer_num = layers.stop - layers.start
        h, w = self.emseg1.seg.current_seg.shape
        labels_img = np.zeros((layer_num, h, w), dtype=np.uint32)
        for i, z in enumerate(range(layers.start, layers.stop)):
            if (z + 1) <= self.emseg1.current_slice:
                labels_img[i, ...] = self.to_labels_img(z + 1, self.emseg1.seg_img_cache)
            else:
                break
        return labels_img.transpose((1, 2, 0))

    def link_or_append_labels(self):
        if self.emseg1.current_slice == 1:
            self.append_labels(self.emseg1.seg)
            return

        seg_pre, seg_post, list_post, list_pre_1d = self.get_seg_and_labels_tolink()
        list_pre_1d_linked, list_post_linked = link_previous_slices_round1(
            seg_pre, seg_post, list_pre_1d, list_post, self.ratio_overlap)
        self._labels = self._to_labels2d(list_pre_1d_linked, self._label_nums) + [list_post_linked]

    def relink_or_append_labels(self):
        labels_pre_now, labels_s2_now, labels_pre_past, seg_s1_past, seg_s2_now = \
            self.get_seg_and_labels_to_relink()
        list_pre_1d_linked, list_post_linked = link_a_divided_label_round1(
            labels_pre_now, labels_s2_now, labels_pre_past, seg_s1_past, seg_s2_now,
            self.emseg1.labels_divided, self.ratio_overlap)
        self._labels = self._to_labels2d(list_pre_1d_linked, self._label_nums) + [list_post_linked]

    def get_seg_and_labels_to_relink(self) -> Tuple[List[int], List[int], List[int], ndarray, ndarray]:
        """Prepare the segmentations and label for relinking"""
        current_labels_pre, _ = flatten_2d_list(self._labels[:-1])  # For linking by searching for same labels: (1)
        current_labels_s2 = copy.deepcopy(self._labels[-1])  # (1)
        current_seg_s2 = self.emseg1.seg.current_seg.copy()  # For linking by overlapping seg1 and seg2: (2)
        self.rollback()
        history_labels_pre, self._label_nums = self.flatten()  # (1) and (2)
        history_seg_s1 = self.to_labels_img(self.emseg1.current_slice, self.emseg1.seg_img_cache)  # (2)
        self.emseg1.current_slice += 1
        return current_labels_pre, current_labels_s2, history_labels_pre, history_seg_s1, current_seg_s2

    def relabel(self):
        """Relabel all N cells with label from 1 to N and save the current state"""
        labels1d, label_nums = self.flatten()
        labels1d_re = relabel_min_change(np.asarray(labels1d), labels1d[:-label_nums[-1]])
        self._labels = self._to_labels2d(labels1d_re.tolist(), label_nums)

    def relabel_deprecated(self):
        """Relabel all N cells with label from 1 to N and save the current state (use skimage.relabel_sequential)"""
        labels1d, label_nums = self.flatten()
        labels1d_re, fw, _ = relabel_sequential(np.asarray(labels1d))
        self._labels = self._to_labels2d(labels1d_re.tolist(), label_nums)


def relabel_min_change(labels_array: ndarray, used_labels_1d: Iterable) -> ndarray:
    """
    Relabel the new labels with unused labels
    """
    labels = np.unique(labels_array)
    max_label_used = np.max(used_labels_1d)
    labels_new = labels[labels > max_label_used]
    len_new = len(labels_new)

    if len_new == 0:
        return labels_array

    ori = labels_new.tolist()
    tgt = get_unused_labels_quick(used_labels_1d, len_new)

    labels_result = labels_array.copy()
    for i, j in zip(ori, tgt):
        labels_result[labels_array == i] = j
    return labels_result


class Segmentation:
    """Segment cells in each 2D slice"""
    __slots__ = ['enable_mask', 'cell_region', 'mask', 'ratio_mask', 'current_seg']

    def __init__(self, cell_region: ndarray, enable_mask: bool, mask: Optional[ndarray], ratio_mask: float):
        self.cell_region = cell_region
        self.enable_mask = enable_mask
        self.mask = mask
        self.ratio_mask = ratio_mask
        self.current_seg = np.array([], dtype=np.uint32)

    def watershed(self, layer_idx: int):
        """Segment a 2D label regions and save the result"""
        current_seg = dist_watershed(self.cell_region[..., layer_idx - 1].compute(),
                                     h=parameters.pars.h_watershed)
        if self.enable_mask:
            self.current_seg = mask_cells(current_seg, self.mask[..., layer_idx - 1].compute(), self.ratio_mask)
        else:
            self.current_seg = current_seg

    def reseg(self, label_img: ndarray, layer_idx: int):
        """Resegment based on the modified segmentation"""
        current_seg = ski.measure.label(label_img, connectivity=1)
        if self.enable_mask:
            self.current_seg = mask_cells(current_seg, self.mask[..., layer_idx - 1].compute(), self.ratio_mask)
        else:
            self.current_seg = current_seg


class Archive:
    def __init__(self, emseg1: "Seg2LinkR1", path_save: Path):
        self.emseg1 = emseg1
        self._path_labels = path_save / "History_labels"
        self._path_seg = path_save / "History_seg"

    def make_folders(self):
        self._path_labels = make_folder(self._path_labels)
        self._path_seg = make_folder(self._path_seg)

    def retrieve_history(self, target_slice: int, seg_img_cache: OrderedDict) -> Optional[Tuple[Labels, ndarray]]:
        latest_slice = self.latest_slice
        if latest_slice == 0:
            print("No label files found")
            return None

        print(f"The latest slice is {latest_slice}")
        if target_slice == 0:
            self.del_label_files(0, latest_slice)
            print("Restart from slice 1")
            return None
        return self.read_state(target_slice, latest_slice, seg_img_cache)

    @property
    def latest_slice(self) -> int:
        """Return the latest slice of the label stored in the hard disk"""
        if not self._path_labels.exists():
            return 0
        list_files = self.get_file_list(parameters.re_filename_v2)

        if len(list_files) == 0:
            list_files = self.get_file_list(parameters.re_filename_v1)

        regex_slice_nums = re.compile(r'\d+')
        list_slice_nums = [int(num) for fn in list_files for num in regex_slice_nums.findall(fn)]

        if not list_slice_nums:
            return 0
        return max(list_slice_nums)

    def get_file_list(self, re_filename):
        return list(filter(re.compile(re_filename).search, os.listdir(self._path_labels)))

    def archive_labels_and_seg2d(self):
        """Archive the label and segmented image"""
        self.save_labels_v2()
        self.save_seg_img()

    def save_labels_v2(self):
        """Save the labels"""
        labels = self.emseg1.labels
        if labels.emseg1.current_slice >= 1:
            with open(self._path_labels / (parameters.label_filename_v2 % labels.emseg1.current_slice), 'wb') as f:
                pickle.dump(labels._labels, f, pickle.HIGHEST_PROTOCOL)

    def save_seg_img(self):
        np.savez_compressed(self._path_seg / ('segmentation_slice%04i.npz' % self.emseg1.current_slice),
                            segmentation=self.emseg1.seg.current_seg)
        self.append_seg(self.emseg1.seg_img_cache, self.emseg1.seg.current_seg, self.emseg1.current_slice)

    @staticmethod
    def append_seg(seg_img_cache: OrderedDict, seg: ndarray, z: int):
        seg_img_cache[z] = seg
        if len(seg_img_cache) > parameters.pars.max_draw_layers_r1 // 2:
            print(f"pop {seg_img_cache.popitem(last=False)[0]}")
            print(f"len: {len(seg_img_cache)}")

    def read_state(self, slice_num: int, latest_slice: int, seg_img_cache: OrderedDict) -> Tuple[Labels, ndarray]:
        self.del_label_files(slice_num, latest_slice)
        label = self.read_labels(slice_num)
        seg_img = self.read_seg_img(seg_img_cache, slice_num)
        return label, seg_img

    def read_labels(self, slice_num: int) -> Optional[Union[List[List[int]], Labels]]:
        """Load a state of the label"""
        if slice_num <= 0:
            return None
        try:
            labels = self.load_labels_v2(slice_num)
        except FileNotFoundError:
            labels = self.load_labels_v1(slice_num)
            self.transform_v1_to_v2(slice_num)
        return labels

    def load_labels_v1(self, slice_num: int) -> Labels:
        with open(self._path_labels / (parameters.label_filename_v1 % slice_num), 'rb') as f:
            return pickle.load(f)

    def load_labels_v2(self, slice_num: int) -> List[List[int]]:
        with open(self._path_labels / (parameters.label_filename_v2 % slice_num), 'rb') as f:
            return pickle.load(f)

    def transform_v1_to_v2(self, slice_num: int):
        for i in range(1, slice_num + 1):
            with open(self._path_labels / (parameters.label_filename_v2 % i), 'wb') as f:
                pickle.dump(self.load_labels_v1(i)._labels, f, pickle.HIGHEST_PROTOCOL)

    def del_label_files(self, current_slice_num: int, latest_slice_num: int):
        for s in range(current_slice_num + 1, latest_slice_num + 1):
            try:
                os.remove(self._path_labels / (parameters.label_filename_v2 % s))
            except FileNotFoundError:
                pass

    def read_seg_img(self, seg_img_cache: OrderedDict, layer_idx: int) -> Optional[ndarray]:
        """Load a 2D segmentation result"""
        if layer_idx <= 0:
            return None
        if layer_idx not in seg_img_cache:
            seg = np.load(str(self._path_seg / ('segmentation_slice%04i.npz' % layer_idx)))["segmentation"]
            self.append_seg(seg_img_cache, seg, layer_idx)
        return seg_img_cache[layer_idx]
