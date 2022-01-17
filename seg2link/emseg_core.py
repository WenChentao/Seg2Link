from __future__ import annotations

import itertools
import os
import pickle
import re
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Optional, Union, Set

import numpy as np
import skimage as ski
from numpy import ndarray
from scipy import ndimage as ndi
from skimage.segmentation import relabel_sequential

import config
from link_by_overlap import link2slices
from misc import make_folder, replace
from watersheds import _dist_watershed


class Labels:
    """Labels are stored as a python list corresponding to the values in the segmented images"""

    __slots__ = ['archive', '_labels', '_voxels', 'current_slice', 'ratio_overlap', '_label_nums']

    _labels: List[List[int]]
    _label_nums: List[int]
    current_slice: int

    def __init__(self, archive: Archive, ratio_overlap: float = 0.5):
        self._labels = []
        self._voxels = {}  # Unused and will be removed in future
        self.current_slice = 0
        self.archive = archive
        self.ratio_overlap = ratio_overlap
        self._label_nums = []

    def __repr__(self):
        return "Current slice number: " + str(self.current_slice) + ";   Current label number: " + str(self._cell_num)

    def reset(self):
        self._labels.clear()
        self.current_slice = 0

    def rollback(self):
        if len(self._labels) > 0:
            labels = self.archive.read_labels(self.current_slice - 1)
            if isinstance(labels, Labels):
                self._labels = labels._labels
            else:
                self._labels = labels
            self.current_slice -= 1

    def _flatten(self) -> Tuple[List[int], List[int]]:
        labels1d = [item for sublist in self._labels for item in sublist]
        label_nums = [len(sublist) for sublist in self._labels]
        return labels1d, label_nums

    def delete(self, delete_list: Union[int, Set[int]]):
        """Delete a label (modify the value in self._labels to 0)"""
        labels1d, label_nums = self._flatten()
        labels1d_array = np.asarray(labels1d, dtype=int)
        labels1d_array = replace(delete_list, 0, labels1d_array)
        self._labels = self._to_labels2d(labels1d_array.tolist(), label_nums)

    def merge(self, merge_list: Set[int]):
        """Merge the cells in the label_list and modify the transformation list"""
        labels1d, label_nums = self._flatten()
        labels1d_array = np.asarray(labels1d, dtype=int)

        target = min(merge_list)
        if not np.isin(target, labels1d_array):
            raise ValueError("Label ", target, " not exist")
        for label in merge_list:
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
        self.current_slice += 1

    def to_labels_img(self, layer: int, seg_img_cache: OrderedDict) -> ndarray:
        _labels = np.asarray([0] + self._labels[layer - 1])
        seg_img = self.archive.read_seg_img(seg_img_cache, layer)
        return _labels[seg_img]

    def _get_images_tolink(self, initial_seg: Segmentation, seg_img_cache: OrderedDict) \
            -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """Prepare the segmentations and label for linking"""
        seg_pre = self.to_labels_img(self.current_slice, seg_img_cache)

        labels1d, self._label_nums = self._flatten()
        labels_pre_1d = np.asarray(labels1d)

        seg_post = initial_seg.current_seg.copy()
        seg_post[seg_post != 0] += max(labels1d)

        labels_post = np.unique(seg_post)
        labels_post = labels_post[labels_post != 0]

        return seg_pre, seg_post, labels_post, labels_pre_1d

    def _align_slice_post(self, seg_post: ndarray, cells: ndarray, align: Alignment, reset_align: bool,
                          should_align: bool) -> Tuple[Optional[ndarray], ndarray]:
        z = self.current_slice + 1
        if should_align:
            cells_aligned, seg_aligned = align.align_(cells[:, :, z - 2], cells[:, :, z - 1], seg_post, reset_align)
        else:
            cells_aligned = cells[:, :, z - 1]
            seg_aligned = seg_post
        return cells_aligned, seg_aligned

    def to_multiple_labels(self, layers: slice, initial_seg: Segmentation, seg_img_cache: OrderedDict) -> ndarray:
        """Get segmentation results (images) around current slice"""
        layer_num = layers.stop - layers.start
        h, w = initial_seg.current_seg.shape
        labels_img = np.zeros((layer_num, h, w), dtype=initial_seg.current_seg.dtype)
        for i, z in enumerate(range(layers.start, layers.stop)):
            if (z + 1) <= self.current_slice:
                labels_img[i, ...] = self.to_labels_img(z + 1, seg_img_cache)
            else:
                break
        return labels_img.transpose((1, 2, 0))

    def link_next_slice(self, initial_seg: Segmentation, align: Alignment, reset_align: bool,
                        seg_img_cache: OrderedDict, should_align: bool = True) -> ndarray:
        """Link current label with the segmentation in next slice"""
        if self.current_slice == 0:
            self.append_labels(initial_seg)
            return initial_seg.cell_region[..., 0]

        newseg_pre, newseg_post, list_post, list_pre_1d = self._get_images_tolink(initial_seg, seg_img_cache)
        cells_aligned, newseg_post_aligned = self._align_slice_post(newseg_post, initial_seg.cell_region,
                                                                    align, reset_align, should_align)
        list_pre_1d_linked, list_post_linked = link2slices(newseg_pre, newseg_post_aligned, list_pre_1d, list_post,
                                                           self.ratio_overlap)
        self._labels = self._to_labels2d(list_pre_1d_linked, self._label_nums) + [list_post_linked]
        self.current_slice += 1

        return np.array(cells_aligned, dtype=np.uint8)

    def relabel(self):
        """Relabel all N cells with label from 1 to N and save the current state"""
        labels1d, label_nums = self._flatten()
        max_label_pre = max(labels1d[:-label_nums[-1]])
        labels1d_re = self.relabel_min_change(np.asarray(labels1d), max_label_pre)
        self._labels = self._to_labels2d(labels1d_re.tolist(), label_nums)

    def relabel_deprecated(self):
        """Relabel all N cells with label from 1 to N and save the current state (use skimage.relabel_sequential)"""
        labels1d, label_nums = self._flatten()
        labels1d_re, fw, _ = relabel_sequential(np.asarray(labels1d))
        self._labels = self._to_labels2d(labels1d_re.tolist(), label_nums)

    @staticmethod
    def relabel_min_change(labels_array: ndarray, boundary: int) -> ndarray:
        """
        Return the labels_array and the area_dict with labels > boundary modified to
        a value <= boundary if possible
        """
        labels = np.unique(labels_array)
        labels_new = labels[labels > boundary]
        ori = []
        tgt = []
        len_new = len(labels_new)
        for label in range(1, boundary + 1):
            if label not in labels:
                if len_new == 0:
                    break
                else:
                    ori.append(labels_new[len_new - 1])
                    tgt.append(label)
                    len_new -= 1

        if len_new > 0:
            ori += labels_new[:len_new].tolist()
            tgt += list(range(boundary + 1, boundary + len_new + 1))

        labels_result = labels_array.copy()
        for i, j in zip(ori, tgt):
            labels_result[labels_array == i] = j
        return labels_result


class Segmentation:
    """Segment cells in each 2D slice"""
    __slots__ = ['cell_region', 'mask', 'ratio_mask', 'current_seg']

    def __init__(self, cell_region: ndarray, mask: Optional[ndarray], ratio_mask: float):
        self.cell_region = cell_region
        self.mask = mask
        self.ratio_mask = ratio_mask
        self.current_seg = np.array([], dtype=np.uint32)

    def watershed(self, layer_idx: int):
        """Segment a 2D label regions and save the result"""
        current_seg = _dist_watershed(self.cell_region[..., layer_idx - 1])
        self.current_seg = self._mask_cells(current_seg, self.mask, layer_idx, self.ratio_mask)

    def reseg(self, label_img: ndarray, layer_idx: int):
        """Resegment based on the modified segmentation"""
        current_seg = ski.measure.label(label_img, connectivity=1)
        self.current_seg = self._mask_cells(current_seg, self.mask, layer_idx, self.ratio_mask)

    @staticmethod
    def _mask_cells(label_img: ndarray, mask: Optional[ndarray], layer_idx: int, ratio_mask: float) -> ndarray:
        if mask is None:
            return label_img

        index = np.unique(label_img)
        ratio = np.array(ndi.mean(mask[..., layer_idx - 1], labels=label_img, index=index[1:]))

        mask_idx = np.zeros_like(index)
        idx_mask = list(np.where(ratio > ratio_mask)[0] + 1)
        for idx in idx_mask:
            mask_idx[idx] = idx
        return relabel_sequential(mask_idx[label_img])[0]


class Alignment:
    __slots__ = ['cells_fix', 'cells_move', 'aligned_cells', 'fitted_para', 'init_para']

    def __init__(self, should_align: bool):
        self.aligned_cells = None
        self.fitted_para = None
        if should_align:
            self.init_para = self._init_para()

    def align_(self, cells_fix: ndarray, cells_mov: ndarray, seg: ndarray, reset_align: bool):
        import itk
        cells_fix_, cells_move_, seg_ = self._transform_format(cells_fix, cells_mov, seg)
        if reset_align:
            self.aligned_cells, self.fitted_para = itk.elastix_registration_method(
                cells_fix_, cells_move_, parameter_object=self.init_para, log_to_console=False)
        aligned_seg = itk.transformix_filter(seg_, self.fitted_para, log_to_console=False)
        return self.aligned_cells, aligned_seg

    @staticmethod
    def _transform_format(*images: ndarray):
        import itk
        return [itk.image_view_from_array(img.astype(np.float32)) for img in images]

    @staticmethod
    def _init_para():
        import itk
        parameter_object = itk.ParameterObject.New()
        # Here used the multi-scale (=7) image pyramid to improve the robustness
        default_affine_parameter_map = parameter_object.GetDefaultParameterMap('affine', 7)
        # Final BSpline Interpolation Order: possible values are 0-5
        default_affine_parameter_map['FinalBSplineInterpolationOrder'] = ['0']
        default_affine_parameter_map['MaximumNumberOfIterations'] = ["500"]
        parameter_object.AddParameterMap(default_affine_parameter_map)
        return parameter_object


class Archive:
    def __init__(self, emseg, path_save: Path):
        self.emseg = emseg
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
        list_files = self.get_file_list(config.re_filename_v2)

        if len(list_files) == 0:
            list_files = self.get_file_list(config.re_filename_v1)

        regex_slice_nums = re.compile(r'\d+')
        list_slice_nums = [int(num) for fn in list_files for num in regex_slice_nums.findall(fn)]

        if not list_slice_nums:
            return 0
        return max(list_slice_nums)

    def get_file_list(self, re_filename):
        return list(filter(re.compile(re_filename).search, os.listdir(self._path_labels)))

    def archive_state(self):
        """Archive the label and segmented image"""
        self.save_labels_v2()
        self.save_seg_img()

    def save_labels_v2(self):
        """Save the labels"""
        labels = self.emseg.labels
        if labels.current_slice >= 1:
            with open(self._path_labels / (config.label_filename_v2 % labels.current_slice), 'wb') as f:
                pickle.dump(labels._labels, f, pickle.HIGHEST_PROTOCOL)

    def save_seg_img(self):
        np.savez_compressed(self._path_seg / ('segmentation_slice%04i.npz' % self.emseg.labels.current_slice),
                            segmentation=self.emseg.seg.current_seg)
        self.append_seg(self.emseg.seg_img_cache, self.emseg.seg.current_seg, self.emseg.labels.current_slice)

    @staticmethod
    def append_seg(seg_img_cache: OrderedDict, seg: ndarray, z: int):
        seg_img_cache[z] = seg
        if len(seg_img_cache) > config.max_draw_layers // 2:
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
            print("No label returned")
            return None
        try:
            labels = self.load_labels_v2(slice_num)
            print("Loaded labels (v2), type=", type(labels), "slice_num", slice_num)
            print("labels_len:", len(labels))
        except FileNotFoundError:
            labels = self.load_labels_v1(slice_num)
            print("Loaded labels (v1), type=", type(labels))
            self.transform_v1_to_v2(slice_num)
        return labels

    def load_labels_v1(self, slice_num: int) -> Labels:
        with open(self._path_labels / (config.label_filename_v1 % slice_num), 'rb') as f:
            return pickle.load(f)

    def load_labels_v2(self, slice_num: int) -> List[List[int]]:
        with open(self._path_labels / (config.label_filename_v2 % slice_num), 'rb') as f:
            return pickle.load(f)

    def transform_v1_to_v2(self, slice_num: int):
        for i in range(1, slice_num + 1):
            with open(self._path_labels / (config.label_filename_v2 % i), 'wb') as f:
                pickle.dump(self.load_labels_v1(i)._labels, f, pickle.HIGHEST_PROTOCOL)

    def del_label_files(self, current_slice_num: int, latest_slice_num: int):
        for s in range(current_slice_num + 1, latest_slice_num + 1):
            os.remove(self._path_labels / (config.label_filename_v2 % s))

    def read_seg_img(self, seg_img_cache: OrderedDict, layer_idx: int) -> Optional[ndarray]:
        """Load a 2D segmentation result"""
        if layer_idx <= 0:
            return None
        if layer_idx not in seg_img_cache:
            seg = np.load(str(self._path_seg / ('segmentation_slice%04i.npz' % layer_idx)))["segmentation"]
            self.append_seg(seg_img_cache, seg, layer_idx)
        return seg_img_cache[layer_idx]
