import pickle
import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union, Set, TYPE_CHECKING

import numpy as np
from numpy import ndarray
from scipy import ndimage

from seg2link.misc import get_unused_labels_quick

if TYPE_CHECKING:
    from seg2link.seg2link_round2 import Seg2LinkR2

Bbox = Tuple[slice, slice, slice]


class CacheBbox:
    def __init__(self, emseg2: "Seg2LinkR2"):
        self.emseg2 = emseg2
        self.seg_shape = self.emseg2.labels.shape
        self.pad = (50, 50, 5)
        self.new_labels = set()
        self.bbox: Dict[int, Bbox] = {}
        self.load_or_generate_bbox(emseg2.labels_path)

    def load_or_generate_bbox(self, labels_path: Path):
        bbox_path = self.generate_bbox_path(labels_path)
        last_modi_time_labels = os.path.getmtime(str(labels_path))
        if bbox_path.exists():
            last_modi_time_bbox = os.path.getmtime(str(bbox_path))
            if last_modi_time_bbox > last_modi_time_labels:
                self.load_bbox(bbox_path)
                return
        self.refresh_bboxes()
        self._save_bbox(bbox_path)
        return

    def load_bbox(self, bbox_path: Path):
        with open(bbox_path, 'rb') as f:
            self.bbox = pickle.load(f)

    def cal_unused_labels(self) -> Set[int]:
        return set(get_unused_labels_quick(self.bbox.keys()))

    @property
    def unused_labels(self) -> str:
        return str(f"{self.cal_unused_labels()}, and {np.max(list(self.bbox.keys())) + 1}...")

    def insert_label(self):
        new_label = get_unused_labels_quick(self.bbox.keys(), max_num=1)[0]
        self.bbox[new_label] = slice(0, None), slice(0, None), slice(0, None)
        self.new_labels.add(new_label)
        return new_label

    def refresh_bboxes(self):
        print("Refresh the bbox information")
        self.emseg2.vis.widgets.show_state_info("Calculating bboxes for all labels... Please wait")
        _subregions = get_all_subregions_3d(self.emseg2.labels)
        self.bbox: Dict[int, Bbox] = {label + 1: bbox for label, bbox in enumerate(_subregions) if bbox is not None}
        self.emseg2.vis.widgets.show_state_info("Bboxes were calculated")

    def _save_bbox(self, bbox_path: Path):
        bbox_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bbox_path, 'wb') as f:
            pickle.dump(self.bbox, f, pickle.HIGHEST_PROTOCOL)

    def save_bbox(self, labels_path: Path):
        bbox_path = self.generate_bbox_path(labels_path)
        self._save_bbox(bbox_path)

    @staticmethod
    def generate_bbox_path(labels_path: Path):
        return labels_path.parent / "cache_bbox" / (labels_path.stem + ".pickle")

    def update_bbox_for_division(self, seg_subregion: ndarray, label_ori: int, divide_list: List[int], bbox_with_division: Bbox):
        """Add new divided labels, update changed labels, and delete the removed label after division"""
        bboxes_subregion_with_division = get_all_subregions_3d(seg_subregion)
        divide_set = set(divide_list).union({label_ori})
        for label in divide_set:
            if label == label_ori:
                # Update the bbox of the original label, or delete it if it no longer exists
                try:
                    self.bbox[label], _ = self.get_subregion_3d(label)
                except NoLabelError:
                    self.bbox.pop(label)
            else:
                # Get the bbox of a divided label in the subregion
                bbox_rel_div = bboxes_subregion_with_division[label - 1]
                bbox_abs_div = unzip_nested_box(bbox_with_division, bbox_rel_div)

                if not self.bbox.get(label):
                    # The bbox does not need merge if this is a new label
                    self.bbox[label] = bbox_abs_div
                else:
                    # The bbox needs merge if this is label already exists
                    self.bbox[label] = merge_bbox([self.bbox[label], bbox_abs_div])

    def get_subregion_3d(self, labels: Union[int, Set[int]]) \
            -> Optional[Tuple[Bbox, ndarray]]:
        """Used this to get cached bbox and update it if necessary"""
        labels: Union[int, List[int]]= list(labels) if isinstance(labels, set) else labels
        bbox_searched_in = self._get_bbox(labels)
        subarray_bool = array_isin_labels_quick(labels, self.emseg2.labels[bbox_searched_in])

        need_update = self.need_update_bboxes(subarray_bool,)
        if np.any(need_update):
            self.update_bboxes(labels, need_update)
            bbox_searched_in = self._get_bbox(labels)
            subarray_bool = array_isin_labels_quick(labels, self.emseg2.labels[bbox_searched_in])

        bbox_relative = bbox_3D_quick(subarray_bool)
        bbox_absolute = unzip_nested_box(bbox_searched_in, bbox_relative)
        return bbox_absolute, subarray_bool[bbox_relative]

    @staticmethod
    def need_update_bboxes(subarray_bool: ndarray) -> Tuple[bool,bool,bool,bool,bool,bool]:
        r0, r1 = np.any(subarray_bool[0,...]), np.any(subarray_bool[-1,...])
        c0, c1 = np.any(subarray_bool[:, 0, :]), np.any(subarray_bool[:, -1, :])
        z0, z1 = np.any(subarray_bool[..., 0]), np.any(subarray_bool[..., -1])
        return r0, r1, c0, c1, z0, z1

    def update_bboxes(self, labels: Union[int, List[int]], need_update: Tuple[bool,bool,bool,bool,bool,bool]):
        labels_ = labels if isinstance(labels, list) else [labels]
        for label in labels_:
            self.bbox_expand(label, need_update)

    def bbox_expand(self, label: int, need_update: Tuple[bool,bool,bool,bool,bool,bool]):
        up_r, up_c, up_z = np.any(need_update[:2]), np.any(need_update[2:4]), np.any(need_update[4:])
        bbox_searched_in = self._get_bbox(label)
        seg = self.emseg2.labels[bbox_searched_in]
        expand_r = (bbox_searched_in[0].start != 0 and np.any(seg[0, ...] == label)) or \
                   (bbox_searched_in[0].stop != self.seg_shape[0] and np.any(seg[-1, ...] == label))
        expand_c = (bbox_searched_in[1].start != 0 and np.any(seg[:, 0, :] == label)) or \
                   (bbox_searched_in[1].stop != self.seg_shape[1] and np.any(seg[:, -1, :] == label))
        expand_z = (bbox_searched_in[2].start != 0 and np.any(seg[..., 0] == label)) or \
                   (bbox_searched_in[2].stop != self.seg_shape[2] and np.any(seg[..., -1] == label))
        s_r = slice(0, None) if up_r and expand_r else bbox_searched_in[0]
        s_c = slice(0, None) if up_c and expand_c else bbox_searched_in[1]
        s_z = slice(0, None) if up_z and expand_z else bbox_searched_in[2]
        bbox_searched_in_ = (s_r, s_c, s_z)
        if expand_r or expand_c or expand_z:
            subarray_bool = array_isin_labels_quick(label, self.emseg2.labels[bbox_searched_in_])
            bbox_relative = bbox_3D_quick(subarray_bool)
            self.bbox[label] = unzip_nested_box(bbox_searched_in_, bbox_relative)

    def remove_bboxes(self, labels: Set[int]):
        for label in labels:
            try:
                self.bbox.pop(label)
            except KeyError:
                self.emseg2.vis.widgets.show_state_info(f"Warning: Bbox of label {label} was not cached")

    def set_bbox(self, label: int, bbox: Bbox):
        self.bbox[label] = bbox

    def _get_bbox(self, labels: Union[int, List[int]]):
        """Use this only to get cached bbox"""
        if isinstance(labels, list):
            return merge_bbox([self.get_bbox_padded(label) for label in labels])
        else:
            return self.get_bbox_padded(labels)

    def get_bbox_padded(self, label: int) -> Bbox:
        self.update_new_labels()
        result = self.bbox.get(label)
        if result is None:
            print(f"Label {label} was not found!")
            raise NoLabelError
        return self.pad_bbox(result)

    def update_new_labels(self):
        new_labels_ = self.new_labels.copy()
        for label in new_labels_:
            try:
                self.bbox[label] = bbox_3D_quick(array_isin_labels_quick(label, self.emseg2.labels))
            except NoLabelError:
                self.bbox.pop(label)
            self.new_labels.remove(label)

    def pad_bbox(self, bbox: Bbox) -> Bbox:
        x0, x1 = bbox[0].start, bbox[0].stop
        y0, y1 = bbox[1].start, bbox[1].stop
        z0, z1 = bbox[2].start, bbox[2].stop

        x_siz, y_siz, z_siz = self.seg_shape
        x_pad, y_pad, z_pad = self.pad

        x0_, x1_ = pad_range(x0, x1, x_pad, x_siz)
        y0_, y1_ = pad_range(y0, y1, y_pad, y_siz)
        z0_, z1_ = pad_range(z0, z1, z_pad, z_siz)
        return slice(x0_, x1_), slice(y0_, y1_), slice(z0_, z1_)


def pad_range(lower: int, upper: int, pad_size: int, max_range: int):
    lower_ = lower - pad_size if lower - pad_size >= 0 else 0
    upper_ = upper + pad_size if upper + pad_size <= max_range else max_range
    return lower_, upper_


def merge_bbox(bboxes: List[Bbox]):
    x0_ = min([bbox[0].start for bbox in bboxes])
    y0_ = min([bbox[1].start for bbox in bboxes])
    z0_ = min([bbox[2].start for bbox in bboxes])

    x1_ = max([bbox[0].stop for bbox in bboxes])
    y1_ = max([bbox[1].stop for bbox in bboxes])
    z1_ = max([bbox[2].stop for bbox in bboxes])
    return slice(x0_, x1_), slice(y0_, y1_), slice(z0_, z1_)


def unzip_nested_box(bbox_subregion: Bbox, bbox_relative: Bbox):
    x0, y0, z0 = bbox_subregion[0].start, bbox_subregion[1].start, bbox_subregion[2].start
    x_min, y_min, z_min = bbox_relative[0].start, bbox_relative[1].start, bbox_relative[2].start
    x_max, y_max, z_max = bbox_relative[0].stop, bbox_relative[1].stop, bbox_relative[2].stop
    slice_subregion = slice(x0 + x_min, x0 + x_max), \
                      slice(y0 + y_min, y0 + y_max), \
                      slice(z0 + z_min, z0 + z_max)
    return slice_subregion


def get_all_subregions_3d(labels_img3d: ndarray) -> List[Bbox]:
    """Return a list of np.s_, corresponding to labels 1, 2, ..., largest_label
    When a label i was not found, return None instead of np.s_"""
    return ndimage.find_objects(labels_img3d)


def bbox_3D_quick(img_3d: ndarray) -> Bbox:
    """first compute along z axis"""
    z = np.any(img_3d, axis=(0, 1))
    if not np.any(z):
        raise NoLabelError
    zmin, zmax = np.where(z)[0][[0, -1]]

    c = np.any(img_3d[:, :, zmin:zmax + 1], axis=(0, 2))
    cmin, cmax = np.where(c)[0][[0, -1]]

    r = np.any(img_3d[:, cmin:cmax + 1, zmin:zmax + 1], axis=(1, 2))
    rmin, rmax = np.where(r)[0][[0, -1]]
    return slice(rmin, rmax+1), slice(cmin, cmax+1), slice(zmin, zmax+1)


def array_isin_labels_quick(labels: Union[int, List[int]], labels_img: ndarray) -> ndarray:
    if isinstance(labels, list):
        return np.isin(labels_img, labels).view(np.int8)
    else:
        return (labels_img == labels).view(np.int8)


class NoLabelError(Exception):
    pass
