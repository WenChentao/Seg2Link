import pickle
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union, Set

import numpy as np
from numpy import ndarray
from scipy import ndimage

Bbox = Tuple[slice, slice, slice]


class CacheBbox:
    def __init__(self, emseg2):
        self.emseg2 = emseg2
        self.seg_shape = self.emseg2.labels.shape
        bbox_path = self.get_bbox_path(emseg2.labels_path)
        self.bbox: Dict[int, Bbox] = {}
        if bbox_path.exists():
            self.load_bbox(emseg2.labels_path)
        else:
            self.refresh_bboxes()
            self.save_bbox(self.emseg2.labels_path)

    def refresh_bboxes(self):
        self.emseg2.vis.widgets.show_state_info("Calculating bboxes for all labels... Please wait")
        _subregions = get_all_subregions_3d(self.emseg2.labels)
        self.bbox: Dict[int, Bbox] = {label + 1: bbox for label, bbox in enumerate(_subregions) if bbox is not None}
        self.emseg2.vis.widgets.show_state_info("Bboxes were calculated")

    def save_bbox(self, labels_path: Path):
        bbox_path = self.get_bbox_path(labels_path)
        bbox_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bbox_path, 'wb') as f:
            pickle.dump(self.bbox, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def get_bbox_path(labels_path: Path):
        return labels_path.parent / "cache_bbox" / (labels_path.stem + ".pickle")

    def load_bbox(self, labels_path: Path):
        bbox_path = self.get_bbox_path(labels_path)
        with open(bbox_path, 'rb') as f:
            self.bbox = pickle.load(f)

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

    def update_bbox_for_division_3d(self, divide_list: List[int], bbox_with_division: Bbox):
        bboxes_subregion_with_division = get_all_subregions_3d(self.emseg2.labels[bbox_with_division])
        for label in divide_list:
            bbox_rel_div = bboxes_subregion_with_division[label - 1]
            self.bbox[label] = unzip_nested_box(bbox_with_division, bbox_rel_div)

    def get_subregion_3d(self, labels: Union[int, Set[int]]) \
            -> Optional[Tuple[Bbox, ndarray]]:
        labels = list(labels) if isinstance(labels, set) else labels
        bbox_searched_in = self.get_bbox(labels)
        subarray_bool = array_isin_labels_quick(labels, self.emseg2.labels[bbox_searched_in])
        bbox_relative = bbox_3D_quick(subarray_bool)
        bbox_absolute = unzip_nested_box(bbox_searched_in, bbox_relative)
        return bbox_absolute, subarray_bool[bbox_relative]

    def remove_bboxes(self, labels: Set[int]):
        for label in labels:
            try:
                self.bbox.pop(label)
            except KeyError:
                self.emseg2.vis.widgets.show_state_info(f"Warning: Bbox of label {label} was not cached")

    def set_bbox(self, label: int, bbox: Bbox):
        self.bbox[label] = bbox

    def get_bbox(self, labels: Union[int, List[int]]):
        if isinstance(labels, list):
            return merge_bbox([self.get_bbox_padded(label) for label in labels])
        else:
            return self.get_bbox_padded(labels)

    def get_bbox_padded(self, label: int) -> Bbox:
        result = self.bbox.get(label)
        if result is None:
            print(f"Label {label} was not found!")
            raise NoLabelError
        return self.pad_bbox(result)

    def pad_bbox(self, bbox: Bbox, pad: Tuple[int, int, int] = (50, 50, 5)) -> Bbox:
        x0, x1 = bbox[0].start, bbox[0].stop
        y0, y1 = bbox[1].start, bbox[1].stop
        z0, z1 = bbox[2].start, bbox[2].stop

        x_siz, y_siz, z_siz = self.seg_shape
        x_pad, y_pad, z_pad = pad

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
