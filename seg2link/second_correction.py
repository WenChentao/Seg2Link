import os
import webbrowser
from pathlib import Path
from typing import Tuple, List, Optional, Set, Union

import numpy as np
from PyQt5.QtWidgets import QApplication
from napari.utils.colormaps import low_discrepancy_image
from numpy import ndarray

import config
from first_correction import Cache, VisualizeBase
from misc import print_information, replace
from new_windows_r2 import message_delete_labels
from single_cell_division import separate_one_label, get_subregion, NoLabelError, NoDivisionError
from widgets import WidgetsB

if config.debug:
    from config import qprofile, lprofile
    from memory_profiler import profile as mprofile

class Seg2LinkR2:
    """Segment the cells in 3D EM images"""

    def __init__(self, raw: ndarray, cell_region: ndarray, mask: ndarray, labels: ndarray, labels_path: Path):
        self.labels = labels
        self.divide_list = []
        self.label_list: Set[int] = set()
        self.s = slice(0, self.labels.shape[2])
        self.divide_subregion_slice = None
        self.labels_path = labels_path.parent / "seg-modified.npy"
        self.vis = VisualizeAll(self, raw, cell_region, mask)
        self.cache = CacheSubArray()
        self.update_info()
        self.keys_binding()
        self.vis.widgets.widget_binding()
        self.update_cmap()
        self.layer_selected = 0
        self.message_delete_labels = message_delete_labels

    def _update_segmentation(self):
        self.vis.update_segmentation_r2()

    def reset_labels(self, labels):
        self.labels = labels
        self.divide_list.clear()
        self.label_list.clear()
        self.s = slice(0, self.labels.shape[2])
        self._update_segmentation()
        self.cache = CacheSubArray()
        self.update_info()

    def reset_division_list(self):
        self.divide_list.clear()

    def merge(self, merge_list: Set[int]) -> Tuple[ndarray, ndarray, Tuple[slice, slice, slice]]:
        """Merge the cells in the label_list and modify the transformation list"""
        target = min(merge_list)
        merge_list_ = {label for label in merge_list if label != target}
        subarray_old, subarray_new, slice_ = self.subarray(merge_list_)

        subarray_new = replace(merge_list_, target, subarray_new)
        for label in merge_list_:
            if label in self.divide_list:
                self.divide_list.remove(label)
        return subarray_old, subarray_new, slice_

    def delete(self, delete_list: Union[int, Set[int]]):
        subarray_old, subarray_new, slice_ = self.subarray(delete_list)
        subarray_new = replace(delete_list, 0, subarray_new)
        delete_list = delete_list if isinstance(delete_list, list) else [delete_list]
        for label in delete_list:
            if label in self.divide_list:
                self.divide_list.remove(label)
        return subarray_old, subarray_new, slice_

    def subarray(self, label: Union[int, Set[int]]):
        _, slice_, _ = get_subregion(self.labels, label)
        subarray_old = self.labels[slice_].copy()
        subarray_new = self.labels[slice_].copy()
        return subarray_old, subarray_new, slice_

    def update_cmap(self):
        viewer_seg = self.vis.viewer.layers["segmentation"]
        viewer_seg._all_vals = low_discrepancy_image(np.arange(int(self.labels.max()) + 10), viewer_seg._seed)
        viewer_seg._all_vals[0] = 0

    def update_info(self, label_pre_division: Optional[int] = None):
        self.vis.update_widgets(label_pre_division)

    def update(self, subarray_old: ndarray, subarray_new:ndarray, slice_: Tuple[slice, slice, slice], action: str):
        self.labels[slice_] = subarray_new
        self._update_segmentation()
        self.cache.cache_state(subarray_old, subarray_new, slice_, action)
        self.update_info()

    def show_warning_delete_cells(self):
        self.message_delete_labels.width = 500
        self.message_delete_labels.height = 80
        self.message_delete_labels.show(run=True)
        self.message_delete_labels.info.value = \
            f"Please reduce cell number! (Current: {self.vis.widgets.label_max}, " \
            f"Limitation: {config.upper_limit_labels_r2})\n" \
            f"Do it by pressing [Sort labels and remove tiny cells] button"
        self.message_delete_labels.ok_button.changed.connect(self.hide_warning_delete_cells)

    def hide_warning_delete_cells(self):
        self.message_delete_labels.hide()

    def keys_binding(self):
        """Set the hotkeys for user's operations"""
        viewer_seg = self.vis.viewer.layers['segmentation']

        @viewer_seg.bind_key(config.key_add)
        @print_information("Add a label to be processed")
        def append_label_list(viewer_seg):
            """Add label to be merged into a list"""
            if viewer_seg.mode != "pick":
                print("\nPlease switch to pick mode in segmentation layer")
            elif viewer_seg.selected_label == 0:
                print("\nLabel 0 should not be processed!")
            else:
                self.label_list.add(viewer_seg.selected_label)
                print("Labels to be processed: ", self.label_list)
                self.update_info()

        @viewer_seg.bind_key(config.key_clean)
        @print_information("Clean the label list")
        def clear_label_list(viewer_seg):
            """Clear labels in the merged list"""
            self.label_list.clear()
            print(f"Cleaned the label list: {self.label_list}")
            self.update_info()

        @viewer_seg.bind_key(config.key_merge)
        @print_information("Merge labels")
        def _merge(viewer_seg):
            if not self.label_list:
                print("No labels were merged")
            else:
                subarray_old, subarray_new, slice_ = self.merge(self.label_list)
                self.label_list.clear()
                self.update(subarray_old, subarray_new, slice_, "Merge labels")

        @viewer_seg.bind_key(config.key_delete)
        @print_information("Delete the selected label(s)")
        def del_label(viewer_seg):
            """Delete the selected label"""
            if viewer_seg.mode != "pick":
                print("\nPlease switch to pick mode in segmentation layer")
            elif viewer_seg.selected_label == 0:
                print("\nLabel 0 cannot be deleted!")
            else:
                try:
                    delete_list = self.label_list if self.label_list else viewer_seg.selected_label
                    subarray_old, subarray_new, slice_ = self.delete(delete_list)
                    self.label_list.clear()
                    self.update(subarray_old, subarray_new, slice_, "Delete label(s)")
                    print(f"Label(s) {delete_list} were deleted")
                except NoLabelError:
                    print(f"Tried to delete label(s) but not found")

        @viewer_seg.bind_key(config.key_separate)
        @print_information("Separate")
        def separate_label(viewer_seg):
            if config.dtype_r2==np.uint16 and self.vis.widgets.label_max >= config.upper_limit_labels_r2:
                self.show_warning_delete_cells()
                return
            if viewer_seg.selected_label == 0:
                print("\nLabel 0 should not be separated!")
            else:
                self.layer_selected = self.vis.viewer.dims.current_step[-1]
                try:
                    subarray_old, subarray_new, slice_, divide_list = \
                        separate_one_label(self.labels,
                                           viewer_seg.selected_label,
                                           self.vis.widgets.threshold_area.value,
                                           mode=self.vis.widgets.divide_mode.value,
                                           layer_from0=self.layer_selected)
                except NoDivisionError:
                    self.vis.widgets.divide_msg.value = f"Label {viewer_seg.selected_label} was not separated"
                except NoLabelError:
                    self.vis.widgets.divide_msg.value = f"Label {viewer_seg.selected_label} was not found"
                else:
                    label_before_division = viewer_seg.selected_label
                    print(f"Label {label_before_division} was separated into {divide_list}")
                    self.divide_list = divide_list

                    self.labels[slice_] = subarray_new
                    self.update_cmap()
                    self._update_segmentation()
                    self.cache.cache_state(subarray_old, subarray_new, slice_, "Divide")
                    self.update_info(label_before_division)
                    self.divide_subregion_slice = slice_

                    self.vis.widgets.locate_label_divided()

        @viewer_seg.bind_key(config.key_undo)
        @print_information("Undo")
        def undo(viewer_seg):
            """Undo one keyboard command"""
            history = self.cache.load_cache(method="undo")
            if history is None:
                return None
            subarray_old, slice_, action = history
            self.labels[slice_] = subarray_old
            self.reset_division_list()
            self._update_segmentation()
            self.update_info()

        @viewer_seg.bind_key(config.key_redo)
        @print_information("Redo")
        def redo(viewer_seg):
            """Undo one keyboard command"""
            future = self.cache.load_cache(method="redo")
            if future is None:
                return None
            subarray_new, slice_, action = future
            self.labels[slice_] = subarray_new
            self.reset_division_list()
            self._update_segmentation()
            self.update_info()

        @viewer_seg.bind_key(config.key_switch_one_label_all_labels)
        @print_information("Switch showing one label/all labels")
        def switch_showing_one_or_all_labels(viewer_seg):
            """Show the selected label"""
            self.vis.viewer.layers["segmentation"].show_selected_label = \
                not self.vis.viewer.layers["segmentation"].show_selected_label

        @viewer_seg.bind_key(config.key_online_help)
        def help(viewer_seg):
            html_path = "file://" + os.path.abspath("../Help/help2.html")
            print(html_path)
            webbrowser.open(html_path)


class VisualizeAll(VisualizeBase):
    """Visualize the segmentation results"""

    def __init__(self, emseg2: Seg2LinkR2, raw: ndarray, cell_region: ndarray, cell_mask: ndarray):
        super().__init__(raw, cell_region, cell_mask)
        self.emseg2 = emseg2
        self.viewer.title = "Seg2link 2nd round"
        self.widgets = WidgetsB(self)
        self.show_segmentation_r2()

    def update_widgets(self, label_pre_division: int):
        self.widgets.update_info(label_pre_division)
        self.viewer.dims.set_axis_label(axis=2, label=f"Slice ({self.emseg2.s.start + 1}-{self.emseg2.s.stop})")

    def show_segmentation_r2(self):
        """show the segmentation results and other images/label"""
        if self.cell_mask is not None:
            self.viewer.layers['mask_cells'].data = self.cell_mask[..., self.emseg2.s]
        self.viewer.layers['raw_image'].data = self.raw[..., self.emseg2.s]
        if self.cell_region is not None:
            self.viewer.layers['cell_region'].data = self.cell_region[..., self.emseg2.s]
        self.viewer.layers['segmentation'].data = self.emseg2.labels
        QApplication.processEvents()

    def update_segmentation_r2(self):
        """Update the segmentation results and other images/label"""
        self.viewer.layers['segmentation'].data = self.emseg2.labels


class CacheR2(Cache):
    def undo(self):
        if not self.history:
            print("No earlier cached state!")
            return None
        else:
            self.future.append(self.history.pop())
            return self.future[-1]

    def redo(self):
        if not self.future:
            print("No later state!")
            return None
        else:
            self.history.append(self.future.pop())
            return self.history[-1]

    def reset_cache_b(self):
        self.history.clear()
        self.future.clear()


class CacheSubArray:
    def __init__(self):
        self.cache = CacheR2(maxlen=config.cache_length_second)

    def cache_state(self, subarray_old: ndarray, subarray_new: ndarray, slice_: Tuple[slice, slice, slice],
                    action: str):
        """Cache the previous & current states"""
        self.cache.append([subarray_old, subarray_new, slice_, action])

    def load_cache(self, method: str) -> Optional[Tuple[ndarray, Tuple[slice, slice, slice], str]]:
        """Load the cache"""
        if method == "undo":
            history = self.cache.undo()
            if history is None:
                return None
            else:
                subarray_old, _, slice_, action = history
                return subarray_old, slice_, action
        elif method == "redo":
            future = self.cache.redo()
            if future is None:
                return None
            else:
                _, subarray_new, slice_, action = future
                return subarray_new, slice_, action
        else:
            raise ValueError("Method must be 'undo' or 'redo'!")

    @property
    def cached_actions(self) -> List[str]:
        history = [hist[-1] + "\n" for hist in self.cache.history]
        future = [fut[-1] + "\n" for fut in self.cache.future][::-1]
        return history + [f"****(Head!)****\n"] + future
