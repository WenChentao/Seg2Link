import webbrowser
from collections import namedtuple
from pathlib import Path
from typing import Tuple, List, Optional, Set, Union, NamedTuple

import numpy as np
from PyQt5.QtWidgets import QApplication
from napari.utils.colormaps import low_discrepancy_image
from numpy import ndarray

from seg2link.seg2dlink_core import relabel_min_change
from seg2link._tests_r2 import test_merge_r2, test_delete_r2, test_divide_r2
from seg2link import parameters
from seg2link.seg2link_round1 import Cache, VisualizeBase
from seg2link.misc import print_information, replace, get_unused_labels_quick
from seg2link.message_windows_round2 import message_delete_labels
from seg2link.cache_bbox import NoLabelError, CacheBbox, merge_bbox, Bbox
from seg2link.widgets_round2 import WidgetsR2
from seg2link.single_cell_division import DivideMode, get_subregion2d_and_preslice, NoDivisionError, \
    separate_one_cell_3d, divide_link, segment_one_cell_2d_watershed, suppress_largest_label

if parameters.DEBUG:
    from seg2link.parameters import lprofile

BiState = namedtuple("BiState", ["array", "bboxes"])


class StateR2(NamedTuple):
    old: BiState
    new: BiState
    bbox: Bbox
    action: str


class Seg2LinkR2:
    """Segment the cells in 3D EM images"""
    # TODO: The action insert now is not supported by undo/redo and this will be fixed in next version.
    def __init__(self, raw: ndarray, cell_region: ndarray, mask: ndarray, labels: ndarray, labels_path: Path):
        self.labels = labels
        self.divide_list = []
        self.label_list: Set[int] = set()
        self.s = slice(0, self.labels.shape[2])
        self.divide_subregion_slice = None
        self.labels_path = labels_path
        self.vis = VisualizeAll(self, raw, cell_region, mask)
        self.cache_bbox = CacheBbox(self)
        self.cache = CacheSubArray(self)
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
        self.cache = CacheSubArray(self)
        self.update_info()

    def reset_division_list(self):
        self.divide_list.clear()

    def merge(self, merge_list: Set[int]) -> Tuple[ndarray, ndarray, Tuple[slice, slice, slice], int, Bbox]:
        """Merge the cells in the label_list and modify the transformation list"""
        target = min(merge_list)
        merge_list_ = {label for label in merge_list if label != target}
        subarray_old, subarray_new, bbox_not_target = self.subarray(merge_list_)
        bbox_target, _ = self.cache_bbox.get_subregion_3d(target)
        bbox_updated_target = merge_bbox([bbox_not_target, bbox_target])

        subarray_new = replace(merge_list_, target, subarray_new)
        for label in merge_list_:
            if label in self.divide_list:
                self.divide_list.remove(label)
        return subarray_old, subarray_new, bbox_not_target, target, bbox_updated_target

    def delete(self, delete_list: Union[int, Set[int]]):
        subarray_old, subarray_new, slice_ = self.subarray(delete_list)
        subarray_new = replace(delete_list, 0, subarray_new)
        delete_list = delete_list if isinstance(delete_list, list) else [delete_list]
        for label in delete_list:
            if label in self.divide_list:
                self.divide_list.remove(label)
        return subarray_old, subarray_new, slice_

    def subarray(self, label: Union[int, Set[int]]):
        slice_, _ = self.cache_bbox.get_subregion_3d(label)
        subarray_old = self.labels[slice_].copy()
        subarray_new = self.labels[slice_].copy()
        return subarray_old, subarray_new, slice_

    def update_cmap(self):
        viewer_seg = self.vis.viewer.layers["segmentation"]
        viewer_seg._all_vals = low_discrepancy_image(
            np.arange(int(max(self.cache_bbox.bbox)) + 10), viewer_seg._seed
        )
        viewer_seg._all_vals[0] = 0

    def update_info(self, label_pre_division: Optional[int] = None):
        self.vis.update_widgets(label_pre_division)

    def update(self, state: StateR2, update_cmap: bool=False, label_pre_division: Optional[int] = None):
        self.labels[state.bbox] = state.new.array
        if update_cmap:
            self.update_cmap()
        self._update_segmentation()
        self.cache.cache_state(state)
        self.update_info(label_pre_division)

    def show_warning_delete_cells(self):
        self.message_delete_labels.width = 500
        self.message_delete_labels.height = 80
        self.message_delete_labels.show(run=True)
        self.message_delete_labels.info.value = \
            f"Please reduce cell number! (Current: {self.vis.widgets.label_max}, " \
            f"Limitation: {parameters.pars.upper_limit_labels_r2})\n" \
            f"Do it by pressing [Sort labels and remove tiny cells] button"
        self.message_delete_labels.ok_button.changed.connect(self.hide_warning_delete_cells)

    def hide_warning_delete_cells(self):
        self.message_delete_labels.hide()

    def keys_binding(self):
        """Set the hotkeys for user's operations"""
        viewer_seg = self.vis.viewer.layers['segmentation']

        @viewer_seg.bind_key(parameters.pars.key_add)
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

        @viewer_seg.bind_key(parameters.pars.key_clean)
        @print_information("Clean the label list")
        def clear_label_list(viewer_seg):
            """Clear labels in the merged list"""
            self.label_list.clear()
            print(f"Cleaned the label list: {self.label_list}")
            self.update_info()

        @viewer_seg.bind_key(parameters.pars.key_merge)
        @print_information("Merge labels")
        @test_merge_r2(self)
        def _merge(viewer_seg):
            if not self.label_list:
                self.vis.widgets.show_state_info("No label was selected")
            elif len(self.label_list)==1:
                self.vis.widgets.show_state_info("Only one label was selected")
            else:
                self.vis.widgets.show_state_info("Merging... Please wait")
                subarray_old, subarray_new, slice_not_target, target, slice_updated_target = \
                    self.merge(self.label_list)
                # Update bbox cache
                old_state = BiState(subarray_old, self.cache_bbox.bbox.copy())
                self.cache_bbox.remove_bboxes(self.label_list)
                self.label_list.clear()
                self.cache_bbox.set_bbox(label=target, bbox=slice_updated_target)
                new_state = BiState(subarray_new, self.cache_bbox.bbox.copy())
                # Update information
                state = StateR2(old_state, new_state, slice_not_target, "Merge labels")
                self.update(state)
                self.vis.widgets.show_state_info("Multiple labels were merged")

        @viewer_seg.bind_key(parameters.pars.key_delete)
        @print_information("Delete the selected label(s)")
        @test_delete_r2(self)
        def del_label(viewer_seg):
            """Delete the selected label"""
            if viewer_seg.mode != "pick":
                self.vis.widgets.show_state_info("Warning: Switch to pick mode in segmentation layer!")
            elif viewer_seg.selected_label == 0 and not self.label_list:
                self.vis.widgets.show_state_info("Label 0 cannot be deleted!")
            else:
                try:
                    self.vis.widgets.show_state_info("Deleting... Please wait")
                    delete_list = self.label_list if self.label_list else viewer_seg.selected_label
                    subarray_old, subarray_new, slice_ = self.delete(delete_list)
                    # Update bbox cache
                    old_state = BiState(subarray_old, self.cache_bbox.bbox.copy())
                    if self.label_list:
                        self.cache_bbox.remove_bboxes(self.label_list)
                    else:
                        self.cache_bbox.remove_bboxes({viewer_seg.selected_label})
                    self.label_list.clear()
                    new_state = BiState(subarray_new, self.cache_bbox.bbox.copy())
                    # Update information
                    state = StateR2(old_state, new_state, slice_, "Delete label(s)")
                    self.update(state)
                    self.vis.widgets.show_state_info(f"Label(s) {delete_list} were deleted")
                except NoLabelError:
                    self.vis.widgets.show_state_info(f"Tried to delete label(s) but not found")

        @viewer_seg.bind_key(parameters.pars.key_separate)
        @print_information("Divide")
        @test_divide_r2(self)
        def separate_label(viewer_seg):
            mode = self.vis.widgets.divide_mode.value
            separate_label_3_modes(viewer_seg, mode)

        def separate_label_3_modes(viewer_seg, mode):
            viewer_seg.mode = "pick"
            if parameters.pars.dtype_r2==np.uint16 and self.vis.widgets.label_max >= parameters.pars.upper_limit_labels_r2:
                self.show_warning_delete_cells()
                return
            if viewer_seg.selected_label == 0:
                self.vis.widgets.show_state_info("Label 0 should not be separated!")
            else:
                self.layer_selected = self.vis.viewer.dims.current_step[-1]
                try:
                    self.vis.widgets.show_state_info("Dividing... Please wait")
                    subarray_old, subarray_new, bbox_divided, labels_post = divide_and_relabel(mode)
                except NoDivisionError:
                    self.vis.widgets.show_state_info("")
                    self.vis.widgets.divide_msg.value = f"Label {viewer_seg.selected_label} was not separated"
                except NoLabelError:
                    self.vis.widgets.show_state_info("")
                    self.vis.widgets.divide_msg.value = f"Label {viewer_seg.selected_label} was not found"
                else:
                    label_ori = viewer_seg.selected_label
                    self.vis.widgets.show_state_info(
                        f"Label {label_ori} was separated into {short_str(labels_post)}")
                    self.divide_list = labels_post
                    # Update Bbox cache
                    old_state = BiState(subarray_old, self.cache_bbox.bbox.copy())
                    self.cache_bbox.update_bbox_for_division(subarray_new, label_ori, labels_post, bbox_divided)
                    new_state = BiState(subarray_new, self.cache_bbox.bbox.copy())

                    state = StateR2(old_state, new_state, bbox_divided, "Divide")
                    self.update(state, update_cmap=True, label_pre_division=label_ori)
                    self.divide_subregion_slice = bbox_divided

                    self.vis.widgets.locate_label_divided()

        @viewer_seg.bind_key(parameters.pars.key_separate_link)
        @print_information("Divide-Relink")
        def separate_relink_label(viewer_seg):
            separate_label_3_modes(viewer_seg, DivideMode._2D_Link)

        @viewer_seg.bind_key(parameters.pars.key_insert)
        @print_information("Insert")
        def insert_label(viewer_seg):
            self.cache_bbox.update_new_labels()
            label = self.cache_bbox.insert_label()
            self.vis.viewer.layers["segmentation"].selected_label = label
            self.vis.viewer.layers["segmentation"].mode = "paint"
            self.vis.widgets.show_state_info(f"Inserted a new label: {label}. Please draw with it.")

        def divide_and_relabel(mode) -> Tuple[ndarray, ndarray, Bbox, List[int]]:
            max_label = max(self.cache_bbox.bbox)
            pre_region, seg_subregion, slice_subregion = divide(max_label, mode)

            divided_labels = np.unique(seg_subregion)
            divided_labels = divided_labels[divided_labels > 0]
            if len(divided_labels) == 1:
                raise NoDivisionError

            subregion_old = self.labels[slice_subregion].copy()
            subregion_new = self.labels[slice_subregion].copy()

            subregion_new, labels = assign_new_labels(
                divided_labels, max_label, pre_region, seg_subregion, subregion_new, mode)

            return subregion_old, subregion_new, slice_subregion, labels

        def divide(max_label, mode):
            if mode == DivideMode._3D:
                # 3D mode
                bbox_subregion, subarray_bool = self.cache_bbox.get_subregion_3d(viewer_seg.selected_label)
                pre_region = None
                segmented_subregion = separate_one_cell_3d(subarray_bool)
            else:
                subarray_bool, bbox_subregion, pre_region = get_subregion2d_and_preslice(
                    self.labels, viewer_seg.selected_label, self.layer_selected)
                if mode == DivideMode._2D_Link:
                    # 2D link mode
                    segmented_subregion = divide_link(
                        subarray_bool, self.vis.widgets.max_division.value, pre_region, max_label)
                else:
                    # 2D mode
                    segmented_subregion = segment_one_cell_2d_watershed(
                        subarray_bool, self.vis.widgets.max_division.value)
            return pre_region, segmented_subregion, bbox_subregion

        def assign_new_labels(divided_labels, max_label: int, pre_region, seg_subregion, subregion_ori, mode):
            if mode == DivideMode._3D:
                # 3D mode
                return assign_new_labels_3d(max_label, seg_subregion, subregion_ori)
            else:
                mode = DivideMode._2D if pre_region is None else mode
                if mode == DivideMode._2D:
                    # 2D mode
                    return assign_new_labels_2d_wo_link(divided_labels, seg_subregion, subregion_ori)
                else:
                    # 2D link mode
                    return assign_new_labels_2d_link(seg_subregion, subregion_ori)

        def assign_new_labels_3d(max_label, segmented_subregion, subregion_ori):
            subregion_new, smaller_labels = keep_largest_label_unchange(segmented_subregion, subregion_ori)
            labels = [viewer_seg.selected_label] + smaller_labels
            return subregion_new, labels

        def keep_largest_label_unchange(seg, seg_ori):
            seg, other_labels = suppress_largest_label(seg)
            return replace_with_unused_labels(seg, other_labels, seg_ori)

        def assign_new_labels_2d_wo_link(divided_labels, seg_subregion, subregion_new) -> Tuple[ndarray, List[int]]:
            return replace_with_unused_labels(seg_subregion, divided_labels, subregion_new)

        def replace_with_unused_labels(seg, labels_ori, seg_result):
            ori = labels_ori  # 1d array of int
            tgt = get_unused_labels_quick(list(self.cache_bbox.bbox.keys()), len(labels_ori))
            for l_ori, l_tgt in zip(ori, tgt):
                seg_result[seg == l_ori] = l_tgt
            return seg_result, tgt

        def assign_new_labels_2d_link(segmented_subregion, subregion_new):
            updated_regions = segmented_subregion > 0
            segmented_subregion = relabel_min_change(segmented_subregion, list(self.cache_bbox.bbox.keys()))
            subregion_new[updated_regions] = segmented_subregion[updated_regions]

            labels = np.unique(segmented_subregion)
            labels: List[int] = labels[labels != 0].tolist()
            if viewer_seg.selected_label in labels:
                labels.remove(viewer_seg.selected_label)
                labels = [viewer_seg.selected_label] + labels
            return subregion_new, labels

        def short_str(list_: list):
            if len(list_)>3:
                return "[" + str(list_[0]) + ", " + str(list_[1]) + ", ..., " +str(list_[-1]) + "]"
            else:
                return str(list_)

        @viewer_seg.bind_key(parameters.pars.key_undo)
        @print_information()
        def undo(viewer_seg):
            """Undo one keyboard command"""
            self.vis.widgets.show_state_info("Undo")
            history: Optional[StateR2] = self.cache.load_cache(method="undo")
            if history is None:
                return
            self.cache_bbox.bbox = history.old.bboxes.copy()
            self.labels[history.bbox] = history.old.array
            self.reset_division_list()
            self._update_segmentation()
            self.update_info()

        @viewer_seg.bind_key(parameters.pars.key_redo)
        @print_information()
        def redo(viewer_seg):
            """Redo one keyboard command"""
            self.vis.widgets.show_state_info("Redo")
            future: Optional[StateR2] = self.cache.load_cache(method="redo")
            if future is None:
                return
            self.cache_bbox.bbox = future.new.bboxes.copy()
            self.labels[future.bbox] = future.new.array
            self.reset_division_list()
            self._update_segmentation()
            self.update_info()

        @viewer_seg.bind_key(parameters.pars.key_switch_one_label_all_labels)
        @print_information("Switch showing one label/all labels")
        def switch_showing_one_or_all_labels(viewer_seg):
            """Show the selected label"""
            self.vis.viewer.layers["segmentation"].show_selected_label = \
                not self.vis.viewer.layers["segmentation"].show_selected_label

        @viewer_seg.bind_key(parameters.pars.key_online_help)
        def help(viewer_seg):
            html_path = "https://github.com/WenChentao/Seg2Link/blob/master/Doc/help2.md"
            print(html_path)
            webbrowser.open(html_path)


class VisualizeAll(VisualizeBase):
    """Visualize the segmentation results"""

    def __init__(self, emseg2: Seg2LinkR2, raw: ndarray, cell_region: ndarray, cell_mask: ndarray):
        super().__init__(raw, cell_region, cell_mask)
        self.emseg2 = emseg2
        self.viewer.title = "Seg2link 2nd round"
        self.widgets = WidgetsR2(self)
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
    def __init__(self, emseg2: Seg2LinkR2):
        self.cache = CacheR2(maxlen=parameters.pars.cache_length_r2)
        self.emseg2 = emseg2

    def cache_state(self, state: StateR2):
        """Cache the previous & current states"""
        self.cache.append(state)

    def load_cache(self, method: str) -> Optional[Tuple[ndarray, Tuple[slice, slice, slice], str]]:
        """Load the cache"""
        if method == "undo":
            return self.cache.undo()
        elif method == "redo":
            return self.cache.redo()
        else:
            raise ValueError("Method must be 'undo' or 'redo'!")

    @property
    def cached_actions(self) -> List[str]:
        history = [hist.action + "\n" for hist in self.cache.history]
        future = [fut.action + "\n" for fut in self.cache.future][::-1]
        return history + [f"****(Head!)****\n"] + future


