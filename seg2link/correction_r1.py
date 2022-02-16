from __future__ import annotations

import copy
import datetime
import os
import webbrowser
from collections import deque, OrderedDict, namedtuple
from pathlib import Path
from typing import Tuple, Optional, List, Union, Set

import napari
import numpy as np
from PyQt5.QtWidgets import QApplication
from magicgui import use_app
from magicgui.types import FileDialogMode
from napari.utils.colormaps import low_discrepancy_image
from numpy import ndarray
from skimage.segmentation import relabel_sequential

from seg2link import config
from seg2link.emseg_core import Labels, Segmentation, Alignment, Archive
from seg2link.misc import print_information, TinyCells
from seg2link.single_cell_division import separate_one_label_r1
from seg2link.widgets_r1 import WidgetsR1

if config.DEBUG:
    from seg2link.config import lprofile

StateR1 = namedtuple("StateR1", ["labels", "seg_img", "aligned_img", "action"])


class Seg2LinkR1:
    """Segment the cells in 3D EM images"""

    def __init__(self, raw: ndarray, cell_region: ndarray, mask: Optional[ndarray], enable_mask: bool,
                 layer_num: int, path_save: Path, ratio_overlap: float, ratio_mask: float,
                 target_slice: int, enable_align: bool):
        self.current_slice = 0
        self.layer_num = layer_num
        self.seg_img_cache = OrderedDict()
        self.enable_align = enable_align
        self.label_list: Set[int] = set()
        self.cells_aligned = np.array([], dtype=np.uint8)
        self.cache = CacheState(self)
        self.archive = Archive(self, path_save)
        self.archive.make_folders()
        self.path_export = path_save
        self.seg = Segmentation(cell_region, enable_mask, mask, ratio_mask)
        self.vis = VisualizePartial(self, raw, cell_region, mask)
        self.labels = Labels(self, ratio_overlap)
        self.align = Alignment(enable_align)
        self.keys_binding()
        self.widget_binding()
        self.retrieve_history(target_slice)

    def initialize(self):
        print(self.labels)
        self.next_slice()
        print(self.labels)
        self.archive_state()
        self.vis.show_segmentation_r1()
        self.cache.cache_state(f"Next slice ({self.current_slice})")
        self.vis.update_info()

    def archive_state(self):
        self.archive.archive_state()

    def _update_state(self, current_seg: ndarray, cells_aligned: ndarray):
        self.seg.current_seg = current_seg.copy()
        self.cells_aligned = cells_aligned
        self.current_slice = self.labels.current_slice

    def _set_labels(self, labels: Union[List[List[int]], Labels]):
        """Used for setting two types of possible stored labels: list or Labels object"""
        if isinstance(labels, Labels):
            self.labels = copy.deepcopy(labels)
        else:
            self.labels._labels = labels
            self.labels.current_slice = len(labels)

    def retrieve_history(self, target_slice: int):
        history = self.archive.retrieve_history(target_slice, self.seg_img_cache)
        if history is None:
            self.initialize()
        else:
            labels, seg_img = history
            self._set_labels(labels)
            self._update_state(seg_img, self.seg.cell_region[..., self.labels.current_slice - 1])
            print(f"Retrieved the slice {self.labels.current_slice}")
            self.vis.show_segmentation_r1()
            self.cache.cache_state(f"Retrieve ({self.current_slice})")
            self.vis.update_info()

    def _link_and_relabel(self, reset_align: bool):
        self.labels.link_next_slice(reset_align)
        if self.current_slice > 1:
            self.labels.relabel()

    def next_slice(self):
        """Save label until current slice and then segment and link to the next slice"""
        self.current_slice += 1
        self.vis.widgets.show_state_info(f"Segmenting slice {self.current_slice} by watershed... Please wait")
        self.seg.watershed(self.current_slice)
        self.vis.widgets.show_state_info(f"Linking with previous slice {self.current_slice}... Please wait")
        self._link_and_relabel(reset_align=True)
        self.vis.widgets.show_state_info(f"Linking was done")

    def reseg_link(self, modified_label: ndarray):
        z = self.current_slice - self.vis.get_slice(self.current_slice).start - 1
        self.seg.reseg(modified_label[..., z], self.current_slice)
        if self.current_slice == 1:
            self.labels.reset()
        else:
            self.labels.rollback()
        self._link_and_relabel(reset_align=False)

    def divide_one_cell(self, modified_label: ndarray, selected_label: int):
        z = self.current_slice - self.vis.get_slice(self.current_slice).start - 1
        current_seg, _ = separate_one_label_r1(modified_label[..., z], selected_label, self.labels.max_label)
        _labels = np.unique(current_seg)
        self.labels._labels[-1] = _labels[_labels != 0].tolist()
        self.seg.current_seg = relabel_sequential(current_seg)[0]

    @lprofile
    def update(self, cache_action: Optional[str] = None):
        self.archive_state()
        self.vis.show_segmentation_r1()
        if cache_action:
            self.cache.cache_state(cache_action)
        self.vis.update_info()

    def keys_binding(self):
        """Set the hotkeys for user's operations"""
        viewer_seg = self.vis.viewer.layers["segmentation"]

        @viewer_seg.bind_key(config.pars.key_reseg_link_r1)
        @print_information("Re-segmentation and link")
        def re_seg_link(viewer_seg):
            """Re-segment current slice"""
            print(f"Resegment slice {self.current_slice}")
            self.reseg_link(viewer_seg.data)
            self.update("Re-segment")
            viewer_seg.mode = "pick"


        @viewer_seg.bind_key(config.pars.key_separate)
        @print_information("Divide a label")
        def re_divide_2d(viewer_seg):
            """Re-segment current slice"""
            if viewer_seg.selected_label == 0:
                print("\nLabel 0 should not be divided!")
            else:
                self.divide_one_cell(viewer_seg.data, viewer_seg.selected_label)

                viewer_seg._all_vals = low_discrepancy_image(
                    np.arange(self.labels.max_label + 1), viewer_seg._seed)
                viewer_seg._all_vals[0] = 0
                self.update("Divide")
                viewer_seg.mode = "pick"

        @viewer_seg.bind_key(config.pars.key_add)
        @print_information("Add labels to be processed")
        def append_label_list(viewer_seg):
            """Add label to be merged into a list"""
            if viewer_seg.mode != "pick":
                print("\nPlease switch to pick mode in segmentation layer to merge label")
            elif viewer_seg.selected_label == 0:
                print("\nLabel 0 should not be merged!")
            else:
                self.label_list.add(viewer_seg.selected_label)
                print("Labels to be processed: ", self.label_list)
                self.vis.update_info()

        @viewer_seg.bind_key(config.pars.key_clean)
        @print_information("Clean the label list")
        def clear_label_list(viewer_seg):
            """Clear labels in the merged list"""
            self.label_list.clear()
            print(f"Cleaned the label list: {self.label_list}")
            self.vis.update_info()

        @viewer_seg.bind_key(config.pars.key_merge)
        @print_information("Merge labels")
        def _merge(viewer_seg):
            if not self.label_list:
                print("No labels were merged")
            else:
                self.labels.merge(self.label_list)
                self.label_list.clear()
                self.update("Merge labels")

        @viewer_seg.bind_key(config.pars.key_delete)
        @print_information("Delete the selected label(s)")
        def del_label(viewer_seg):
            """Delete the selected label"""
            if viewer_seg.mode != "pick":
                print("\nPlease switch to pick mode in segmentation layer")
            elif viewer_seg.selected_label == 0:
                print("\nLabel 0 should not be deleted!")
            else:
                delete_list = self.label_list if self.label_list else viewer_seg.selected_label
                self.labels.delete(delete_list)
                self.label_list.clear()
                self.update("Delete label(s)")
                print(f"Label(s) {delete_list} were deleted")

        @viewer_seg.bind_key(config.pars.key_next_r1)
        @print_information("\nTo next slice")
        def _next_slice(viewer_seg):
            """To the next slice"""
            self.next_slice()
            self.update(f"Next slice ({self.current_slice})")


        @viewer_seg.bind_key(config.pars.key_undo)
        @print_information("Undo")
        def undo(viewer_seg):
            """Undo one keyboard command"""
            state: StateR1 = self.cache.load_cache("undo")
            self._set_labels(state.labels)
            self._update_state(state.seg_img, state.aligned_img)
            self.update()

        @viewer_seg.bind_key(config.pars.key_redo)
        @print_information("Redo")
        def redo(viewer_seg):
            """Undo one keyboard command"""
            state: StateR1 = self.cache.load_cache("redo")
            self._set_labels(state.labels)
            self._update_state(state.seg_img, state.aligned_img)
            self.update()

        @viewer_seg.bind_key(config.pars.key_switch_one_label_all_labels)
        @print_information("Switch showing one label/all labels")
        def switch_showing_one_or_all_labels(viewer_seg):
            """Show the selected label"""
            self.vis.viewer.layers["segmentation"].show_selected_label = \
                not self.vis.viewer.layers["segmentation"].show_selected_label

        @viewer_seg.bind_key(config.pars.key_online_help)
        def help(viewer_seg):
            html_path = "file://" + os.path.abspath("../Help/help1.html")
            print(html_path)
            webbrowser.open(html_path)

    def widget_binding(self):
        export_button = self.vis.widgets.export_button
        state_info = self.vis.widgets.state_info

        @export_button.changed.connect
        def export_array():
            seg_filename = "Seg-" + datetime.datetime.now().strftime("%Y-%h-%d") + ".npy"
            mode_ = FileDialogMode.OPTIONAL_FILE
            path = use_app().get_obj("show_file_dialog")(
                mode_,
                caption=export_button.text,
                start_path=str(self.path_export / seg_filename),
                filter=".npy"
            )
            if path:
                self.vis.widgets.show_state_info("Generating segmentation... Please wait")
                seg_array = self.labels.to_multiple_labels(slice(0, self.current_slice))
                self.vis.widgets.show_state_info("Sorting labels... Please wait")
                sorted_labels = self.sort_remove_tiny(seg_array)
                self.vis.widgets.show_state_info("Save segmentation as npy... Please wait")
                np.save(path, sorted_labels)
                self.vis.widgets.show_state_info("Segmentation was exported")
            else:
                self.vis.widgets.show_state_info("Warning: Folder doesn't exist!")

    @staticmethod
    def sort_remove_tiny(seg_array):
        tc = TinyCells(seg_array)
        tc.sort_by_areas()
        if config.pars.dtype_r2 == np.uint16 and seg_array.dtype == np.uint32:
            sorted_labels = tc.remove_and_relabel(seg_array, config.pars.upper_limit_export_r1).astype(np.uint16)
        elif config.pars.dtype_r2 == np.uint32:
            sorted_labels = tc.remove_and_relabel(seg_array)
        else:
            raise ValueError("config.pars.dtype_r2 should be np.uint32 or np.uint16")
        return sorted_labels


class VisualizeBase:
    """Visualize the segmentation results"""

    def __init__(self, raw: ndarray, cell_region: ndarray, cell_mask: ndarray):
        self.raw = raw
        self.cell_region = cell_region
        self.cell_mask = cell_mask
        self.scale = config.pars.scale_xyz
        self.viewer = self.initialize_viewer()

    def initialize_viewer(self):
        """Initialize the napari viewer"""
        viewer = napari.Viewer()
        QApplication.processEvents()

        putative_data32bit = np.zeros((*self.raw.shape[:2], 2), dtype=np.uint32)
        putative_data = np.zeros((*self.raw.shape[:2], 2), dtype=np.uint8)
        if self.cell_mask is not None:
            viewer.add_labels(putative_data, name='mask_cells', color={0: "k", 1: "w"}, visible=False, scale=self.scale)
        viewer.add_image(
            putative_data, name='raw_image', contrast_limits=[0, 2 ** config.pars.raw_bit - 1], scale=self.scale
        )
        if self.cell_region is not None:
            viewer.add_labels(putative_data, name='cell_region', color={0: "k", 1: "w"}, opacity=0.4, scale=self.scale)
        viewer.add_labels(putative_data32bit, name='segmentation', num_colors=100, scale=self.scale)
        viewer.dims.set_axis_label(axis=2, label="Slice (0-0)")
        viewer.dims.order = (2, 0, 1)
        viewer.layers["segmentation"].mode = "pick"
        return viewer


class VisualizePartial(VisualizeBase):
    """Visualize the segmentation results"""

    def __init__(self, emseg1: Seg2LinkR1, raw: ndarray, cell_region: ndarray, cell_mask: Optional[ndarray]):
        super().__init__(raw, cell_region, cell_mask)
        self.emseg1 = emseg1
        self.layer_num = cell_region.shape[-1]
        self.add_layers()
        self.widgets = WidgetsR1(self, cell_region.shape)

    def update_info(self):
        s = self.get_slice(self.emseg1.current_slice)
        self.widgets.update_info()
        self.viewer.dims.set_axis_label(axis=2, label=f"Slice ({s.start + 1}-{s.stop})")

    def get_slice(self, current_layer: int) -> slice:
        if current_layer == 0:
            return slice(-1, 0)
        start = max(current_layer - config.pars.max_draw_layers_r1 // 2, 0)
        stop = min(start + config.pars.max_draw_layers_r1, self.layer_num)
        return slice(start, stop)

    def add_layers(self):
        """Initialize the napari viewer"""
        self.viewer.title = "Seg2Link 1st round"
        if self.emseg1.enable_align:
            putative_data = np.zeros((2, *self.cell_region.shape[:2]), dtype=np.uint8)
            self.viewer.add_labels(putative_data, name="cell_region_aligned", color={0: "k", 1: "w"}, visible=False)
        return None

    @lprofile
    def show_segmentation_r1(self, reset_focus: bool = True):
        """Update the segmentation results and other images/label"""
        current_slice = self.emseg1.current_slice
        slice_layers = self.get_slice(current_slice)
        labels = self.emseg1.labels.to_multiple_labels(slice_layers)

        if self.cell_mask is not None:
            self.viewer.layers['mask_cells'].data = self.cell_mask[..., slice_layers]
        self.viewer.layers['raw_image'].data = self.raw[..., slice_layers]
        if self.cell_region is not None:
            self.viewer.layers['cell_region'].data = self.cell_region[..., slice_layers]
        self.viewer.layers['segmentation'].data = labels
        if self.emseg1.enable_align:
            self.viewer.layers['cell_region_aligned'].data = \
                self._expand_align_to_3d(slice_layers, current_slice)

        current_layer_relative = current_slice - slice_layers.start - 1
        if reset_focus:
            self.viewer.dims.set_current_step(axis=2, value=current_layer_relative)

        QApplication.processEvents()

    def _expand_align_to_3d(self, layers: slice, current_layer: int) -> ndarray:
        aligned_img = self.emseg1.cells_aligned
        aligned_cell_3d = np.tile(np.zeros_like(aligned_img), reps=(layers.stop - layers.start, 1, 1))
        aligned_cell_3d[current_layer - layers.start - 1, ...] = aligned_img
        return aligned_cell_3d

refaction = 1

class Cache:
    def __init__(self, maxlen: int):
        self.history: deque = deque(maxlen=maxlen)
        self.future: deque = deque(maxlen=maxlen)

    def __repr__(self):
        return f"Cached history: {len(self.history)}; Cached future: {len(self.future)}"

    def undo(self):
        raise NotImplementedError

    def redo(self):
        raise NotImplementedError

    def append(self, element):
        self.history.append(element)
        self.future.clear()


class CacheR1(Cache):
    def undo(self):
        if len(self.history) == 1:
            print("No earlier cached state!")
        else:
            self.future.append(self.history.pop())
        return self.history[-1]

    def redo(self):
        if not self.future:
            print("No later state!")
        else:
            self.history.append(self.future.pop())
        return self.history[-1]


class CacheState:
    def __init__(self, emseg1: Seg2LinkR1):
        self.cache = CacheR1(maxlen=config.pars.cache_length_r1)
        self.emseg1 = emseg1

    def cache_state(self, action: str):
        """Cache the current state"""
        state = StateR1(copy.deepcopy(self.emseg1.labels._labels), self.emseg1.seg.current_seg.copy(),
                        self.emseg1.cells_aligned.copy(), action)
        self.cache.append(state)

    def load_cache(self, method: str) -> Tuple[Labels, ndarray, ndarray, str]:
        """load the cache of the emseg1 states"""
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
        history_str_list = history + [f"***Head ({len(self.cache.history[-1].labels)})***\n"] + future
        return history_str_list
