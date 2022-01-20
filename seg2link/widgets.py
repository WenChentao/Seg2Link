import datetime
import textwrap
from enum import Enum
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from magicgui import widgets, use_app
from magicgui.types import FileDialogMode
from magicgui.widgets import Container
from scipy.stats import mode

import config
from misc import add_blank_lines, TinyCells, make_folder
from single_cell_division import DivideMode
from watersheds import labels_with_boundary, remove_boundary_scipy

if config.debug:
    from config import qprofile, lprofile
    from memory_profiler import profile as mprofile


class WidgetsA:
    def __init__(self, vis, img_shape: Tuple):
        self.viewer = vis.viewer
        self.emseg = vis.emseg

        shape_str = f"H: {img_shape[0]}  W: {img_shape[1]}  D: {img_shape[2]}"
        self.image_size = widgets.LineEdit(label="Image shape", value=shape_str, enabled=False)
        self.max_label = widgets.LineEdit(label="Largest label", enabled=False)
        self.cached_action = widgets.TextEdit(label="Cached actions",
                                              tooltip=(f"Less than {config.cache_length_first} action can be cached"),
                                              enabled=False)
        self.label_list_msg = widgets.LineEdit(label="Label list", enabled=False)

        self.hotkeys_info_value = '[Shift + N]: To next slice\n---------------' \
                                  '\n[R]:  Re-segment the last slice and link\n---------------' \
                                  '\n[K]:  Divide a label in the last slice\n---------------' \
                                  '\n[A]: Add labels to label list\n[C]: Clear the label list\n' \
                                  '[M]: Merge labels in label list' \
                                  '\n[D]: Delete labels in label list or the label selected\n ---------------' \
                                  '\n[Q]: Switch (selected | all labels)\n ---------------' \
                                  '\n[U]: Undo     [F]: Redo' \
                                  '\n[L]:  Picker   [E]: Eraser' \
                                  '\n[H]: Online Help'

        self.hotkeys_info = widgets.Label(value=self.hotkeys_info_value)

        self.export_button = widgets.PushButton(text="Export segmentation as .npy file")
        self.export_result = widgets.Label(value="")

        self.add_widgets()

    def add_widgets(self):
        container_states = Container(widgets=[self.image_size, self.max_label, self.cached_action, self.label_list_msg])
        container_export = Container(widgets=[self.export_button, self.export_result])
        container_states.min_height = 300
        self.viewer.window.add_dock_widget(container_states, name="States", area="right")
        self.viewer.window.add_dock_widget([self.hotkeys_info], name="HotKeys", area="right")
        self.viewer.window.add_dock_widget(container_export, name="Save/Export", area="right")

    def update_info(self):
        self.max_label.value = str(self.emseg.labels.max_label)
        self.cached_action.value = add_blank_lines("".join(self.emseg.cache.cached_actions),
                                                   config.cache_length_first + 1)
        self.label_list_msg.value = tuple(self.emseg.label_list)
        return None


class WidgetsB:
    def __init__(self, visualizeall):
        self.viewer = visualizeall.viewer
        self.emseg2 = visualizeall.emseg2

        # Hotkeys panel
        self.hotkeys_info_value = '[K]:  Divide one cell' \
                                  '\n---------------' \
                                  '\n[A]: Add labels to label list' \
                                  '\n[C]: Clear the label list' \
                                  '\n[M]: Merge labels in label list' \
                                  '\n[D]: Delete labels in label list or the label selected\n---------------' \
                                  '\n[Q]: Switch (selected | all labels)' \
                                  '\n[U]: Undo     [F]: Redo' \
                                  '\n[L]: Picker    [E]: Eraser'
        self.hotkeys_info = widgets.Label(value=self.hotkeys_info_value)

        # Label/cache panel
        self.max_label = widgets.LineEdit(label="Largest label", enabled=False)
        self.cached_action = widgets.TextEdit(label="Cached actions",
                                              tooltip=(f"Less than {config.cache_length_second} action can be cached"),
                                              value="", enabled=False)
        self.locate_cell_button = LocateSelectedCellButton(label="Select label")
        self.label_list_msg = widgets.LineEdit(label="Label list", enabled=False)

        # Divide panel
        self.divide_mode = widgets.RadioButtons(choices=DivideMode, value=DivideMode._2D,
                                                orientation="horizontal", label='Mode')
        tooltip = "Avoid generating a label with too small area,\nused when dividing a single cell"
        self.threshold_area = widgets.RadioButtons(choices=[0, 1, 5, 10], value=10, label="Min_area (%)",
                                                   orientation="horizontal", tooltip=tooltip)
        self.divide_msg = widgets.LineEdit(label="Divide cell", value="", enabled=False, visible=True)
        self.choose_box = widgets.SpinBox(min=1, max=1, label="Check it", visible=True)

        # Save/Export panel
        self.save_button = SaveAndSaveAs(label="Save segmentation")
        self.load_dialog = widgets.PushButton(text="Load segmentation (.npy)")
        self.max_cell_num = SpinBoxWithButton(label="Max cell No.")
        self.remove_and_save = widgets.PushButton(text="Remove tiny cells, Sorting and Save (.npy)")
        self.export_button = widgets.PushButton(text="Export segmentation as .tiff files")
        self.export_result = widgets.Label(value="")
        self.boundary_action = widgets.RadioButtons(choices=Boundary, value=Boundary.Default, label='Boundary',
                                                    orientation="horizontal")

        self.add_widgets()
        self.choose_box.changed.connect(self.locate_label_divided)

    def add_widgets(self):
        container_save_states = Container(widgets=[self.max_label, self.cached_action, self.locate_cell_button,
                                                   self.label_list_msg])
        container_divide_cell = Container(widgets=[
            self.divide_mode, self.threshold_area, self.divide_msg, self.choose_box])
        container_save_export = Container(
            widgets=[self.save_button, self.load_dialog, self.max_cell_num, self.remove_and_save,
                     self.boundary_action, self.export_button, self.export_result])

        self.viewer.window.add_dock_widget(container_save_states, name="States", area="right")
        self.viewer.window.add_dock_widget(container_divide_cell, name="Divide a single cell", area="right")
        self.viewer.window.add_dock_widget(container_save_export, name="Save/Export", area="right")
        self.viewer.window.add_dock_widget([self.hotkeys_info], name="HotKeys", area="left")

    def update_info(self, label_pre_division: int):
        label_max = np.max(self.emseg2.labels)
        labels_post_division = self.emseg2.divide_list
        if len(labels_post_division) != 0:
            self.choose_box.max = len(labels_post_division)
            self.divide_msg.value = f"Label {label_pre_division} " + u"\u2192" + f" {len(labels_post_division)} cells"
        else:
            self.choose_box.max = 1
            self.divide_msg.value = ""
        self.label_list_msg.value = tuple(self.emseg2.label_list)
        if label_max is not None:
            self.max_label.value = str(label_max)
            self.locate_cell_button.selected_label_.max = label_max

        self.cached_action.value = "".join(self.emseg2.cache.cached_actions)
        return None

    def locate_cell(self, label, dmode=DivideMode._3D):
        locs = np.where(self.emseg2.labels == label)
        if locs[0].size == 0:
            self.locate_cell_button.location.value = f"Not found"
        else:
            if dmode == DivideMode._3D:
                current_layer = mode(locs[2])[0][0]
            else:
                current_layer = self.emseg2.layer_selected
            self.emseg2.vis.viewer.dims.set_current_step(axis=2, value=current_layer)
            locs_current_layer = np.where(self.emseg2.labels[..., current_layer] == label)
            x_loc, y_loc = np.mean(locs_current_layer[0], dtype=int), np.mean(locs_current_layer[1], dtype=int)
            self.locate_cell_button.location.value = f"[{x_loc}, {y_loc}]"

    @lprofile
    def locate_label_divided(self):
        if self.emseg2.divide_list:
            label = self.emseg2.divide_list[self.choose_box.value - 1]
            self.locate_cell_button.selected_label_.value = label
            self.locate_cell(label, dmode=self.divide_mode.value)

    def widget_binding(self):
        search_button = self.locate_cell_button.locate_btn
        choose_cell_all = self.locate_cell_button.selected_label_
        save_button = self.save_button.save_btn
        save_as_button = self.save_button.save_as_btn
        load_dialog = self.load_dialog
        max_cell_num_btn = self.max_cell_num.estimate_btn
        max_cell_num = self.max_cell_num.spin_box
        remove_and_save = self.remove_and_save
        boundary_action = self.boundary_action
        export_button = self.export_button
        export_result = self.export_result

        @choose_cell_all.changed.connect
        def choose_label_all_():
            self.viewer.layers["segmentation"].selected_label = choose_cell_all.value

        @search_button.changed.connect
        def search_label():
            label = self.viewer.layers["segmentation"].selected_label
            self.locate_cell(label)

        @save_button.changed.connect
        def save_overwrite():
            if self.emseg2.labels_path.parent.exists():
                print("pressed save button")
                print(self.viewer.layers["segmentation"].data.dtype)
                export_result.value = "Saving segmentation as .npy file ..."
                np.save(self.emseg2.labels_path, self.viewer.layers["segmentation"].data)
                export_result.value = f"{self.emseg2.labels_path.name} was saved at: " \
                                      f"{datetime.datetime.now().strftime('%H:%M:%S')}"
            else:
                export_result.value = "Warning: Folder doesn't exist!"

        @save_as_button.changed.connect
        def save_as():
            if self.emseg2.labels_path.parent.exists():
                print("pressed save as button")
                print(self.viewer.layers["segmentation"].data.dtype)
                path = select_file()
                if path:
                    export_result.value = "Saving segmentation as .npy file ..."
                    np.save(path, self.viewer.layers["segmentation"].data)
                    export_result.value = f"{Path(path).name} was saved"
            else:
                export_result.value = "Warning: Folder doesn't exist!"

        def select_file():
            seg_filename = "seg-modified-" + datetime.datetime.now().strftime("%Y-%h-%d-%p%I") + ".npy"
            mode_ = FileDialogMode.OPTIONAL_FILE
            return use_app().get_obj("show_file_dialog")(
                mode_,
                caption=export_button.text,
                start_path=str(self.emseg2.labels_path.parent / seg_filename),
                filter=".npy")

        @load_dialog.changed.connect
        def load_npy():
            mode_ = FileDialogMode.EXISTING_FILE
            path = use_app().get_obj("show_file_dialog")(
                mode_,
                caption=load_dialog.text,
                start_path=str(self.emseg2.labels_path),
                filter='*.npy'
            )
            if path:
                self.emseg2.reset_labels(np.load(path))

        @max_cell_num_btn.changed.connect
        def show_info_remove_cells():
            tc = TinyCells(self.emseg2.labels)
            if max_cell_num.value >= len(tc.sorted_labels):
                export_result.value = f"No cells will be deleted. Total={len(tc.sorted_labels)}"
                max_cell_num.value = len(tc.sorted_labels)
            else:
                min_area, del_num = tc.min_area(max_cell_num.value)
                export_result.value = textwrap.fill(f"{del_num} cell < {min_area} voxels will be deleted. "
                                                    f"(total={len(tc.sorted_labels)})", width=27,
                                                    break_long_words=False)

        @remove_and_save.changed.connect
        def remove_save():
            if self.emseg2.labels_path.parent.exists():
                print("pressed remove and save button")
                export_result.value = "Saving .npy before removing ..."
                np.save(self.emseg2.labels_path.parent / "seg-modified_before_removing_tiny_cells.npy",
                        self.viewer.layers["segmentation"].data)
                export_result.value = "Removing tiny cells ..."
                tc = TinyCells(self.emseg2.labels)
                self.emseg2.labels = tc.remove_and_relabel(self.emseg2.labels, max_cell_num.value)
                self.emseg2._update_segmentation()
                export_result.value = "Saving .npy after removing ..."
                np.save(self.emseg2.labels_path.parent / "seg-modified_after_removing_tiny_cells.npy",
                        self.viewer.layers["segmentation"].data)
                export_result.value = f"Segementation were saved as:\n  {self.emseg2.labels_path.name}"
                self.emseg2.cache.cache.reset_cache_b()
                self.emseg2.update_info()
            else:
                export_result.value = "Warning: Folder doesn't exist!"

        @export_button.changed.connect
        def export():
            mode_ = FileDialogMode.EXISTING_DIRECTORY
            path = use_app().get_obj("show_file_dialog")(
                mode_,
                caption=export_button.text,
                start_path=str(self.emseg2.labels_path),
                filter=None
            )
            if path:
                export_result.value = "Exporting segmentation as .tiff files ..."
                if np.max(self.emseg2.labels) <= 65535:
                    dtype = np.uint16
                else:
                    dtype = np.uint32

                if boundary_action.value == Boundary.Add:
                    np.save(self.emseg2.labels_path.parent / "seg-modified_before_adding_boundary.npy",
                            self.viewer.layers["segmentation"].data)
                    self.emseg2.labels = labels_with_boundary(self.emseg2.labels)
                    self.emseg2._update_segmentation()
                    np.save(self.emseg2.labels_path.parent / "seg-modified_after_adding_boundary.npy",
                            self.viewer.layers["segmentation"].data)
                elif boundary_action.value == Boundary.Remove:
                    np.save(self.emseg2.labels_path.parent / "seg-modified_before_removing_boundary.npy",
                            self.viewer.layers["segmentation"].data)
                    self.emseg2.labels = remove_boundary_scipy(self.emseg2.labels)
                    self.emseg2._update_segmentation()
                    np.save(self.emseg2.labels_path.parent / "seg-modified_after_removing_boundary.npy",
                            self.viewer.layers["segmentation"].data)
                else:
                    pass

                path_ = make_folder(Path(path) / "seg_tiff")
                for z in range(self.emseg2.labels.shape[2]):
                    modified_seg = self.emseg2.labels[..., z].astype(dtype)
                    Image.fromarray(modified_seg).save(str(path_ / "seg_slice%04i.tiff") % z)
                export_result.value = "Segementation was exported as tiff images"
            else:
                export_result.value = "Warning: Folder doesn't exist!"


class Boundary(Enum):
    Default: str = "None"
    Add: str = "add"
    Remove: str = "remove"


class SpinBoxWithButton(Container):
    def __init__(self, **kwargs):
        self.spin_box = widgets.SpinBox(min=1, max=10 ** 8, value=65535)
        self.estimate_btn = widgets.PushButton(text="Estimate")
        kwargs["widgets"] = [self.spin_box, self.estimate_btn]
        kwargs["labels"] = False
        kwargs["layout"] = "horizontal"
        super().__init__(**kwargs)
        self.margins = (0, 0, 0, 0)
        self.estimate_btn.changed.disconnect()
        self.spin_box.changed.disconnect()


class LocateSelectedCellButton(Container):
    def __init__(self, **kwargs):
        self.selected_label_ = widgets.SpinBox(min=1, value=1)
        self.locate_btn = widgets.PushButton(text="Locate")
        self.location = widgets.LineEdit(enabled=False)
        kwargs["widgets"] = [self.selected_label_, self.locate_btn, self.location]
        kwargs["labels"] = False
        kwargs["layout"] = "horizontal"
        super().__init__(**kwargs)
        self.margins = (0, 0, 0, 0)
        self.locate_btn.changed.disconnect()
        self.selected_label_.changed.disconnect()


class SaveAndSaveAs(Container):
    def __init__(self, **kwargs):
        self.save_btn = widgets.PushButton(text="Save (.npy)")
        self.save_as_btn = widgets.PushButton(text="Save As (.npy)")
        kwargs["widgets"] = [self.save_btn, self.save_as_btn]
        kwargs["labels"] = False
        kwargs["layout"] = "horizontal"
        super().__init__(**kwargs)
        self.margins = (0, 0, 0, 0)
