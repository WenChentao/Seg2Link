import datetime
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QApplication
from magicgui import widgets, use_app
from magicgui.types import FileDialogMode
from magicgui.widgets import Container

from seg2link import config
from seg2link.misc import TinyCells, make_folder
from seg2link.msg_windows_r2 import sort_remove_window
from seg2link.single_cell_division import DivideMode
from seg2link.cache_bbox import NoLabelError
from seg2link.watersheds import labels_with_boundary, remove_boundary_scipy


class WidgetsR2:
    def __init__(self, visualizeall):
        self.viewer = visualizeall.viewer
        self.emseg2 = visualizeall.emseg2
        self.tiny_cells = TinyCells(self.emseg2.labels)
        self.label_max = 0

        # Hotkeys panel
        self.hotkeys_info_value = '[K]:  Divide one label' \
                                  '\n---------------' \
                                  '\n[A]: Add a label into the label list (LL)' \
                                  '\n[C]: Clear LL' \
                                  '\n[M]: Merge labels in LL' \
                                  '\n[D]: Delete a selected label or labels in LL' \
                                  '\n---------------' \
                                  '\n[Q]: Switch: Viewing one label | all labels' \
                                  '\n[U]: Undo     [F]: Redo' \
                                  '\n[L]: Picker    [E]: Eraser' \
                                  '\n[H]: Online Help'
        self.hotkeys_info = widgets.Label(value=self.hotkeys_info_value)

        # Label/cache panel
        self.max_label_info = widgets.LineEdit(label="Largest label", enabled=False)
        self.cached_action = widgets.TextEdit(label="Cached actions",
                                              tooltip=(f"Less than {config.pars.cache_length_r2} action can be cached"),
                                              value="", enabled=False)
        self.locate_cell_button = LocateSelectedCellButton(label="Select label")
        self.label_list_msg = widgets.LineEdit(label="Label list", enabled=False)

        # Divide panel
        self.divide_mode = widgets.RadioButtons(choices=["2D","2D Link","3D"], value="2D",
                                                orientation="horizontal", label='Mode')
        tooltip = "Avoid generating a label with too small area,\nused when dividing a single cell"
        self.max_division = widgets.RadioButtons(choices=[2, 4, 8, "Inf"], value=2, label="Max division",
                                                 orientation="horizontal", tooltip=tooltip)
        self.divide_msg = widgets.LineEdit(label="Divide cell", value="", enabled=False, visible=True)
        self.choose_box = widgets.SpinBox(min=1, max=1, label="Check it", visible=True)

        # Save/Export panel
        self.save_button = SaveAndSaveAs(label="Save segmentation")
        self.load_dialog = widgets.PushButton(text="Load segmentation (.npy)")
        self.remove_and_save = widgets.PushButton(text="Sort labels and remove tiny cells")
        self.export_button = widgets.PushButton(text="Export segmentation as .tiff files")
        self.state_info = widgets.Label(value="")
        self.boundary_action = widgets.RadioButtons(choices=Boundary, value=Boundary.Default, label='Boundary',
                                                    orientation="horizontal")

        self.remove_sort_window = sort_remove_window

        self.add_widgets()
        self.choose_box.changed.connect(self.locate_label_divided)
        QApplication.processEvents()

    def add_widgets(self):
        container_save_states = Container(widgets=[self.max_label_info, self.cached_action, self.locate_cell_button,
                                                   self.label_list_msg])
        container_divide_cell = Container(widgets=[
            self.divide_mode, self.max_division, self.divide_msg, self.choose_box])
        container_save_export = Container(
            widgets=[self.save_button, self.load_dialog, self.remove_and_save,
                     self.boundary_action, self.export_button])

        self.viewer.window.add_dock_widget(container_save_states, name="States", area="right")
        self.viewer.window.add_dock_widget(container_divide_cell, name="Divide a single cell", area="right")
        self.viewer.window.add_dock_widget(container_save_export, name="Save/Export", area="right")
        self.viewer.window.add_dock_widget([self.state_info], name="State info", area="right")
        self.viewer.window.add_dock_widget([self.hotkeys_info], name="HotKeys", area="left")

    def update_info(self, label_pre_division: int):
        self.label_max = max(self.emseg2.cache_bbox.bbox)
        labels_post_division = self.emseg2.divide_list
        if len(labels_post_division) != 0:
            self.choose_box.max = len(labels_post_division)
            self.divide_msg.value = f"Label {label_pre_division} " + u"\u2192" + f" {len(labels_post_division)} cells"
        else:
            self.choose_box.max = 1
            self.divide_msg.value = ""
        self.label_list_msg.value = tuple(self.emseg2.label_set)
        if self.label_max is not None:
            self.max_label_info.value = str(self.label_max)
            self.locate_cell_button.selected_label_.max = self.label_max

        self.cached_action.value = "".join(self.emseg2.cache.cached_actions)
        return None

    def locate_label_divided(self):
        if self.emseg2.divide_list:
            target_label = self.emseg2.divide_list[self.choose_box.value - 1]
            self.locate_cell_button.selected_label_.value = target_label
            try:
                self.locate_cell(target_label,
                                 dmode=self.divide_mode.value,
                                 subregion_slice=self.emseg2.divide_subregion_slice)
            except NoLabelError:
                self.locate_cell_button.location.value = "Not found"

    def locate_cell(self, label, dmode=DivideMode._3D, subregion_slice=None):
        if dmode == DivideMode._3D:
            return self.locate_cell_3d(label, subregion_slice)
        else:
            return self.locate_cell_2d(label)

    def locate_cell_2d(self, label):
        current_layer = self.emseg2.layer_selected
        locs_current_layer = np.where(self.emseg2.labels[..., current_layer] == label)
        if locs_current_layer[0].size == 0:
            self.locate_cell_button.location.value = f"Not found in current slice"
        else:
            self.emseg2.vis.viewer.dims.set_current_step(axis=2, value=current_layer)
            x_loc, y_loc = np.mean(locs_current_layer[0], dtype=int), np.mean(locs_current_layer[1], dtype=int)
            self.locate_cell_button.location.value = f"[{x_loc}, {y_loc}]"

    def locate_cell_3d(self, label, subregion_slice=None):
        try:
            stored_bbox = self.emseg2.cache_bbox.bbox[label]
        except KeyError:
            raise NoLabelError
        center_layer, x_loc, y_loc = self.locate_cell_3d_subregion(
            self.emseg2.labels[stored_bbox], label, subregion_slice
        )
        center_layer, x_loc, y_loc = center_layer + stored_bbox[2].start, \
                                     x_loc + stored_bbox[0].start, \
                                     y_loc + stored_bbox[1].start
        self.emseg2.vis.viewer.dims.set_current_step(axis=2, value=center_layer)
        self.locate_cell_button.location.value = f"[{x_loc}, {y_loc}]"

    @staticmethod
    def locate_cell_3d_subregion(label_img, label, subregion_slice=None):
        label_subimg = label_img.view() if subregion_slice is None else label_img[subregion_slice]
        label_projected_along_z = np.any(label_subimg == label, axis=(0, 1))
        if not np.any(label_projected_along_z):
            raise NoLabelError
        layers_with_label = np.flatnonzero(label_projected_along_z)
        center_layer = layers_with_label[len(layers_with_label) // 2]
        locs_current_layer = np.where(label_subimg[..., center_layer] == label)
        x_loc, y_loc = np.mean(locs_current_layer[0], dtype=int), np.mean(locs_current_layer[1], dtype=int)
        if subregion_slice is None:
            return center_layer, x_loc, y_loc
        else:
            return center_layer + subregion_slice[2].start, \
                   x_loc + subregion_slice[0].start, \
                   y_loc + subregion_slice[1].start

    def show_state_info(self, info: str):
        self.state_info.value = info
        print(info)
        QApplication.processEvents()

    def widget_binding(self):
        search_button = self.locate_cell_button.locate_btn
        choose_cell_all = self.locate_cell_button.selected_label_
        save_button = self.save_button.save_btn
        save_as_button = self.save_button.save_as_btn
        load_dialog = self.load_dialog
        remove_and_save = self.remove_and_save
        boundary_action = self.boundary_action
        export_button = self.export_button

        @choose_cell_all.changed.connect
        def choose_label_all_():
            self.viewer.layers["segmentation"].selected_label = choose_cell_all.value

        @search_button.changed.connect
        def search_label():
            label = self.viewer.layers["segmentation"].selected_label
            try:
                self.show_state_info("Searching... Please wait")
                self.locate_cell(label)
                self.show_state_info(f"Label {label} was found")
            except NoLabelError:
                self.show_state_info(f"Label {label} was Not found")

        @save_button.changed.connect
        def save_overwrite():
            if self.viewer.layers["segmentation"].data.dtype != config.pars.dtype_r2:
                self.show_state_info(f"Warning: dtype should be {config.pars.dtype_r2}!")
            elif self.emseg2.labels_path.parent.exists():
                self.show_state_info("Saving segmentation as .npy file... Please wait")
                np.save(self.emseg2.labels_path, self.viewer.layers["segmentation"].data)
                self.show_state_info(f"{self.emseg2.labels_path.name} was saved at: "
                                     f"{datetime.datetime.now().strftime('%H:%M:%S')}")
            else:
                self.show_state_info("Warning: Folder doesn't exist!")

        @save_as_button.changed.connect
        def save_as():
            if self.viewer.layers["segmentation"].data.dtype != config.pars.dtype_r2:
                self.show_state_info(f"Warning: dtype should be {config.pars.dtype_r2}!")
            elif self.emseg2.labels_path.parent.exists():
                path = select_file()
                if path:
                    self.show_state_info("Saving segmentation as .npy file... Please wait")
                    np.save(path, self.viewer.layers["segmentation"].data)
                    self.show_state_info(f"{Path(path).name} was saved")
            else:
                self.show_state_info("Warning: Folder doesn't exist!")

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

        @remove_and_save.changed.connect
        def show_info_remove_cells():
            self.show_state_info("Sorting cells... Please wait")
            self.tiny_cells.sort_by_areas()
            self.remove_sort_window.width = 400
            self.remove_sort_window.height = 200
            self.remove_sort_window.show(run=True)
            cell_num = self.tiny_cells.sorted_labels.size
            self.show_state_info(f"Found {cell_num} cells")
            self.remove_sort_window.info.value = f"Found {cell_num} cells"
            self.remove_sort_window.max_cell_num.max = cell_num
            self.remove_sort_window.max_cell_num.value = cell_num if cell_num < 65535 else 65535
            self.remove_sort_window.save_button.changed.connect(save_sorted_labels)
            self.remove_sort_window.cancel_button.changed.connect(cancel)
            self.remove_sort_window.max_cell_num.changed.connect(estimate_cell_sizes_removed)
            estimate_cell_sizes_removed()

        def save_sorted_labels():
            remove_save(self.remove_sort_window.max_cell_num.value)
            self.remove_sort_window.hide()

        def remove_save(max_cell_num):
            if self.emseg2.labels_path.parent.exists():
                self.show_state_info("Saving segmentation before relabeling... Please wait")
                np.save(self.emseg2.labels_path.parent / "seg-modified_before_sort_remove.npy",
                        self.viewer.layers["segmentation"].data)
                self.show_state_info("Relabeling/Removing tiny cells... Please wait")
                self.emseg2.labels = self.tiny_cells.remove_and_relabel(self.emseg2.labels, max_cell_num)
                self.emseg2._update_segmentation()
                self.show_state_info("Saving segmentation after relabeling... Please wait")
                np.save(self.emseg2.labels_path.parent / "seg-modified_after_sort_remove.npy",
                        self.viewer.layers["segmentation"].data)
                self.show_state_info(f"Segmentation was saved as: seg-modified_after_sort_remove.npy")
                self.emseg2.cache.cache.reset_cache_b()
                self.emseg2.update_info()
            else:
                self.show_state_info("Warning: Folder doesn't exist!")

        def cancel():
            self.remove_sort_window.hide()

        def estimate_cell_sizes_removed():
            max_area_delete, num_delete = self.tiny_cells.min_area(self.remove_sort_window.max_cell_num.value)
            if num_delete == 0:
                self.remove_sort_window.removed_cell_size.value = "No cell will be removed"
            else:
                self.remove_sort_window.removed_cell_size.value = \
                    f"Cells < {self.tiny_cells.min_area(self.remove_sort_window.max_cell_num.value)[0]} " \
                    f"voxels will be removed"

        @export_button.changed.connect
        def boundary_process_export():
            mode_ = FileDialogMode.EXISTING_DIRECTORY
            path = use_app().get_obj("show_file_dialog")(
                mode_,
                caption=export_button.text,
                start_path=str(self.emseg2.labels_path),
                filter=None
            )
            if path:
                self.show_state_info("Modifying boundary... Please wait")
                modify_boundary()
                transformed_labels = transform_dtype(self.emseg2.labels)

                self.show_state_info("Saving images... Please wait")
                path_ = make_folder(Path(path) / "seg_tiff")
                for z, img_z in enumerate(transformed_labels):
                    Image.fromarray(img_z).save(str(path_ / "seg_slice%04i.tiff") % z)
                self.show_state_info("Segementation was exported as tiff images")
            else:
                self.show_state_info("Warning: Folder doesn't exist!")

        def modify_boundary():
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

        def transform_dtype(labels):
            if labels.dtype == np.uint16 or np.max(labels) > 65535:
                transformed_labels = labels.transpose((2, 0, 1))
            elif labels.dtype == np.uint32 or labels.dtype == np.int32:
                transformed_labels = labels.view(np.uint16)[:, :, ::2].transpose((2, 0, 1))
            else:
                raise ValueError(f"emseg2.labels.dtype is {labels.dtype} "
                                 f"but should be np.uint32 or np.uint16")
            return transformed_labels


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


class SaveAndSaveAs(Container):
    def __init__(self, **kwargs):
        self.save_btn = widgets.PushButton(text="Save (.npy)")
        self.save_as_btn = widgets.PushButton(text="Save As (.npy)")
        kwargs["widgets"] = [self.save_btn, self.save_as_btn]
        kwargs["labels"] = False
        kwargs["layout"] = "horizontal"
        super().__init__(**kwargs)
        self.margins = (0, 0, 0, 0)


class Boundary(Enum):
    Default: str = "None"
    Add: str = "add"
    Remove: str = "remove"