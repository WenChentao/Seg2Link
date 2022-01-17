from pathlib import Path

import numpy as np
from magicgui import magicgui, use_app
from magicgui.types import FileDialogMode

from entry_1 import load_cells, load_raw, load_mask, read_ini, save_ini, _npy_name, update_error_info
from second_correction import Seg2LinkR2


@magicgui(
    call_button="Start Seg2Link (2nd round)",
    layout="vertical",
    load_para={"widget_type": "PushButton", "text": "Load parameters (*.r2.ini)"},
    save_para={"widget_type": "PushButton", "text": "Save parameters (*.r2.ini)"},
    cell_value={"label": "Value of the cell region"},
    mask_value={"label": "Value of the mask region", "visible": False},
    paths_exist={"visible": False, "enabled": False},
    error_info={"label": "   !Following path(s) are not exist:", "enabled": False, "visible": False},
    image_size={"label": "Image size (segmentation)", "enabled": False},
    path_cells={"label": "Folder for cell images (*.tiff):", "mode": "d"},
    path_raw={"label": "Folder for the raw images (*.tiff):", "mode": "d"},
    path_mask={"label": "Folder for the mask images (*.tiff):", "mode": "d", "visible": False},
    file_seg={"label": "Segmentation file (*.npy):", "mode": "r", "filter": '*.npy'},
    enable_mask={"label": "Use the mask image"},
)
def widget_entry2(
        load_para,
        save_para,
        enable_mask=False,
        path_cells=Path.cwd(),
        path_raw=Path.cwd(),
        path_mask=Path.cwd(),
        file_seg=Path.cwd() / "Seg.npy",
        paths_exist=["", "", "", ""],
        error_info="",
        image_size="",
        cell_value=2,
        mask_value=2,
):
    """Run some computation."""
    cells = load_cells(cell_value, path_cells, file_cached=_npy_name(path_cells))
    images = load_raw(path_raw, file_cached=_npy_name(path_raw))
    if enable_mask:
        mask_dilated = load_mask(mask_value, path_mask, file_cached=_npy_name(path_mask))
    else:
        mask_dilated = None
    segmentation = np.load(str(file_seg))
    label_shape = segmentation.shape
    widget_entry2.image_size.value = f"H: {label_shape[0]}  W: {label_shape[1]}  D: {label_shape[2]}"
    Seg2LinkR2(images, cells, mask_dilated, segmentation, file_seg)
    return None


@widget_entry2.enable_mask.changed.connect
def use_mask():
    visible = widget_entry2.enable_mask.value
    widget_entry2.path_mask.visible = visible
    widget_entry2.mask_value.visible = visible


@widget_entry2.save_para.changed.connect
def _on_save_para_changed():
    seg_filename = "para_data_xx.r2.ini"
    mode_ = FileDialogMode.OPTIONAL_FILE
    path = use_app().get_obj("show_file_dialog")(
        mode_,
        caption="Save ini",
        start_path=str(Path.cwd() / seg_filename),
        filter='*.r2.ini'
    )
    if path:
        save_ini({"path_cells": widget_entry2.path_cells.value,
                  "path_raw": widget_entry2.path_raw.value,
                  "path_mask": widget_entry2.path_mask.value,
                  "file_seg": widget_entry2.file_seg.value,
                  "cell_value": widget_entry2.cell_value.value,
                  "mask_value": widget_entry2.mask_value.value},
                 Path(path))


@widget_entry2.load_para.changed.connect
def _on_load_para_changed():
    mode_ = FileDialogMode.EXISTING_FILE
    path = use_app().get_obj("show_file_dialog")(
        mode_,
        caption="Load ini",
        start_path=str(Path.cwd()),
        filter='*.r2.ini'
    )
    if path:
        parameters = read_ini(Path(path))
        widget_entry2.path_cells.value = parameters["path_cells"]
        widget_entry2.path_raw.value = parameters["path_raw"]
        widget_entry2.path_mask.value = parameters["path_mask"]
        widget_entry2.file_seg.value = parameters["file_seg"]
        widget_entry2.cell_value.value = int(parameters["cell_value"])
        widget_entry2.mask_value.value = int(parameters["mask_value"])


def set_path_error_info(widget_entry2_, num: int, error: bool):
    num_str = {1: "Cell", 2: "Raw", 3: "Mask", 4: "Seg"}
    update_error_info(error, num, num_str, widget_entry2_)


@widget_entry2.file_seg.changed.connect
def _on_file_seg_changed():
    if widget_entry2.file_seg.value.exists():
        set_path_error_info(widget_entry2, 4, False)
    else:
        set_path_error_info(widget_entry2, 4, True)


@widget_entry2.path_cells.changed.connect
def _on_path_cells_changed():
    if widget_entry2.path_cells.value.exists():
        new_cwd = widget_entry2.path_cells.value.parent
        widget_entry2.path_raw.value = new_cwd
        widget_entry2.path_mask.value = new_cwd
        widget_entry2.file_seg.value = new_cwd.parent / "Seg.npy"
        set_path_error_info(widget_entry2, 1, False)
    else:
        set_path_error_info(widget_entry2, 1, True)


@widget_entry2.path_raw.changed.connect
def _on_path_raw_changed():
    if widget_entry2.path_raw.value.exists():
        set_path_error_info(widget_entry2, 2, False)
    else:
        set_path_error_info(widget_entry2, 2, True)


@widget_entry2.path_mask.changed.connect
def _on_path_mask_changed():
    if widget_entry2.path_mask.value.exists():
        set_path_error_info(widget_entry2, 3, False)
    else:
        set_path_error_info(widget_entry2, 3, True)


if __name__ == "__main__":
    widget_entry2.show(run=True)
