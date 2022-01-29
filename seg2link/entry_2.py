import warnings
from pathlib import Path

import numpy as np
from magicgui import magicgui

from seg2link import config
from seg2link.entry_1 import load_cells, load_raw, load_mask, _npy_name, update_error_info
from seg2link.second_correction import Seg2LinkR2
from seg2link.userconfig import UserConfig

CURRENT_DIR = Path.home()
USR_CONFIG = UserConfig()


@magicgui(
    call_button="Start Seg2Link (2nd round)",
    layout="vertical",
    load_para={"widget_type": "PushButton", "text": "Load parameters (*.ini)"},
    save_para={"widget_type": "PushButton", "text": "Save parameters (*.ini)"},
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
    enable_cell={"label": "Use the cell/noncell image"},
)
def widget_entry2(
        load_para,
        save_para,
        enable_mask=False,
        enable_cell=False,
        path_cells=CURRENT_DIR,
        path_raw=CURRENT_DIR,
        path_mask=CURRENT_DIR,
        file_seg=CURRENT_DIR / "Seg.npy",
        paths_exist=["", "", "", ""],
        error_info="",
        image_size="",
        cell_value=2,
        mask_value=2,
):
    """Run some computation."""
    cells = load_cells(cell_value, path_cells, file_cached=_npy_name(path_cells)) if enable_cell else None
    images = load_raw(path_raw, file_cached=_npy_name(path_raw))
    mask_dilated = load_mask(mask_value, path_mask, file_cached=_npy_name(path_mask)) if enable_mask else None
    segmentation = load_segmentation(file_seg)
    Seg2LinkR2(images, cells, mask_dilated, segmentation, file_seg)
    return None


def load_segmentation(file_seg):
    segmentation = np.load(str(file_seg))
    if segmentation.dtype != config.pars.dtype_r2:
        warnings.warn(f"segmentation should has dtype {config.pars.dtype_r2}. Transforming...")
        segmentation = segmentation.astype(config.pars.dtype_r2, copy=False)
    label_shape = segmentation.shape
    widget_entry2.image_size.value = f"H: {label_shape[0]}  W: {label_shape[1]}  D: {label_shape[2]}"
    print("Segmentation shape:", label_shape, "dtype:", segmentation.dtype)
    return segmentation


@widget_entry2.enable_mask.changed.connect
def use_mask():
    visible = widget_entry2.enable_mask.value
    widget_entry2.path_mask.visible = visible
    widget_entry2.mask_value.visible = visible


@widget_entry2.save_para.changed.connect
def _on_save_para_changed():
    parameters_r2 = {"path_cells": widget_entry2.path_cells.value,
                     "path_raw": widget_entry2.path_raw.value,
                     "path_mask": widget_entry2.path_mask.value,
                     "file_seg": widget_entry2.file_seg.value,
                     "cell_value": widget_entry2.cell_value.value,
                     "mask_value": widget_entry2.mask_value.value}
    USR_CONFIG.save_ini_r2(parameters_r2)


@widget_entry2.load_para.changed.connect
def _on_load_para_changed():
    USR_CONFIG.load_ini()
    config.pars.set_from_dict(USR_CONFIG.pars.advanced)

    if USR_CONFIG.pars.r2:
        set_pars_r2(USR_CONFIG.pars.r2)
    else:
        set_pars_r2(USR_CONFIG.pars.r1)


def set_pars_r2(parameters: dict):
    widget_entry2.path_cells.value = parameters["path_cells"]
    widget_entry2.path_raw.value = parameters["path_raw"]
    widget_entry2.path_mask.value = parameters["path_mask"]
    if parameters.get("file_seg"):
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
