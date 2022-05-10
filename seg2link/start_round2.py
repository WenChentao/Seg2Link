import warnings
from pathlib import Path
from typing import List

import numpy as np
from magicgui import magicgui

from seg2link.misc import load_image_pil, load_image_lazy
from seg2link import parameters
from seg2link.start_round1 import load_cells, load_mask, _npy_name, check_existence_path, show_error_msg, set_pars_r1r2, \
    check_tiff_existence
from seg2link.seg2link_round2 import Seg2LinkR2
from seg2link.userconfig import UserConfig, get_config_dir

try:
    CONFIG_DIR = get_config_dir()
except Exception:
    CONFIG_DIR = Path.home()

CURRENT_DIR = Path.home()
USR_CONFIG = UserConfig()


@magicgui(
    call_button="Start Round #2 - 3D_Correction",
    layout="vertical",
    load_para={"widget_type": "PushButton", "text": "Load parameters (*.ini)"},
    save_para={"widget_type": "PushButton", "text": "Save parameters (*.ini)"},
    cell_value={"label": "Value of the cell region"},
    mask_value={"label": "Value of the mask region", "visible": False},
    error_info={"widget_type": "TextEdit", "label": "Warnings:", "visible": False},
    image_size={"label": "Image size (segmentation)", "enabled": False},
    path_cells={"label": "Open image sequence: Cell regions (*.tiff):", "mode": "d"},
    path_raw={"label": "Open image sequence: Raw images (*.tiff):", "mode": "d"},
    path_mask={"label": "Open image sequence: Mask images (*.tiff):", "mode": "d", "visible": False},
    path_result={"label": "Open file: segmentation (*.npy):", "mode": "r", "filter": '*.npy'},
    seg_dir={"label": "Open image sequence: segmentation (*.tiff): ", "mode": "d", "visible": False},
    enable_mask={"label": "Use the Mask images", "visible": False},
    enable_cell={"label": "Use the Cell-region images"},
    load_seg_dir={"label": "Use image sequence as segmentation"},
)
def start_r2(
        load_para,
        save_para,
        enable_mask=False,
        enable_cell=False,
        load_seg_dir=False,
        path_cells=CURRENT_DIR,
        path_raw=CURRENT_DIR,
        path_mask=CURRENT_DIR,
        path_result=CURRENT_DIR,
        seg_dir=CURRENT_DIR,
        image_size="",
        cell_value=2,
        mask_value=2,
        error_info="",
):
    """Run some computation."""
    if test_paths_r2():
        cells = load_cells(cell_value, path_cells, file_cached=_npy_name(path_cells)) if enable_cell else None
        images = load_image_lazy(path_raw)
        mask_dilated = load_mask(mask_value, path_mask, file_cached=_npy_name(path_mask)) if enable_mask else None
        segmentation = load_segmentation(seg_dir) if load_seg_dir else load_segmentation(path_result)
        Seg2LinkR2(images, cells, mask_dilated, segmentation, path_result)
        start_r2.close()
        return None


start_r2.error_info.min_height = 70


def test_paths_r2() -> bool:
    msg = check_existence_path(paths_r2())
    if msg:
        show_error_msg(start_r2.error_info, msg)
        return False

    msg = "\n".join([check_tiff_existence(tiff_folders_r2()), check_seg_file()])
    if msg == '\n':
        show_error_msg(start_r2.error_info, "")
        return True
    else:
        show_error_msg(start_r2.error_info, msg)
        return False


def check_seg_file() -> str:
    if not start_r2.path_result.value.name.endswith(".npy"):
        return f'Warning: "{start_r2.path_result.value.name}" is not a .npy file'
    elif not start_r2.path_result.value.exists():
        return f'Warning: File "{start_r2.path_result.value.name}" does not exist'
    else:
        return ""


def paths_r2() -> List[Path]:
    seg_dir = start_r2.seg_dir.value if start_r2.load_seg_dir.value else start_r2.path_result.value
    paths = [start_r2.path_raw.value, seg_dir]
    if start_r2.enable_mask.value:
        paths.insert(-1, start_r2.path_mask.value)
    if start_r2.path_cells.value:
        paths.insert(0, start_r2.path_cells.value)
    return paths


def tiff_folders_r2() -> List[Path]:
    paths = [start_r2.path_raw.value]
    if start_r2.load_seg_dir.value:
        paths.append(start_r2.seg_dir.value)
    if start_r2.enable_mask.value:
        paths.insert(-1, start_r2.path_mask.value)
    if start_r2.path_cells.value:
        paths.insert(0, start_r2.path_cells.value)
    return paths


def load_segmentation(path_seg: Path):
    if path_seg.is_dir():
        print("Caching segmentation... Please wait")
        segmentation = load_image_pil(path_seg)
        np.save(path_seg.parent / "seg_from_img_sequence.npy", segmentation)
        start_r2.path_result.value = path_seg.parent / "seg_from_img_sequence.npy"
    else:
        segmentation = np.load(str(path_seg))

    if segmentation.dtype != parameters.pars.dtype_r2:
        warnings.warn(f"segmentation should has dtype {parameters.pars.dtype_r2}. Transforming...")
        segmentation = segmentation.astype(parameters.pars.dtype_r2, copy=False)
    label_shape = segmentation.shape
    start_r2.image_size.value = f"H: {label_shape[0]}  W: {label_shape[1]}  D: {label_shape[2]}"
    print("Segmentation shape:", label_shape, "dtype:", segmentation.dtype)
    return segmentation


@start_r2.enable_mask.changed.connect
def use_mask():
    visible = start_r2.enable_mask.value
    start_r2.path_mask.visible = visible
    start_r2.mask_value.visible = visible

    msg = check_existence_path(paths_r2())
    show_error_msg(start_r2.error_info, msg)


@start_r2.load_seg_dir.changed.connect
def load_seg_dir():
    visible = start_r2.load_seg_dir.value
    start_r2.path_result.visible = not visible
    start_r2.seg_dir.visible = visible

    msg = check_existence_path(paths_r2())
    show_error_msg(start_r2.error_info, msg)


@start_r2.save_para.changed.connect
def _on_save_para_changed():
    parameters_r1r2 = {"path_cells": start_r2.path_cells.value,
                       "path_raw": start_r2.path_raw.value,
                       "path_mask": start_r2.path_mask.value,
                       "path_result": start_r2.path_result.value.parent,
                       "cell_value": start_r2.cell_value.value,
                       "mask_value": start_r2.mask_value.value}
    parameters_r2 = {"seg_file": start_r2.path_result.value}
    USR_CONFIG.save_ini_r1r2(parameters_r1r2, CURRENT_DIR)
    USR_CONFIG.save_ini_r2(parameters_r2, CURRENT_DIR)


@start_r2.load_para.changed.connect
def _on_load_para_changed():
    try:
        USR_CONFIG.load_ini(CONFIG_DIR)
    except ValueError:
        return
    parameters.pars.set_from_dict(USR_CONFIG.pars.advanced)

    set_pars_r1r2(start_r2, USR_CONFIG.pars.r1r2)
    start_r2.path_result.value = start_r2.path_result.value
    if USR_CONFIG.pars.r2:
        set_pars_r2(USR_CONFIG.pars.r2)
    if USR_CONFIG.pars.r1["use_mask"]=="True":
        start_r2.enable_mask.visible = True


def set_pars_r2(parameters_r2: dict):
    if parameters_r2.get("seg_file"):
        start_r2.path_result.value = parameters_r2["seg_file"]


@start_r2.path_result.changed.connect
def _on_path_result_changed():
    msg = check_existence_path(paths_r2())
    show_error_msg(start_r2.error_info, msg)


@start_r2.seg_dir.changed.connect
def _on_seg_dir_changed():
    msg = check_existence_path(paths_r2())
    show_error_msg(start_r2.error_info, msg)


@start_r2.path_cells.changed.connect
def _on_path_cells_changed():
    if start_r2.path_cells.value.exists():
        new_cwd = start_r2.path_cells.value.parent
        start_r2.path_raw.value = new_cwd
        start_r2.path_mask.value = new_cwd
        start_r2.path_result.value = new_cwd
    msg = check_existence_path(paths_r2())
    show_error_msg(start_r2.error_info, msg)


@start_r2.path_raw.changed.connect
def _on_path_raw_changed():
    msg = check_existence_path(paths_r2())
    show_error_msg(start_r2.error_info, msg)


@start_r2.path_mask.changed.connect
def _on_path_mask_changed():
    msg = check_existence_path(paths_r2())
    show_error_msg(start_r2.error_info, msg)
