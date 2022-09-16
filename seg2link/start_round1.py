from pathlib import Path
from typing import Callable, List

import numpy as np
from magicgui import magicgui
from numpy import ndarray

from seg2link import parameters
from seg2link.seg2dlink_core import Archive
from seg2link.seg2link_round1 import Seg2LinkR1
from seg2link.misc import load_image_pil, load_image_lazy, load_array_lazy, fill_holes_scipy
from seg2link.userconfig import UserConfig, get_config_dir

try:
    CONFIG_DIR = get_config_dir()
except Exception:
    CONFIG_DIR = Path.home()

CURRENT_DIR = Path.home()
USR_CONFIG = UserConfig()


def cache_images_lazy(func) -> Callable:
    def wrapper(*args, file_cached: Path, **kwargs) -> ndarray:
        if file_cached is None:
            array = func(*args, **kwargs)
        elif file_cached.exists():
            array = load_array_lazy(file_cached)
        else:
            print("Caching data... Please wait")
            array = func(*args, **kwargs)
            np.save(file_cached, array)
            array = load_array_lazy(file_cached)
        print("Image shape:", array.shape)
        return array

    return wrapper


@cache_images_lazy
def load_cells(cell_value, path_cells):
    return load_image_pil(path_cells) == cell_value


@cache_images_lazy
def load_mask(mask_value: int, path_mask: Path, fill_holes: bool) -> ndarray:
    mask_images = load_image_pil(path_mask) == mask_value
    if not mask_images.any():
        raise ValueError("No cell region found in Mask images. Check if the value for mask regions is correct!")
    if fill_holes:
        return fill_holes_scipy(mask_images, filter_size=parameters.pars.mask_dilate_kernel)
    else:
        return mask_images


def show_error_msg(widget_error_state, msg):
    widget_error_state.show()
    widget_error_state.value = msg


@magicgui(
    call_button="Start Round #1 - Seg2D + Link",
    layout="vertical",
    load_para={"widget_type": "PushButton", "text": "Load parameters (*.ini)"},
    save_para={"widget_type": "PushButton", "text": "Save parameters (*.ini)"},
    cell_value={"label": "Value of the cell region"},
    mask_value={"label": "Value of the mask region", "visible": False},
    historical_info={"label": "   Historical info:", "visible": False},
    error_info={"widget_type": "TextEdit", "label": "Warnings:", "visible": False},
    threshold_link={"widget_type": "FloatSlider", "label": "Min_Overlap (linking)", "min": 0.05, "max": 0.95},
    threshold_mask={"widget_type": "FloatSlider", "label": "Min_Overlap (masking)", "min": 0.05, "max": 0.95,
                    "visible": False},
    retrieve_slice={"widget_type": "Slider", "max": 1, "visible": False},
    path_cells={"label": "Open image sequences: Cell regions (*.tiff):", "mode": "d"},
    path_raw={"label": "Open image sequences: Raw images (*.tiff):", "mode": "d"},
    path_mask={"label": "Open image sequences: Mask images (*.tiff):", "mode": "d", "visible": False},
    path_result={"label": "Select a folder for storing results:", "mode": "d"},
    enable_mask={"label": "Use the Mask images"},
    enable_fill_holes={"label": "Fill holes", "visible": False},
    enable_update_mask={"label": "Update cache of mask", "visible": False},
)
def start_r1(
        load_para,
        save_para,
        enable_mask=False,
        enable_fill_holes=False,
        enable_update_mask=False,
        path_cells=CURRENT_DIR,
        path_raw=CURRENT_DIR,
        path_mask=CURRENT_DIR,
        path_result=CURRENT_DIR,
        historical_info="",
        retrieve_slice=0,
        cell_value=1,
        mask_value=1,
        threshold_link=0.5,
        threshold_mask=0.8,
        error_info="",
):
    """Run some computation."""
    if test_paths_r1():
        print("Loading cell image... Please wait")
        cells = load_cells(cell_value, path_cells, file_cached=_npy_name(path_cells))
        print("Loading raw image... Please wait")
        images = load_image_lazy(path_raw)
        if enable_mask:
            print("Loading mask image... Please wait")
            if enable_update_mask:
                delete_npy(path_mask)
            mask_dilated = load_mask(mask_value, path_mask, enable_fill_holes, file_cached=_npy_name(path_mask))
        else:
            mask_dilated = None
        layer_num = cells.shape[2]
        print("Initiating the soft... Please wait")
        Seg2LinkR1(images, cells, mask_dilated, enable_mask, layer_num, path_result, threshold_link, threshold_mask,
                   start_r1.retrieve_slice.value)
        print("The soft was started")
        start_r1.close()


def test_paths_r1() -> bool:
    msg = check_existence_path(folders_r1())
    if msg:
        show_error_msg(start_r1.error_info, msg)
        return False
    else:
        msg = check_tiff_existence(tiff_folders_r1())
        if msg:
            show_error_msg(start_r1.error_info, msg)
            return False
        show_error_msg(start_r1.error_info, "")
        return True


def delete_npy(path_folder):
    _npy_name(path_folder).unlink(missing_ok=True)


start_r1.error_info.min_height = 70


def check_existence_path(paths_list: List[Path]) -> str:
    msg = []
    for path in paths_list:
        if not path.exists():
            if path.name.endswith(".npy"):
                msg.append(f'Warning: File "{path.name}" does not exist')
            else:
                msg.append(f'Warning: Folder "{path.name}" does not exist')
    return "\n".join(msg)


def check_tiff_existence(paths_list: List[Path]) -> str:
    msg = []
    for path in paths_list:
        if not path.name.endswith(".npy") and not list(path.glob("*.tif*")):
            msg.append(f'Warning: Folder "{path.name}" includes no TIFF files')
    return "\n".join(msg)


def _npy_name(path_cells: Path, addi_str: str = "") -> Path:
    return Path(*path_cells.parts[:-1], path_cells.parts[-1] + addi_str + ".npy")


@start_r1.enable_mask.changed.connect
def use_mask():
    visible = start_r1.enable_mask.value
    start_r1.enable_fill_holes.visible = visible
    start_r1.enable_update_mask.visible = visible
    start_r1.path_mask.visible = visible
    start_r1.threshold_mask.visible = visible
    start_r1.mask_value.visible = visible

    test_paths_r1()


def enable_update_mask():
    start_r1.enable_update_mask.value = True


start_r1.enable_fill_holes.changed.connect(enable_update_mask)


@start_r1.save_para.changed.connect
def _on_save_para_changed():
    parameters_r1 = {"use_mask": start_r1.enable_mask.value,
                     "use_fill_holes": start_r1.enable_fill_holes.value,
                     "threshold_link": start_r1.threshold_link.value,
                     "threshold_mask": start_r1.threshold_mask.value}
    parameters_r1r2 = {"path_cells": start_r1.path_cells.value,
                       "path_raw": start_r1.path_raw.value,
                       "path_mask": start_r1.path_mask.value,
                       "path_result": start_r1.path_result.value,
                       "cell_value": start_r1.cell_value.value,
                       "mask_value": start_r1.mask_value.value}
    USR_CONFIG.save_ini_r1(parameters_r1, CURRENT_DIR)
    USR_CONFIG.save_ini_r1r2(parameters_r1r2, CURRENT_DIR)


@start_r1.load_para.changed.connect
def _on_load_para_changed():
    try:
        USR_CONFIG.load_ini(CONFIG_DIR)
    except ValueError:
        return

    start_r1.enable_fill_holes.changed.disconnect()
    set_pars_r1r2(start_r1, USR_CONFIG.pars.r1r2)
    set_pars_r1(USR_CONFIG.pars.r1)
    start_r1.enable_fill_holes.changed.connect(enable_update_mask)

    parameters.pars.set_from_dict(USR_CONFIG.pars.advanced)


def set_pars_r1r2(mgui, parameters_r1r2: dict):
    mgui.path_cells.value = parameters_r1r2["path_cells"]
    mgui.path_raw.value = parameters_r1r2["path_raw"]
    mgui.path_mask.value = parameters_r1r2["path_mask"]
    mgui.path_result.value = parameters_r1r2["path_result"]
    mgui.cell_value.value = int(parameters_r1r2["cell_value"])
    mgui.mask_value.value = int(parameters_r1r2["mask_value"])


def set_pars_r1(parameters_r1: dict):
    start_r1.enable_mask.value = parameters_r1["use_mask"] == "True"
    start_r1.enable_fill_holes.value = parameters_r1["use_fill_holes"] == "True"
    start_r1.threshold_link.value = float(parameters_r1["threshold_link"])
    start_r1.threshold_mask.value = float(parameters_r1["threshold_mask"])


@start_r1.path_result.changed.connect
def _on_path_result_changed():
    if start_r1.path_result.value.exists():
        latest_slice = Archive(emseg1=None, path_save=start_r1.path_result.value).latest_slice
        s1 = 1 if latest_slice >= 1 else 0
        start_r1.historical_info.value = f"Segmented slices: {s1}-{latest_slice} / Restart: 0"
        start_r1.historical_info.visible = True
        start_r1.retrieve_slice.label = f"Retrieve a previous slice"
        start_r1.retrieve_slice.max = latest_slice
        start_r1.retrieve_slice.value = latest_slice
        start_r1.retrieve_slice.visible = True
    test_paths_r1()



@start_r1.path_cells.changed.connect
def _on_path_cells_changed():
    if start_r1.path_cells.value.exists():
        global CURRENT_DIR
        CURRENT_DIR = start_r1.path_cells.value.parent
        start_r1.path_raw.value = CURRENT_DIR
        start_r1.path_mask.value = CURRENT_DIR
        start_r1.path_result.value = CURRENT_DIR
    test_paths_r1()


def folders_r1() -> List[Path]:
    if start_r1.enable_mask.value:
        return [start_r1.path_cells.value,
                start_r1.path_raw.value,
                start_r1.path_mask.value,
                start_r1.path_result.value]
    else:
        return [start_r1.path_cells.value,
                start_r1.path_raw.value,
                start_r1.path_result.value]


def tiff_folders_r1() -> List[Path]:
    if start_r1.enable_mask.value:
        return [start_r1.path_cells.value,
                start_r1.path_raw.value,
                start_r1.path_mask.value]
    else:
        return [start_r1.path_cells.value,
                start_r1.path_raw.value]


@start_r1.path_raw.changed.connect
def _on_path_raw_changed():
    test_paths_r1()


@start_r1.path_mask.changed.connect
def _on_path_mask_changed():
    test_paths_r1()


