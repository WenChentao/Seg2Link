from configparser import ConfigParser
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
from magicgui import magicgui
from numpy import ndarray

from seg2link import config
from seg2link.emseg_core import Archive
from seg2link.first_correction import Seg2LinkR1
from seg2link.misc import load_image_pil, dilation_scipy
from seg2link.userconfig import UserConfig

CURRENT_DIR = Path.home()
USR_CONFIG = UserConfig()


def cache_images(func) -> Callable:
    def wrapper(*args, file_cached: Path, **kwargs) -> ndarray:
        print(f"Running {func.__name__} ...")
        if file_cached is None:
            array = func(*args, **kwargs)
        elif file_cached.exists():
            array = np.load(file_cached)
        else:
            array = func(*args, **kwargs)
            print("Saving the image as cache data...")
            np.save(file_cached, array)
            array = np.load(file_cached)
        print("Image shape:", array.shape)
        return array

    return wrapper


@cache_images
def load_raw(path_raw):
    return load_image_pil(path_raw)


@cache_images
def load_cells(cell_value, path_cells):
    return load_image_pil(path_cells) == cell_value


@cache_images
def load_mask(mask_value: int, path_mask: Path) -> ndarray:
    mask_images = load_image_pil(path_mask) == mask_value
    if config.pars.mask_dilate_kernel is None:
        return mask_images
    else:
        return dilation_scipy(mask_images, filter_size=config.pars.mask_dilate_kernel)


@magicgui(
    call_button="Start Seg2Link (1st round)",
    layout="vertical",
    load_para={"widget_type": "PushButton", "text": "Load parameters (*.ini)"},
    save_para={"widget_type": "PushButton", "text": "Save parameters (*.ini)"},
    cell_value={"label": "Value of the cell region"},
    mask_value={"label": "Value of the mask region", "visible": False},
    historical_info={"label": "   Historical info:", "visible": False},
    paths_exist={"visible": False},
    error_info={"label": "   !Following path(s) are not exist:", "visible": False},
    threshold_link={"widget_type": "FloatSlider", "label": "Min_Overlap (linking)", "min": 0.05, "max": 0.95},
    threshold_mask={"widget_type": "FloatSlider", "label": "Min_Overlap (masking)", "min": 0.05, "max": 0.95,
                    "visible": False},
    retrieve_slice={"widget_type": "Slider", "max": 1, "visible": False},
    path_cells={"label": "Folder for cell images (*.tiff):", "mode": "d"},
    path_raw={"label": "Folder for raw images (*.tiff):", "mode": "d"},
    path_mask={"label": "Folder for mask images (*.tiff):", "mode": "d", "visible": False},
    path_cache={"label": "Folder for cached files:", "mode": "d"},
    enable_mask={"label": "Use the mask image"},
    enable_align={"label": "Use the affine alignment"},
)
def widget_entry1(
        load_para,
        save_para,
        enable_mask=False,
        enable_align=False,
        path_cells=CURRENT_DIR,
        path_raw=CURRENT_DIR,
        path_mask=CURRENT_DIR,
        path_cache=CURRENT_DIR,
        paths_exist=["", "", "", ""],
        error_info="",
        historical_info="",
        retrieve_slice=0,
        cell_value=2,
        mask_value=2,
        threshold_link=0.5,
        threshold_mask=0.8,
):
    """Run some computation."""
    cells = load_cells(cell_value, path_cells, file_cached=_npy_name(path_cells))
    images = load_raw(path_raw, file_cached=_npy_name(path_raw))
    if enable_mask:
        mask_dilated = load_mask(mask_value, path_mask, file_cached=_npy_name(path_mask))
    else:
        mask_dilated = None
    layer_num = cells.shape[2]
    Seg2LinkR1(images, cells, mask_dilated, layer_num, path_cache, threshold_link, threshold_mask,
               widget_entry1.retrieve_slice.value, enable_align)
    return None


def _npy_name(path_cells: Path, addi_str: str = "") -> Path:
    return Path(*path_cells.parts[:-1], path_cells.parts[-1] + addi_str + ".npy")


@widget_entry1.enable_mask.changed.connect
def use_mask():
    visible = widget_entry1.enable_mask.value
    widget_entry1.path_mask.visible = visible
    widget_entry1.threshold_mask.visible = visible
    widget_entry1.mask_value.visible = visible


@widget_entry1.save_para.changed.connect
def _on_save_para_changed():
    parameters_r1 = {"path_cells": widget_entry1.path_cells.value,
                     "path_raw": widget_entry1.path_raw.value,
                     "path_mask": widget_entry1.path_mask.value,
                     "path_cache": widget_entry1.path_cache.value,
                     "cell_value": widget_entry1.cell_value.value,
                     "mask_value": widget_entry1.mask_value.value,
                     "threshold_link": widget_entry1.threshold_link.value,
                     "threshold_mask": widget_entry1.threshold_mask.value}
    USR_CONFIG.save_ini_r1(parameters_r1)


@widget_entry1.load_para.changed.connect
def _on_load_para_changed():
    USR_CONFIG.load_ini()
    parameters_r1 = USR_CONFIG.pars.r1
    config.pars.set_from_dict(USR_CONFIG.pars.advanced)
    widget_entry1.path_cells.value = parameters_r1["path_cells"]
    widget_entry1.path_raw.value = parameters_r1["path_raw"]
    widget_entry1.path_mask.value = parameters_r1["path_mask"]
    widget_entry1.path_cache.value = parameters_r1["path_cache"]
    widget_entry1.cell_value.value = int(parameters_r1["cell_value"])
    widget_entry1.mask_value.value = int(parameters_r1["mask_value"])
    widget_entry1.threshold_link.value = float(parameters_r1["threshold_link"])
    widget_entry1.threshold_mask.value = float(parameters_r1["threshold_mask"])


def set_path_error_info(widget_entry1_, num: int, error: bool):
    num_str = {1: "Cell", 2: "Raw", 3: "Mask", 4: "Cache"}
    update_error_info(error, num, num_str, widget_entry1_)


@widget_entry1.path_cache.changed.connect
def _on_path_cache_changed():
    if widget_entry1.path_cache.value.exists():
        latest_slice = Archive(emseg=None, path_save=widget_entry1.path_cache.value).latest_slice
        s1 = 1 if latest_slice >= 1 else 0
        widget_entry1.historical_info.value = f"Segmented slices: {s1}-{latest_slice} / Restart: 0"
        widget_entry1.historical_info.visible = True
        widget_entry1.retrieve_slice.label = f"Retrieve a previous slice"
        widget_entry1.retrieve_slice.max = latest_slice
        widget_entry1.retrieve_slice.value = latest_slice
        widget_entry1.retrieve_slice.visible = True
        set_path_error_info(widget_entry1, 4, False)
    else:
        set_path_error_info(widget_entry1, 4, True)


@widget_entry1.path_cells.changed.connect
def _on_path_cells_changed():
    if widget_entry1.path_cells.value.exists():
        new_cwd = widget_entry1.path_cells.value.parent
        widget_entry1.path_raw.value = new_cwd
        widget_entry1.path_mask.value = new_cwd
        set_path_error_info(widget_entry1, 1, False)
    else:
        set_path_error_info(widget_entry1, 1, True)


@widget_entry1.path_raw.changed.connect
def _on_path_raw_changed():
    if widget_entry1.path_raw.value.exists():
        set_path_error_info(widget_entry1, 2, False)
    else:
        set_path_error_info(widget_entry1, 2, True)


@widget_entry1.path_mask.changed.connect
def _on_path_mask_changed():
    if widget_entry1.path_mask.value.exists():
        set_path_error_info(widget_entry1, 3, False)
    else:
        set_path_error_info(widget_entry1, 3, True)


def update_error_info(error, num, num_str, widget_entry):
    if error:
        widget_entry.paths_exist.value[num - 1] = num_str[num]
        widget_entry.error_info.value = ",  ".join([s for s in widget_entry.paths_exist.value if len(s) > 0])
        widget_entry.error_info.visible = True
    else:
        widget_entry.paths_exist.value[num - 1] = ""
        widget_entry.error_info.value = ",  ".join([s for s in widget_entry.paths_exist.value if len(s) > 0])

