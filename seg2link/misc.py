import cProfile
import os
import pstats
import traceback
from inspect import signature
from io import StringIO
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Union, Set, Iterable

from PIL import Image
import numpy as np
from numpy import ndarray
from scipy import ndimage as ndi
from scipy.ndimage import grey_dilation, grey_closing, binary_fill_holes, binary_closing
from skimage.segmentation import relabel_sequential
from dask import delayed
import dask.array as da

from seg2link.parameters import DEBUG
from seg2link import parameters

if parameters.DEBUG:
    pass

def load_image_pil(path: Path) -> ndarray:
    """Load image as ndarray into RAM"""
    paths_list = get_files(path)
    imread = lambda fname: np.array(Image.open(fname))
    sample = imread(paths_list[0])

    img_array = np.zeros((sample.shape[0],sample.shape[1],len(paths_list)), dtype=sample.dtype)
    for z, img_path in enumerate(paths_list):
        img_array[..., z] = np.array(Image.open(img_path))
    return img_array


def load_image_lazy(path: Path) -> ndarray:
    """Lazy imread with dask"""
    paths_list = get_files(path)
    imread = lambda fname: np.array(Image.open(fname))
    sample = imread(paths_list[0])

    lazy_imread = delayed(imread)  # lazy reader
    lazy_arrays = [lazy_imread(fn) for fn in paths_list]
    dask_arrays = [
        da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
        for delayed_reader in lazy_arrays
    ]
    return da.stack(dask_arrays, axis=-1)


def load_array_lazy(path: Path):
    """Lazy array load with dask"""
    mask = np.load(path, mmap_mode="r")
    imread = lambda z: mask[...,z]
    sample = imread(0)

    lazy_imread = delayed(imread)  # lazy reader
    lazy_arrays = [lazy_imread(z) for z in range(mask.shape[2])]
    dask_arrays = [
        da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
        for delayed_reader in lazy_arrays
    ]
    return da.stack(dask_arrays, axis=-1)


def get_files(path: Path) -> List[str]:
    """Return all paths of .tiff or .tif files"""
    return [str(file) for file in sorted(path.glob("*.tif*"))]


def shorten_filename(foldername: str, new_length: int):
    for filename in os.listdir(foldername):
        print(filename[-new_length:])
        os.rename(os.path.join(foldername, filename), os.path.join(foldername, filename[-new_length:]))


def dilation_scipy(label_image: ndarray, filter_size: Tuple[int, int, int]) -> ndarray:
    """3D dilation using scipy, quicker than dilation_opencv"""
    return grey_dilation(label_image, filter_size)


def fill_holes_scipy(label_image: ndarray, filter_size: Tuple[int, int, int]) -> ndarray:
    """fill holes after closing using scipy"""
    print("Closing... Please wait")
    closed_img = grey_closing(label_image, filter_size)
    print("Filling holes... Please wait")
    for z in range(closed_img.shape[2]):
        closed_img[..., z] = binary_fill_holes(closed_img[..., z])
    return closed_img


def add_blank_lines(string: str, max_lines: int) -> str:
    num_lines = string.count("\n") + 1
    if num_lines < max_lines:
        return string + "\n" * (max_lines - num_lines)
    else:
        return string


def print_information(operation: Optional[str] = None, print_errors: bool = True) -> Callable:
    def deco(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                if operation is not None:
                    print(operation)
                func(*args, **kwargs)
            except Exception:
                if print_errors:
                    print(f"!!!Error occurred in {func.__name__}()!!!")
                    print(traceback.format_exc())
                    raise
        return wrapper
    return deco


def flatten_2d_list(labels2d: List[List[int]]) -> Tuple[List[int], List[int]]:
    labels1d = [item for sublist in labels2d for item in sublist]
    label_nums = [len(sublist) for sublist in labels2d]
    return labels1d, label_nums


def make_folder(path_i: Path) -> Path:
    if not path_i.exists():
        path_i.mkdir()
    return path_i


class TinyCells:

    def __init__(self, image3d: ndarray):
        self.label_image = image3d

    def sort_by_areas(self):
        labels, areas = np.unique(self.label_image, return_counts=True)
        idxes_sorted = sorted(range(1, len(labels)), key=lambda i: areas[i], reverse=True)
        self.sorted_labels = labels[idxes_sorted]
        self.sorted_areas = areas[idxes_sorted]

    def min_area(self, max_cell_num: int = 65535) -> Tuple[int, int]:
        """Return the minimum areas and the number of labels to be deleted"""
        if max_cell_num >= len(self.sorted_labels):
            max_area_delete = 0
            num_delete = 0
        else:
            max_area_delete = self.sorted_areas[max_cell_num]
            num_delete = len(self.sorted_labels) - max_cell_num
        return max_area_delete, num_delete

    def remove_and_relabel(self, image3d: ndarray, max_cell_num: Optional[int] = None) -> ndarray:
        maps = np.arange(0, np.max(self.sorted_labels) + 1, dtype=image3d.dtype)
        if max_cell_num is None:
            self.relabel_sorted_labels(maps, self.sorted_labels)
        else:
            maps[np.isin(maps, self.sorted_labels[max_cell_num:])] = 0
            self.relabel_sorted_labels(maps, self.sorted_labels[:max_cell_num])
        return maps[image3d]

    def relabel_sorted_labels(self, maps: ndarray, sorted_labels: ndarray) -> int:
        """Relabel according to areas (descending)"""
        cell_num = len(sorted_labels)
        ori_labels = sorted_labels.tolist()
        tgt_labels = list(range(1, cell_num + 1))
        maps_ = maps.copy()
        for o, t in zip(ori_labels, tgt_labels):
            if o != t:
                maps[maps_ == o] = t


def replace(labels_old: Union[int, Set[int]], label_new: int, array: ndarray) -> ndarray:
    if isinstance(labels_old, set):
        array[np.isin(array, list(labels_old))] = label_new
    else:
        array[array == labels_old] = label_new
    return array


def mask_cells(label_img: ndarray, mask: ndarray, ratio_mask: float) -> ndarray:
    index = np.unique(label_img)
    index = index[index != 0]
    ratio = np.array(ndi.mean(mask, labels=label_img, index=index))

    mask_idx = np.arange(index[-1]+1, dtype=label_img.dtype)
    idx_mask = list(np.where(ratio <= ratio_mask)[0])
    for idx in idx_mask:
        mask_idx[index[idx]] = 0
    return relabel_sequential(mask_idx[label_img])[0]


def get_unused_labels_quick(used_labels: Iterable, max_num: Optional[int] = None) -> List[int]:
    def get_unused_labels_init_quick(labels_array_, label_max_):
        """labels_array_: array of positive int"""
        pointer = 0
        new_labels_ = []
        for label in range(1, label_max_ + 1):
            if labels_array_[pointer] != label:
                new_labels_.append(label)
            else:
                pointer += 1
        return new_labels_

    labels_array = np.unique(list(used_labels))
    labels_array = labels_array[labels_array > 0]
    label_max = labels_array[-1]
    new_labels = get_unused_labels_init_quick(labels_array, label_max)
    if max_num is None:
        return new_labels
    if len(new_labels) >= max_num:
        return new_labels[:max_num]
    else:
        return new_labels + [i for i in range(label_max + 1, label_max + max_num - len(new_labels) + 1)]


def qprofile(func):
    """Print runtime information in a function

    References
    ----------
    Modified from the code here: https://stackoverflow.com/questions/40132630/python-cprofiler-a-function
    Author: Sanket Sudake
    """

    def profiled_func(*args, **kwargs):
        para_num = len(signature(func).parameters)

        if not DEBUG:
            return func() if para_num == 0 else func(*args, **kwargs)

        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func() if para_num == 0 else func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            s = StringIO()
            ps = pstats.Stats(profile, stream=s).strip_dirs().sort_stats('cumulative')
            ps.print_stats(15)
            print(s.getvalue())

    return profiled_func