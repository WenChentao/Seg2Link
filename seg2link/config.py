import ast
from dataclasses import dataclass, fields
from typing import Optional, Tuple

import numpy as np

DEBUG = False

if DEBUG:
    import atexit
    import line_profiler
    lprofile = line_profiler.LineProfiler()
    atexit.register(lprofile.print_stats)


@dataclass
class Seg2LinkPara:
    # Cache
    cache_length_r1: int = 10
    cache_length_r2: int = 5

    # Data
    raw_bit: int = 8
    seg_bit_r2: int = 16
    upper_limit_labels_r2: int = 64000

    # Visualization
    max_draw_layers_r1: int = 100
    scale_xyz: Tuple[int, int, int] = (1, 1, 10)

    # Segmentation
    h_watershed: int = 5
    # For adding boundary. '2D' or '3D'
    add_boundary_mode: str = '2D'
    # For removing boundary. Kernel along x, y, z axis. unit: voxels
    labels_dilate_kernel_r2: Tuple[int, int, int] = (3, 3, 1)
    # Used for dilation of the mask image along x, y and z axis. if None, skip the dilation.
    mask_dilate_kernel: Optional[Tuple[int, int, int]] = (25, 25, 7)

    # HotKeys
    key_add: str = 'a'
    key_clean: str = 'c'
    key_merge: str = 'm'
    key_delete: str = 'd'
    key_undo: str = 'u'
    key_redo: str = 'f'
    key_next_r1: str = 'Shift-n'
    key_reseg_link_r1: str = 'r'
    key_separate: str = 'k'
    key_insert: str = 'i'
    key_switch_one_label_all_labels: str = 'q'
    key_online_help: str = 'h'

    @property
    def dtype_r2(self):
        if self.seg_bit_r2 == 16:
            return np.uint16
        elif self.seg_bit_r2 == 32:
            return np.uint32
        else:
            raise TypeError("seg_bit_r2 should be set as 16 or 32")

    @property
    def upper_limit_export_r1(self) -> int:
        return self.upper_limit_labels_r2 - 10

    @property
    def all_attributes(self):
        return vars(self)

    def set_from_dict(self, kwargs: dict):
        field_types = {field.name: field.type for field in fields(self)}
        for key, value in kwargs.items():
            if hasattr(self, key):
                value_ = value if field_types[key] == str else ast.literal_eval(value)
                setattr(self, key, value_)


pars = Seg2LinkPara()

label_filename_v1 = 'label%04i.pickle'  # Deprecated. Will be removed in future
label_filename_v2 = 'labels_list_%04i.pickle'
re_filename_v1 = r'label\d+.pickle'  # Deprecated. Will be removed in future
re_filename_v2 = r'labels_list_\d+.pickle'


