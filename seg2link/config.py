# Cache length
import cProfile
import pstats
from inspect import signature
from io import StringIO
from typing import Optional, Tuple
import numpy as np

debug = True

if debug:
    import atexit
    import line_profiler
    lprofile = line_profiler.LineProfiler()
    atexit.register(lprofile.print_stats)

label_filename_v1 = 'label%04i.pickle' # Deprecated. Will be removed in future
label_filename_v2 = 'labels_list_%04i.pickle'
re_filename_v1 = r'label\d+.pickle' # Deprecated. Will be removed in future
re_filename_v2 = r'labels_list_\d+.pickle'

cache_length_first = 10
cache_length_second = 5

raw_bit = 8

dtype_r2 = np.uint16
upper_limit_labels_r2 = 64000
upper_limit_r1_export = upper_limit_labels_r2 - 10

# Visualization
max_draw_layers = 100
scale_xyz = (1, 1, 10)

# HotKeys
key_add = 'a'
key_clean = 'c'
key_merge = 'm'

key_delete = 'd'

key_undo = 'u'
key_redo = 'f'

key_r1_next = 'Shift-n'
key_r1_reseg_link = 'r'

key_separate = 'k'

key_switch_one_label_all_labels = 'q'

key_online_help = 'h'

# Segmentation
h_watershed = 5

# For removing boundary. Kernel along x, y, z axis. unit: voxels
labels_expand_kernel = (3, 3, 1)

# Used for dilation of the mask image along x, y and z axis. if None, skip the dilation.
mask_dilate_kernel: Optional[Tuple[int, int, int]] = (25, 25, 7)


def qprofile(func):
    """Print runtime information in a function

    References
    ----------
    Modified from the code here: https://stackoverflow.com/questions/40132630/python-cprofiler-a-function
    Author: Sanket Sudake
    """

    def profiled_func(*args, **kwargs):
        para_num = len(signature(func).parameters)

        if not debug:
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

