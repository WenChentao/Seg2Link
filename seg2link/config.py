# Cache length
from typing import Optional, Tuple

debug = True

label_filename_v1 = 'label%04i.pickle' # Deprecated. Will be removed in future
label_filename_v2 = 'labels_list_%04i.pickle'
re_filename_v1 = r'label\d+.pickle' # Deprecated. Will be removed in future
re_filename_v2 = r'labels_list_\d+.pickle'

cache_length_first = 10
cache_length_second = 5

raw_bit = 8

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


