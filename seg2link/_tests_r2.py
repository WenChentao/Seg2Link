import copy
from typing import Callable, TYPE_CHECKING

import numpy as np

from seg2link import parameters
from seg2link.misc import get_unused_labels_quick

if TYPE_CHECKING:
    from seg2link.seg2link_round2 import Seg2LinkR2


def test_merge_r2(emseg2: "Seg2LinkR2"):
    """Merge will generate unused_labels except for the minimum (target) label"""
    def deco(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if parameters.DEBUG:
                merge_list = copy.deepcopy(emseg2.label_list)
                unused_labels_old = emseg2.cache_bbox.cal_unused_labels()

                func(*args, **kwargs)
                labels_new = list(emseg2.cache_bbox.bbox.keys())
                unused_labels_new = emseg2.cache_bbox.cal_unused_labels()

                for label in merge_list:
                    assert label not in unused_labels_old, f"label {label} should not be in unused_labels_old"
                    if label == min(merge_list):
                        assert label not in unused_labels_new, f"label {label} should not be in unused_labels_new"
                    else:
                        assert label in unused_labels_new or label > np.max(labels_new), \
                            f"label {label} should be in unused_labels_new"
                print("Unused labels:", emseg2.cache_bbox.unused_labels)
                print("Merge test was passed")
            else:
                func(*args, **kwargs)
        return wrapper
    return deco


def test_delete_r2(emseg2: "Seg2LinkR2"):
    """Delete label(s) will generate new unused_labels"""
    def deco(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if parameters.DEBUG:
                unused_labels_old = emseg2.cache_bbox.cal_unused_labels()
                if emseg2.label_list:
                    delete_list = copy.deepcopy(emseg2.label_list)
                else:
                    delete_list = [emseg2.vis.viewer.layers["segmentation"].selected_label]
                func(*args, **kwargs)
                labels_new = list(emseg2.cache_bbox.bbox.keys())
                unused_labels_new = emseg2.cache_bbox.cal_unused_labels()

                for label in delete_list:
                    assert label not in unused_labels_old, f"label {label} should not be in unused_labels_old"
                    assert label in unused_labels_new or label > np.max(labels_new), \
                        f"label {label} should be in unused_labels_new"
                print("Unused labels:", emseg2.cache_bbox.unused_labels)
                print("Delete test was passed")
            else:
                func(*args, **kwargs)
        return wrapper
    return deco


def test_divide_r2(emseg2: "Seg2LinkR2"):
    """The divided labels should use the unused labels"""
    def deco(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if parameters.DEBUG:
                label_ori = emseg2.vis.viewer.layers["segmentation"].selected_label
                labels_old = list(emseg2.cache_bbox.bbox.keys())
                bbox_label_ori = emseg2.cache_bbox.get_subregion_3d(label_ori)[0]
                seg_old = emseg2.vis.viewer.layers["segmentation"].data[bbox_label_ori].copy()
                unused_labels_old = emseg2.cache_bbox.cal_unused_labels()

                func(*args, **kwargs)

                seg_new = emseg2.vis.viewer.layers["segmentation"].data[bbox_label_ori].copy()
                unused_labels_new = emseg2.cache_bbox.cal_unused_labels()
                labels_divided, c = np.unique(seg_new[seg_old == label_ori], return_counts=True)
                labels_divided_set = set(labels_divided[labels_divided != 0])
                labels_divided_set = labels_divided_set.difference(set(labels_old))

                expected_labels = get_unused_labels_quick(labels_old, len(labels_divided_set))

                assert labels_divided_set == set(expected_labels), \
                    f"labels_divided_set:{labels_divided_set}, expected_labels: {expected_labels}"
                assert label_ori not in unused_labels_old, f"label {label_ori} should not be in unused_labels_old"
                for label in labels_divided_set:
                    assert label not in unused_labels_new, f"label {label} should not be in unused_labels_new"
                print("Unused labels:", emseg2.cache_bbox.unused_labels)
                print("Divide test was passed")
            else:
                func(*args, **kwargs)
        return wrapper
    return deco

