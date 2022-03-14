import copy
from typing import Callable, TYPE_CHECKING

import numpy as np

from seg2link import parameters
from seg2link.misc import flatten_2d_list, get_unused_labels_quick

if TYPE_CHECKING:
    from seg2link.seg2link_round1 import Seg2LinkR1


def test_merge_r1(emseg1: "Seg2LinkR1"):
    """Merge will generate unused_labels except for the minimum (target) label"""
    def deco(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if parameters.DEBUG:
                seg_old = emseg1.vis.viewer.layers["segmentation"].data.copy()
                labels_old = np.array(flatten_2d_list(emseg1.labels._labels)[0])
                merge_list = copy.deepcopy(emseg1.label_list)
                unused_labels_old = emseg1.labels.cal_unused_labels()

                func(*args, **kwargs)
                seg_new = emseg1.vis.viewer.layers["segmentation"].data.copy()
                labels_new = np.array(flatten_2d_list(emseg1.labels._labels)[0])
                unused_labels_new = emseg1.labels.cal_unused_labels()

                for label in merge_list:
                    assert np.all(labels_new[labels_old == label] == min(merge_list)), \
                        f"Label{label} was not merged correctly (Label part)"
                    assert np.all(seg_new[seg_old == label] == min(merge_list)), \
                        f"Label{label} was not merged correctly (Label -> Segmentation part)"
                    assert label not in unused_labels_old, f"label {label} should not be in unused_labels_old"
                    if label == min(merge_list):
                        assert label not in unused_labels_new, f"label {label} should not be in unused_labels_new"
                    else:
                        assert label in unused_labels_new or label > np.max(labels_new), \
                            f"label {label} should be in unused_labels_new"
                print("Unused labels:", emseg1.labels.unused_labels)
                print("Merge test was passed")
            else:
                func(*args, **kwargs)
        return wrapper
    return deco


def test_delete_r1(emseg1: "Seg2LinkR1"):
    """Delete label(s) will generate new unused_labels"""
    def deco(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if parameters.DEBUG:
                seg_old = emseg1.vis.viewer.layers["segmentation"].data.copy()
                labels_old = np.array(flatten_2d_list(emseg1.labels._labels)[0])
                unused_labels_old = emseg1.labels.cal_unused_labels()
                if emseg1.label_list:
                    delete_list = copy.deepcopy(emseg1.label_list)
                else:
                    delete_list = [emseg1.vis.viewer.layers["segmentation"].selected_label]
                func(*args, **kwargs)
                seg_new = emseg1.vis.viewer.layers["segmentation"].data.copy()
                labels_new = np.array(flatten_2d_list(emseg1.labels._labels)[0])
                unused_labels_new = emseg1.labels.cal_unused_labels()

                for label in delete_list:
                    assert np.all(labels_new[labels_old == label] == 0), \
                        f"Label{label} was not deleted correctly (Label part)"
                    assert np.all(seg_new[seg_old == label] == 0), \
                        f"Label{label} was not deleted correctly (Label -> Segmentation part)"
                    assert label not in unused_labels_old, f"label {label} should not be in unused_labels_old"
                    assert label in unused_labels_new or label > np.max(labels_new), \
                        f"label {label} should be in unused_labels_new"
                print("Unused labels:", emseg1.labels.unused_labels)
                print("Delete test was passed")
            else:
                func(*args, **kwargs)
        return wrapper
    return deco


def test_divide_r1(emseg1: "Seg2LinkR1"):
    """The divided labels should use the unused labels"""
    def deco(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if parameters.DEBUG:
                label_ori = emseg1.vis.viewer.layers["segmentation"].selected_label

                slice = emseg1.current_slice - emseg1.vis.get_slice(emseg1.current_slice).start - 1
                seg_old = emseg1.vis.viewer.layers["segmentation"].data[..., slice].copy()
                labels_old = np.array(flatten_2d_list(emseg1.labels._labels)[0])
                unused_labels_old = emseg1.labels.cal_unused_labels()

                func(*args, **kwargs)

                seg_new = emseg1.vis.viewer.layers["segmentation"].data[..., slice].copy()
                unused_labels_new = emseg1.labels.cal_unused_labels()
                labels_divided = np.unique(seg_new[seg_old == label_ori])
                labels_divided_ = labels_divided[labels_divided != 0]
                expected_labels = get_unused_labels_quick(labels_old, len(labels_divided_))

                assert set(labels_divided_) == set(expected_labels), \
                    f"labels_divided_:{labels_divided_}, expected_labels: {expected_labels}"
                assert label_ori not in unused_labels_old, f"label {label_ori} should not be in unused_labels_old"
                for label in labels_divided_:
                    assert label not in unused_labels_new, f"label {label} should not be in unused_labels_new"
                print("Unused labels:", emseg1.labels.unused_labels)
                print("Divide test was passed")
            else:
                func(*args, **kwargs)
        return wrapper
    return deco


def test_link_r1(emseg1: "Seg2LinkR1"):
    """Link will use the unused_labels"""
    def deco(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if parameters.DEBUG:
                labels_old = np.array(flatten_2d_list(emseg1.labels._labels)[0])
                func(*args, **kwargs)
                labels_new = np.array(flatten_2d_list(emseg1.labels._labels)[0])

                sorted_labels_new = np.unique(labels_new)
                sorted_labels_new = sorted_labels_new[sorted_labels_new > 0]
                assert len(sorted_labels_new) == sorted_labels_new[-1] or sorted_labels_new[-1] == np.max(labels_old), \
                    f"labels_before_link:{np.unique(labels_old)}, labels_after_link: {sorted_labels_new}"
                print("Link test was passed")
            else:
                func(*args, **kwargs)
        return wrapper
    return deco