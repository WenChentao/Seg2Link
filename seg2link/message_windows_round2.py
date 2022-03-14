from magicgui import magicgui

from seg2link import parameters


@magicgui(
    call_button=False,
    layout="vertical",
    info={"widget_type": "LineEdit", "label": "Cell number"},
    max_cell_num={"widget_type": "Slider", "label": "Max cell number", "min": 1, "max": 10 ** 5, "value": 65535},
    removed_cell_size={"widget_type": "LineEdit", "label": "Remove tiny cells"},
    save_button={"widget_type": "PushButton", "text": "Save the sorted cells (.npy)"},
    cancel_button={"widget_type": "PushButton", "text": "Cancel"},
)
def sort_remove_window(
        info="",
        max_cell_num=65535,
        removed_cell_size="",
        save_button=False,
        cancel_button=False
):
    """Run some computation."""
    return None


@magicgui(
    call_button=False,
    layout="vertical",
    info={"widget_type": "Label", "label": "Warning"},
    ok_button={"widget_type": "PushButton", "text": "OK"},
)
def message_delete_labels(
        info="",
        ok_button=False,
):
    """Run some computation."""
    return None

