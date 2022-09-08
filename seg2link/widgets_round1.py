from typing import Tuple, TYPE_CHECKING

from PyQt5.QtWidgets import QApplication
from magicgui import widgets
from magicgui.widgets import Container

from seg2link import parameters
from seg2link.misc import add_blank_lines

if parameters.DEBUG:
    pass

if TYPE_CHECKING:
    from seg2link.seg2link_round1 import VisualizePartial


class WidgetsR1:
    def __init__(self, vis: "VisualizePartial", img_shape: Tuple):
        self.viewer = vis.viewer
        self.emseg1 = vis.emseg1

        shape_str = f"H: {img_shape[0]}  W: {img_shape[1]}  D: {img_shape[2]}"
        self.image_size = widgets.LineEdit(label="Image shape", value=shape_str, enabled=False)
        self.max_label = widgets.LineEdit(label="Largest label", enabled=False)
        self.cached_action = widgets.TextEdit(label="Cached actions",
                                              tooltip=f"Less than {parameters.pars.cache_length_r1} action can be cached",
                                              enabled=True)
        self.label_list_msg = widgets.LineEdit(label="Label list", enabled=False)

        self.hotkeys_info_value = '[Shift + N]: Go to the next slice' \
                                  '\n---------------' \
                                  '\n[K]:  Divide a label in the last slice (LS)' \
                                  '\n[R]:  Divide + Re-link a label in slice LS' \
                                  '\n---------------' \
                                  '\n[A]: Add a label into the label list (LL)' \
                                  '\n[C]: Clear LL' \
                                  '\n[M]: Merge labels in LL' \
                                  '\n[D]: Delete a selected label or labels in LL' \
                                  '\n ---------------' \
                                  '\n[Q]: Switch: Viewing one label | all labels' \
                                  '\n ---------------' \
                                  '\n[U]: Undo     [F]: Redo' \
                                  '\n[L]:  Picker   [E]: Eraser' \
                                  '\n[H]: Online Help'

        self.hotkeys_info = widgets.Label(value=self.hotkeys_info_value)

        self.export_button = widgets.PushButton(text="Export segmentation as .npy file")
        self.state_info = widgets.Label(value="")

        self.add_widgets()
        QApplication.processEvents()

    def show_state_info(self, info: str):
        self.state_info.value = info
        print(info)
        QApplication.processEvents()

    def add_widgets(self):
        container_states = Container(widgets=[self.image_size, self.max_label, self.cached_action, self.label_list_msg])
        container_export = Container(widgets=[self.export_button])
        container_states.min_height = 310
        self.viewer.window.add_dock_widget(container_states, name="States", area="right")
        self.viewer.window.add_dock_widget([self.hotkeys_info], name="HotKeys", area="right")
        self.viewer.window.add_dock_widget(container_export, name="Save/Export", area="right")
        self.viewer.window.add_dock_widget([self.state_info], name="State info", area="right")

    def update_max_actions_labelslist(self):
        self.max_label.value = str(self.emseg1.labels.max_label)
        self.cached_action.value = add_blank_lines("".join(self.emseg1.cache.cached_actions),
                                                   parameters.pars.cache_length_r1 + 1)
        self.label_list_msg.value = tuple(self.emseg1.label_list)
        return None
