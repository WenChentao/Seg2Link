from pathlib import Path

from magicgui import magicgui

from seg2link.start_round1 import start_r1
from seg2link.start_round2 import start_r2

logo = Path(__file__).parent / 'icon_small.png'


@magicgui(
    auto_call=True,
    layout="verticapl",
    title={"widget_type": "Label", "label": f'<h1><img src="{logo}"></h1>'},
    round1={"widget_type": "PushButton", "label": "Round #1 - Seg2D + Link"},
    round2={"widget_type": "PushButton", "label": "Round #2 - 3D_Correction"},
)
def widget_entry(title, round1=False, round2=False):
    return None


@widget_entry.round1.changed.connect
def r1_changed():
    start_r1.show(run=True)
    widget_entry.close()


@widget_entry.round2.changed.connect
def r2_changed():
    start_r2.show(run=True)
    widget_entry.close()


def main():
    widget_entry.show(run=True)


if __name__ == "__main__":
    main()

