from magicgui import magicgui

from seg2link.entry_r1 import start_r1
from seg2link.entry_r2 import start_r2


@magicgui(
    auto_call=True,
    layout="verticapl",
    title={"widget_type": "Label", "label": "Seg2Link:"},
    round1={"widget_type": "PushButton", "label": "Round #1: Segment and link"},
    round2={"widget_type": "PushButton", "label": "Round #2: Overall correction"},
)
def widget_entry(title="Please select a round",round1=True, round2=True):
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

