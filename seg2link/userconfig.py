from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path

from magicgui import use_app
from magicgui.types import FileDialogMode

from seg2link import parameters

CURRENT_DIR = Path.home()


@dataclass
class Pars:
    r1: dict
    r2: dict
    advanced: dict


@dataclass
class UserConfig:
    ini_path: str = None
    pars: Pars = Pars({}, {}, parameters.pars.all_attributes)

    def load_ini(self, current_dir):
        mode_ = FileDialogMode.EXISTING_FILE
        path = use_app().get_obj("show_file_dialog")(
            mode_,
            caption="Load ini",
            start_path=str(current_dir),
            filter='*.ini'
        )
        if path:
            config_ = ConfigParser()
            config_.read(path)

            self.pars = Pars(
                r1=dict(config_["parameters_r1"]),
                r2=dict(config_["parameters_r2"]),
                advanced=dict(config_["advanced_parameters"])
            )
            self.ini_path = path
        else:
            raise ValueError("No folder selected")

    def save_ini_r1(self, pars_r1, current_dir):
        self.pars.r1 = pars_r1
        if self.ini_path is None:
            path = self.get_path_save(current_dir)
            if path:
                self.save_ini(Path(path))
            self.ini_path = path
        else:
            self.save_ini(Path(self.ini_path))

    def save_ini_r2(self, pars_r2, current_dir):
        self.pars.r2 = pars_r2
        if self.ini_path is None:
            path = self.get_path_save(current_dir)
            if path:
                self.save_ini(Path(path))
            self.ini_path = path
        else:
            self.save_ini(Path(self.ini_path))

    def save_ini(self, filename: Path):
        config_ = ConfigParser()
        config_["parameters_r1"] = self.pars.r1
        config_["parameters_r2"] = self.pars.r2
        config_["advanced_parameters"] = self.pars.advanced
        with open(filename, 'w') as configfile:
            config_.write(configfile)

    def get_path_save(self, current_dir):
        seg_filename = "config.ini"
        mode_ = FileDialogMode.OPTIONAL_FILE
        path = use_app().get_obj("show_file_dialog")(
            mode_,
            caption="Save ini",
            start_path=str(current_dir / seg_filename),
            filter='*.ini'
        )
        return path