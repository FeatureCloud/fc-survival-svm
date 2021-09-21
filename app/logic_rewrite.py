import collections
import enum
import logging
import os
import threading
from typing import Optional, List, Dict, Union, Tuple, Any

import pandas as pd
import yaml
from nptyping.ndarray import NDArray
from sksurv.svm import FastSurvivalSVM

from communication.silo_communication import JsonEncodedCommunication, Communication, SmpcCommunication
from federated_pure_regression_survival_svm.model import Client, Coordinator


@enum.unique
class SplitMode(enum.Enum):
    DIRECTORY = "directory"
    FILE = "file"

    @classmethod
    def from_str(cls, mode_descriptor: str) -> Optional['SplitMode']:
        mode: SplitMode
        for mode in cls:
            if mode.value == mode_descriptor:
                return mode
        return None

class Settings:
    def __init__(self):
        self.INPUT_DIR = "/mnt/input"
        self.OUTPUT_DIR = "/mnt/output"

        self.config_file_name = "config.yml"
        self.main_directive = "fc_survival_svm"


class Config:
    def __init__(self):
        self.train_filename: Optional[str] = None
        self.test_filename: Optional[str] = None

        self.model_output: Optional[str] = None
        self.meta_output: Optional[str] = None
        self.pred_output: Optional[str] = None
        self.test_output: Optional[str] = None
        self.train_output: Optional[str] = None

        self.sep: Optional[str] = None
        self.label_time_to_event: Optional[str] = None
        self.label_event: Optional[str] = None
        self.event_truth_value: Optional[str] = None

        self.mode: SplitMode = None
        self.dir = "."


class Parameters:
    def __init__(self):
        self.alpha: Optional[int] = None
        self.fit_intercept: Optional[bool] = None
        self.max_iter: Optional[int] = None


class CvHelper:
    def __init__(self, settings: Settings, split_mode_descriptor: str, dir: str):
        self._split_mode: SplitMode = self._set_split_mode(split_mode_descriptor)

        self.splits: Dict[str, Optional[Tuple[pd.DataFrame, NDArray]]] = {}
        self.models: Dict[str, Optional[Union[Client, Coordinator]]] = {}

        if self._split_mode == SplitMode.DIRECTORY:
            input_dir = os.path.join(settings.INPUT_DIR, dir)
            self.splits = dict.fromkeys([f.path for f in os.scandir(input_dir) if f.is_dir()])  # noqa
            self.models = dict.fromkeys(self.splits.keys())
        else:
            if dir != ".":
                input_dir = os.path.join(settings.INPUT_DIR, dir)
            else:
                input_dir = settings.INPUT_DIR

            self.splits[input_dir]: Dict[str, Tuple[pd.DataFrame, NDArray]] = None
            self.models[input_dir]: Dict[str, Union[Client, Coordinator]] = None

        self.train_data_paths: Dict[str, str] = {}
        self.svm: Dict[str, FastSurvivalSVM] = {}
        self.training_states: Optional[Dict[str, Dict[str, Any]]] = None
        self.last_requests = None

        self.timings: Dict[str, float] = collections.defaultdict(float)


    def _set_split_mode(self, mode_descriptor: str):
        split_mode: Optional[SplitMode] = SplitMode.from_str(mode_descriptor)
        if split_mode is not None:
            raise RuntimeError("Unknown mode")
        logging.debug(f"SplitMode was set to {self._split_mode.value}")


class State:
    def __init__(self):
        self._progress = "waiting for setup call"

    @property
    def progress(self):
        return self._progress

    @progress.setter
    def progress(self, progress: str):
        logging.debug(f"Changed progress to {progress!r}")
        self._progress = progress


class AppLogic:

    def read_and_apply_config(self):
        self.state.progress = "reading config"
        config_file_path: str = os.path.join(self.settings.INPUT_DIR, self.settings.config_file_name)
        with open(config_file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)[self.settings.main_directive]

            enable_smpc = config['privacy']['enable_smpc']
            if enable_smpc:
                self.communicator = SmpcCommunication()

            self.config.train_filename = config['input']['train']
            self.config.test_filename = config['input']['test']

            self.config.model_output = config['output']['model']
            self.config.pred_output = config['output']['pred']
            self.config.meta_output = config['output'].get("meta", "meta.yml")  # default value
            self.config.train_output = config['output'].get("train", self.config.train_filename)  # default value
            self.config.test_output = config['output'].get("test", self.config.test_filename)  # default value

            self.config.sep = config['format']['sep']
            self.config.label_time_to_event = config["format"]["label_survival_time"]
            self.config.label_event = config["format"]["label_event"]
            self.config.event_truth_value = config["format"].get("event_truth_value", True)  # default value

            split_mode = config['split']['mode']
            dir = config['split']['dir']
            self.cv_helper = CvHelper(self.settings, split_mode, dir)

            self.parameters.alpha = config['svm']['alpha']
            self.parameters.fit_intercept = config['svm']['fit_intercept']
            self.parameters.max_iter = config['svm']['max_iterations']

            self._smpc_seed = config['privacy'].get('_smpc_seed', None)
            self._tries_recover = config['svm'].get('_tries_recover', 3 if self.enable_smpc else 0)
            logging.debug(f"TRIES: {self._tries_recover}")

        if self.mode == "directory":
            self.splits = dict.fromkeys([f.path for f in os.scandir(f'{self.INPUT_DIR}/{self.dir}') if f.is_dir()])
            self.models = dict.fromkeys(self.splits.keys())
        else:
            self.splits[self.INPUT_DIR] = None
            self.models[self.INPUT_DIR] = None

        for split in self.splits.keys():
            os.makedirs(split.replace("/input", "/output"), exist_ok=True)
        shutil.copyfile(self.INPUT_DIR + '/config.yml', self.OUTPUT_DIR + '/config.yml')
        logging.info(f'Read config file.')

    def __init__(self):
        self.status_finished = False

        self.settings: Settings = Settings()
        self.config: Config = Config()

        self.communicator: Communication = JsonEncodedCommunication()
        self.cv_helper: Optional[CvHelper] = None

        self.id: Optional[int] = None
        self.is_coordinator: Optional[bool] = None
        self.clients: Optional[List[int]] = None


        self.thread: Optional[threading.Thread] = None

        self.state: State = State()
        self.settings: Settings = Settings()

        self.parameters: Optional[Parameters] = None
