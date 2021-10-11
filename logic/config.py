import enum
import logging
from typing import Optional

import yaml

from app.survival_svm.settings import CONFIG_SECTION


@enum.unique
class SplitMode(enum.Enum):
    DIRECTORY = "directory"
    FILE = "file"

    @classmethod
    def from_str(cls, mode_descriptor: str) -> Optional['SplitMode']:
        mode: SplitMode
        for mode in iter(cls):
            if mode.value == mode_descriptor:
                return mode
        return None


class Parameters:
    def __init__(self):
        self.alpha: Optional[int] = None
        self.fit_intercept: Optional[bool] = None
        self.max_iter: Optional[int] = None
        self._tries_recover: int = 3

    def __repr__(self):
        return f'{self.__class__.__name__}<' \
               f'alpha={self.alpha}, ' \
               f'fit_intercept={self.fit_intercept}, ' \
               f'max_iter={self.max_iter}, ' \
               f'_tries_recover={self._tries_recover}>'


class Config:
    DEFAULT_MIN_SAMPLES = 3

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

        self.mode: Optional[SplitMode] = None
        self.dir = "."

        self.parameters = Parameters()

        self.enable_smpc = True
        self.min_samples = self.DEFAULT_MIN_SAMPLES

    @classmethod
    def from_file(cls, yml_file_path: str):
        with open(yml_file_path, "r") as f:
            config_yml = yaml.load(f, Loader=yaml.FullLoader)[CONFIG_SECTION]

            config = cls()

            config.train_filename = config_yml['input']['train']
            config.test_filename = config_yml['input']['test']

            config.model_output = config_yml['output']['model']
            config.pred_output = config_yml['output']['pred']
            config.meta_output = config_yml['output'].get('meta', 'meta.yml')  # default value
            config.train_output = config_yml['output'].get('train', config.train_filename)  # default value
            config.test_output = config_yml['output'].get('test', config.test_filename)  # default value

            config.sep = config_yml['format']['sep']
            config.label_time_to_event = config_yml['format']['label_survival_time']
            config.label_event = config_yml['format']['label_event']
            config.event_truth_value = config_yml['format'].get('event_truth_value', True)  # default value

            if config_yml.get('svm'):
                config.parameters.alpha = config_yml['svm']['alpha']
                config.parameters.fit_intercept = config_yml['svm']['fit_intercept']
                config.parameters.max_iter = config_yml['svm']['max_iterations']
                config.parameters._tries_recover = config_yml['svm'].get('_tries_recover', 3)

            config.mode = SplitMode.from_str(config_yml['split']['mode'])
            config.dir = config_yml['split']['dir']

            config.enable_smpc = config_yml['privacy']['enable_smpc']
            requested_min_samples = config_yml['privacy'].get('min_samples')
            if requested_min_samples is not None:
                if requested_min_samples > cls.DEFAULT_MIN_SAMPLES:
                    config.min_samples = requested_min_samples
                else:
                    logging.warning(f'Enforcing safe value of {cls.DEFAULT_MIN_SAMPLES} for minimum number of samples')

        return config

        # if self.mode == "directory":
        #     self.splits = dict.fromkeys([f.path for f in os.scandir(f'{self.INPUT_DIR}/{self.dir}') if f.is_dir()])
        #     self.models = dict.fromkeys(self.splits.keys())
        # else:
        #     self.splits[self.INPUT_DIR] = None
        #     self.models[self.INPUT_DIR] = None
        #
        # for split in self.splits.keys():
        #     os.makedirs(split.replace("/input", "/output"), exist_ok=True)
        # shutil.copyfile(self.INPUT_DIR + '/config.yml', self.OUTPUT_DIR + '/config.yml')
        # logging.info(f'Read config file.')

# class CvHelper:
#     def __init__(self, split_mode_descriptor: str, dir: str):
#         self._split_mode: SplitMode = self._set_split_mode(split_mode_descriptor)
#
#         self.splits: Dict[str, Optional[Tuple[pd.DataFrame, NDArray]]] = {}
#         self.models: Dict[str, Optional[Union[Client, Coordinator]]] = {}
#
#         if self._split_mode == SplitMode.DIRECTORY:
#             input_dir = os.path.join(INPUT_DIR, dir)
#             self.splits = dict.fromkeys([f.path for f in os.scandir(input_dir) if f.is_dir()])  # noqa
#             self.models = dict.fromkeys(self.splits.keys())
#         else:
#             if dir != ".":
#                 input_dir = os.path.join(settings.INPUT_DIR, dir)
#             else:
#                 input_dir = settings.INPUT_DIR
#
#             self.splits[input_dir]: Dict[str, Tuple[pd.DataFrame, NDArray]] = None
#             self.models[input_dir]: Dict[str, Union[Client, Coordinator]] = None
#
#         self.train_data_paths: Dict[str, str] = {}
#         self.svm: Dict[str, FastSurvivalSVM] = {}
#         self.training_states: Optional[Dict[str, Dict[str, Any]]] = None
#         self.last_requests = None
#
#         self.timings: Dict[str, float] = collections.defaultdict(float)
#
#
#     def _set_split_mode(self, mode_descriptor: str):
#         split_mode: Optional[SplitMode] = SplitMode.from_str(mode_descriptor)
#         if split_mode is not None:
#             raise RuntimeError("Unknown mode")
#         logging.debug(f"SplitMode was set to {self._split_mode.value}")
