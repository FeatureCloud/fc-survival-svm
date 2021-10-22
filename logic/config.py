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
        self.tries_recover = 3

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

            config.enable_smpc = config_yml['privacy'].get('enable_smpc', True)  # default value
            requested_min_samples = config_yml['privacy'].get('min_samples')
            if requested_min_samples is not None:
                if requested_min_samples > cls.DEFAULT_MIN_SAMPLES:
                    config.min_samples = requested_min_samples
                else:
                    logging.warning(f'Enforcing safe value of {cls.DEFAULT_MIN_SAMPLES} for minimum number of samples')

        return config

    @classmethod
    def from_web(cls, form, is_coordinator):
        config = cls()

        config.train_filename = form.get('train_filename')
        config.test_filename = form.get('test_filename')

        config.model_output = form.get('model_output')
        config.pred_output = form.get('pred_output')
        config.meta_output = form.get('meta_output', 'meta.yml')  # default value
        config.train_output = form.get('train_output', config.train_filename)  # default value
        config.test_output = form.get('test_output', config.test_filename)  # default value

        config.sep = form.get('sep')
        config.label_time_to_event = form.get('label_time_to_event')
        config.label_event = form.get('label_event')
        if form.get('event_truth_value') != "":
            config.event_truth_value = form.get('event_truth_value', True)  # default value
        else:
            config.event_truth_value = True  # default value

        if is_coordinator:
            config.parameters.alpha = int(form.get('alpha'))
            config.parameters.fit_intercept = form.get('fit_intercept')
            config.parameters.max_iter = int(form.get('max_iter'))
            config.parameters._tries_recover = int(form.get('_tries_recover', 3))

        config.mode = SplitMode.from_str(form.get('mode'))
        config.dir = form.get('dir')

        config.enable_smpc = form.get('enable_smpc', True)  # default value
        requested_min_samples = form.get('min_samples')
        if requested_min_samples is not None:
            requested_min_samples = int(requested_min_samples)
            if requested_min_samples > cls.DEFAULT_MIN_SAMPLES:
                config.min_samples = requested_min_samples
            else:
                logging.warning(f'Enforcing safe value of {cls.DEFAULT_MIN_SAMPLES} for minimum number of samples')

        return config
