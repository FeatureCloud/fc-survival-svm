import os
from time import sleep

import yaml

from engine.app import AppState, STATE_RUNNING, STATE_ACTION

INPUT_DIR = "/mnt/input"
OUTPUT_DIR = "/mnt/output"


class BlankState(AppState):

    def __init__(self, next_state=None):
        super().__init__()
        self.next_state = next_state

    def register(self):
        if self.next_state:
            self.register_transition(self.next_state)

    def run(self):
        return self.next_state


# class CopyState(BlankState):
#
#     def __init__(self, next_state=None):
#         super().__init__(next_state)
#
#     def run(self):
#         dir_util.copy_tree('/mnt/input/', '/mnt/output/')
#         return super().run()


class CheckConfigSupplied(AppState):

    def __init__(self, next_state_exists: str, next_state_missing: str, section: str):
        """
        State that checks weather a config file exists and has the expected section.

        :param next_state_exists: State when the config file exists and has the section
        :param next_state_missing: State when the config file is missing or misses the section
        :param section: Name of the section in config.yml file
        """
        super().__init__()
        self.next_state_exists = next_state_exists
        self.next_state_missing = next_state_missing
        self.section = section

    def register(self):
        self.register_transition(self.next_state_exists)
        self.register_transition(self.next_state_missing)

    def run(self) -> str or None:
        self.update(
            message=f'Checking if config file is supplied',
            progress=0.02,
            state=STATE_RUNNING
        )
        if self.section:
            config_file_path: str = os.path.join('/mnt/input', 'config.yml')
            if os.path.isfile(config_file_path):
                with open(config_file_path) as f:
                    self.app.log(f'Config file at {config_file_path} exists')
                    config_yml = yaml.load(f, Loader=yaml.FullLoader)
                    if self.section in config_yml:
                        self.app.log(f'Section {self.section} exists in config file')
                        return self.next_state_exists
                    else:
                        self.app.log(f'Section {self.section} is missing in config file')
            else:
                self.app.log(f'No config file at {config_file_path} found')
            return self.next_state_missing


class ConfigFileState(BlankState):

    def __init__(self, next_state, section, config='config'):
        super().__init__(next_state)
        self.section = section
        self.config = config

    def run(self):
        self.update(
            message=f'Reading config file',
            progress=0.03,
            state=STATE_RUNNING
        )
        if self.section:
            with open('/mnt/input/config.yml') as f:
                self.app.internal[self.config] = yaml.load(f, Loader=yaml.FullLoader)[self.section]
        return super().run()


class ConfigWebState(BlankState):

    def __init__(self, next_state, recheck_period=10):
        super().__init__(next_state)
        self.recheck_period = recheck_period

    def run(self):
        self.update(
            message=f'Please configure app via frontend',
            progress=0.03,
            state=STATE_ACTION
        )
        while self.app.internal.get('config') is None:
            sleep(self.recheck_period)
        return super().run()
