import dataclasses
import enum
import os
from collections import Iterable, Iterator
from typing import Optional, Dict

from app.survival_svm.settings import INPUT_DIR, OUTPUT_DIR


class SplitMode(enum.Enum):
    DIRECTORY = 'directory'
    FILE = 'file'

    @classmethod
    def from_str(cls, mode_descriptor: str) -> Optional['SplitMode']:
        mode: SplitMode
        for name, mode in SplitMode.__members__.items():
            if mode.value == mode_descriptor:
                return mode
        return None


class SplitManager:

    def __init__(self, mode: SplitMode, directory_name: str):
        self.splits = []
        self.mode = mode
        self.output_dirs = {}
        self.data = {}

        if self.mode.value == SplitMode.DIRECTORY.value:
            cv_directory: str = os.path.join(INPUT_DIR, directory_name)
            self.splits = [f.path for f in os.scandir(cv_directory) if f.is_dir()]  # noqa
        else:
            self.splits[INPUT_DIR] = None

        self._create_output_directories()
        self._create_data()

    class Env:
        def __init__(self, split, output_dir, data):
            self.name = split
            self.input_dir = split
            self.output_dir = output_dir
            self.data = data

        def __repr__(self):
            return f'{self.__class__.__name__}<' \
                   f'input_dir={self.input_dir}, ' \
                   f'output_dir={self.output_dir}, ' \
                   f'data={self.data}>'

    def __iter__(self):
        for split in self.splits:
            yield self.Env(split, self.output_dirs.get(split), self.data.get(split))

    def _create_output_directories(self):
        for split in self.splits:
            output_dir = split.replace(INPUT_DIR, OUTPUT_DIR)
            os.makedirs(output_dir, exist_ok=True)
            self.output_dirs[split] = output_dir

    def _create_data(self):
        for split in self.splits:
            self.data[split] = {}
