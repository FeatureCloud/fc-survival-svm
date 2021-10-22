import dataclasses
import enum
import os
from collections import Iterable, Iterator
from typing import Optional, Dict, List, Union, Tuple

import numpy as np

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
            print(cv_directory)
            print(self.splits)
        else:
            self.splits.append(INPUT_DIR)

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

    def add_int_values_in_split_dictionaries(self, data: List[Dict[str, Dict[str, Union[int, float]]]]) -> Dict[str, Dict[str, Union[int, float]]]:
        aggregated = {}
        for split in iter(self):
            data_iterator = iter(data)
            split_aggregated: Dict[str, Union[int, float]] = next(data_iterator)[split.name]  # get first entry to learn structure of inner dict
            for client_part in data_iterator:  # iterate over the rest
                k: str
                for k in split_aggregated.keys():
                    split_aggregated[k] += client_part[split.name][k]
            aggregated[split.name] = split_aggregated
        return aggregated

    def add_answer_values_in_split_dictionaries(self, data: List[Dict[str, Union[Tuple[float,List[float]], List[float]]]]) -> Dict[str, Union[Tuple[float,List[float]], List[float]]]:
        aggregated = {}
        strategy = {}

        client_dict: Dict[str, Union[Tuple[float,List[float]], List[float]]]
        for client_dict in data:
            for split_name, split_data in client_dict.items():
                value = aggregated.get(split_name)  # previous

                if isinstance(split_data, list):
                    strategy[split_name] = 'wdep'
                    value: np.ndarray
                    if value is not None:
                        value += np.array(split_data)
                    else:
                        value = np.array(split_data)
                else:
                    strategy[split_name] = 'hessp'
                    value: List
                    if value is not None:
                        value[0] += split_data[0]
                        value[1] += np.array(split_data[1])
                    else:
                        value: List = [split_data[0], np.array(split_data[1])]

                aggregated[split_name] = value

        ret = {}
        for split_name, aggregated_data in aggregated.items():
            if strategy[split_name] == 'wdep':
                ret[split_name] = aggregated_data.tolist()
            elif strategy[split_name] == 'hessp':
                ret[split_name] = (aggregated_data[0], aggregated_data[1].tolist())

        return ret  # noqa
