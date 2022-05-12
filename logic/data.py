import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from nptyping import NDArray, Bool, Float64
from sklearn.utils import check_array, check_consistent_length


def get_column(dataframe: pd.DataFrame, col_name: str) -> pd.Series:
    try:
        return dataframe[col_name]
    except KeyError as e:
        logging.error(f"Column {col_name} does not exist in the data")
        raise e


def event_value_to_truth_array(event: NDArray[Any], truth_value: Any) -> NDArray[Bool]:
    if truth_value is True and np.issubdtype(event.dtype, np.bool_):  # nothing to do...
        return event

    truth_array = (event == truth_value)
    return truth_array


def read_data_frame(path, sep):
    dataframe = pd.read_csv(path, sep=sep)
    return dataframe


@dataclass
class SurvivalTuple:
    event_indicator: NDArray[Bool]
    time_to_event: NDArray[Float64]


class SurvivalData:
    features: NDArray
    survival: SurvivalTuple

    def __init__(self, X: pd.DataFrame, y: NDArray):
        self.feature_names = X.columns.values

        x_array: NDArray = check_array(X)  # noqa
        event_indicator: NDArray = check_array(y['event_indicator'], ensure_2d=False)  # noqa
        time_to_event: NDArray = check_array(y['time_to_event'], ensure_2d=False)  # noqa
        check_consistent_length(X, event_indicator, time_to_event)

        self.features = x_array
        self.survival = SurvivalTuple(
            event_indicator=event_indicator,
            time_to_event=time_to_event,
        )

    def _drop_rows(self, rows):
        self.survival.time_to_event = np.drop(rows)
        self.survival.event_indicator = np.drop(rows)
        self.features = np.drop(rows)

    def drop_negative_and_zero_timepoints(self) -> int:
        """
        Drop rows with a negative or zero time to prepare data for log-transformation.

        :return: Number of dropped rows
        """
        logging.info("Check data for negative and zero timepoints")
        bad_rows = self.survival.time_to_event <= 0
        n_bad_rows = bad_rows.sum()
        if n_bad_rows > 0:
            logging.warning(f"Dropping {n_bad_rows} rows with negative or zero times")
            # drop rows in all arrays
            self._drop_rows(rows=bad_rows)
        return n_bad_rows

    def log_transform_times(self):
        logging.info("Log transform times for regression objective")
        self.survival.time_to_event = np.log(self.survival.time_to_event)
        if not np.isfinite(self.survival.time_to_event).all():
            raise ValueError('Not all times are finite after log transformation')

    @property
    def x_shape(self):
        return self.features.shape

    @property
    def n_samples(self):
        return self.x_shape[0]

    @property
    def n_uncensored(self):
        return self.survival.event_indicator.sum()

    @property
    def n_censored(self):
        return int(self.n_samples - self.n_uncensored)

    @property
    def n_features(self):
        return self.x_shape[1]

    @property
    def sum_of_times(self) -> float:
        return float(np.sum(self.survival.time_to_event, axis=None))


def read_survival_data_np(path, sep=',',
                          label_event='status', label_time_to_event='time_to_event',
                          event_truth_value=1):
    X: pd.DataFrame = read_data_frame(path, sep=sep)

    event = get_column(X, label_event)
    event_indicator = event_value_to_truth_array(event.to_numpy(), event_truth_value)
    time_to_event = get_column(X, label_time_to_event)
    X.drop([label_event, label_time_to_event], axis=1, inplace=True)

    y = np.zeros(X.shape[0], dtype=[('event_indicator', np.bool_), ('time_to_event', np.float64)])
    y['event_indicator'] = event_indicator
    y['time_to_event'] = time_to_event

    return X, y


def read_survival_data(path, sep=',',
                       label_event='status', label_time_to_event='time_to_event',
                       event_truth_value=1) -> SurvivalData:
    X, y = read_survival_data_np(path, sep=sep,
                                 label_event=label_event, label_time_to_event=label_time_to_event,
                                 event_truth_value=event_truth_value)
    return SurvivalData(X, y)
