import _queue
import jsonpickle
import logging
import numpy as np
import os
import pandas as pd
import pickle
import shutil
import threading
import time
import yaml
from federated_pure_regression_survival_svm.model import Coordinator, Client, SurvivalData, SharedConfig, LocalResult, \
    OptFinished
from federated_pure_regression_survival_svm.stepwise_newton_cg import SteppedEventBasedNewtonCgOptimizer
from nptyping import NDArray, Bool
from scipy.optimize import OptimizeResult
from sksurv.svm import FastSurvivalSVM
from typing import Any, Dict, Union, List, Optional, Tuple


class AppLogic:

    def __init__(self):
        # === Status of this app instance ===

        # Indicates whether there is data to share, if True make sure self.data_out is available
        self.status_available = False

        # Only relevant for coordinator, will stop execution when True
        self.status_finished = False

        # === Parameters set during setup ===
        self.id = None
        self.coordinator = None
        self.clients = None

        # === Data ===
        self.data_incoming = []
        self.data_outgoing = None

        # === Internals ===
        self.thread = None
        self.iteration = 0
        self.progress = 'not started yet'

        # === Custom ===
        self.INPUT_DIR = "/mnt/input"
        self.OUTPUT_DIR = "/mnt/output"

        self.models: Dict[str, Union[Client, Coordinator]] = {}

        self.train_filename = None
        self.test_filename = None

        self.model_output = None
        self.pred_output = None
        self.test_output = None

        self.sep = None
        self.label_time_to_event = None
        self.label_event = None
        self.event_truth_value = None

        self.mode = None
        self.dir = "."

        self.alpha = None
        self.fit_intercept = None
        self.max_iter = None

        self.splits: Dict[str, Tuple[pd.DataFrame, NDArray]] = {}
        self.svm: Dict[str, FastSurvivalSVM] = {}

    def read_config(self):
        with open(os.path.join(self.INPUT_DIR, "config.yml")) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_survival_svm']
            self.train_filename = config['input']['train']
            self.test_filename = config['input']['test']

            self.model_output = config['output']['model']
            self.pred_output = config['output']['pred']
            self.test_output = config['output']['test']

            self.sep = config['format']['sep']
            self.label_time_to_event = config["format"]["label_survival_time"]
            self.label_event = config["format"]["label_event"]
            self.event_truth_value = config["format"].get("event_truth_value", True)  # default value

            self.mode = config['split']['mode']
            self.dir = config['split']['dir']

            self.alpha = config['svm']['alpha']
            self.fit_intercept = config['svm']['fit_intercept']
            self.max_iter = config['svm']['max_iterations']

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

    @staticmethod
    def get_column(dataframe: pd.DataFrame, col_name: str) -> pd.Series:
        try:
            return dataframe[col_name]
        except KeyError as e:
            logging.error(f"Column {col_name} does not exist in the data")
            raise e

    @staticmethod
    def event_value_to_truth_array(event: NDArray[Any], truth_value: Any) -> NDArray[Bool]:
        if truth_value is True and event.dtype == np.dtype('bool'):  # nothing to do...
            return event

        truth_array = (event == truth_value)
        return truth_array

    def read_data_frame(self, path):
        logging.info(f"Read data file at {path}")
        dataframe = pd.read_csv(path, sep=self.sep)
        logging.debug(f"Dataframe:\n{dataframe}")
        return dataframe

    def read_survival_data(self, path) -> Tuple[pd.DataFrame, NDArray]:
        X: pd.DataFrame = self.read_data_frame(path)

        event = self.get_column(X, self.label_event)
        logging.debug(f"event:\n{event}")
        event_occurred = self.event_value_to_truth_array(event.to_numpy(), self.event_truth_value)
        logging.debug(f"event_occurred:\n{event_occurred}")

        time_to_event = self.get_column(X, self.label_time_to_event)
        logging.debug(f"time_to_event:\n{time_to_event}")

        X.drop([self.label_event, self.label_time_to_event], axis=1, inplace=True)
        logging.debug(f"features:\n{X}")
        y = np.zeros(X.shape[0], dtype=[('Status', '?'), ('Survival', '<f8')])  # TODO
        y['Status'] = event
        y['Survival'] = time_to_event

        return [X, y]

    def handle_setup(self, client_id, coordinator, clients):
        # This method is called once upon startup and contains information about the execution context of this instance
        self.id = client_id
        self.coordinator = coordinator
        self.clients = clients
        logging.info(f'Received setup: {self.id} {self.coordinator} {self.clients}')

        self.thread = threading.Thread(target=self.app_flow)
        self.thread.start()

    def handle_incoming(self, data):
        # This method is called when new data arrives
        logging.info("Process incoming data....")
        self.data_incoming.append(data.read())

    def handle_outgoing(self):
        logging.info("Process outgoing data...")
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def _all_finished(self, models: Dict[str, Coordinator]):
        for model in models.values():
            optimizer: SteppedEventBasedNewtonCgOptimizer = model.newton_optimizer
            if not optimizer.finished:
                return False
        return True

    def _get_all_requests(self, models: Dict[str, Coordinator], recheck_timeout=1):
        requests: Dict[str, Optional[SteppedEventBasedNewtonCgOptimizer.Request]] = dict.fromkeys(models.keys())
        for model_identifier, coordinator in models.items():
            optimizer: SteppedEventBasedNewtonCgOptimizer = coordinator.newton_optimizer
            while not optimizer.finished:
                try:
                    requests[model_identifier] = optimizer.check_pending_requests(block=False, timeout=recheck_timeout)
                    break
                except _queue.Empty:
                    continue
        return requests

    def _fulfill_all_requests_local(self, requests: Dict[str, Optional[SteppedEventBasedNewtonCgOptimizer.Request]],
                                    models: Dict[str, Coordinator]):
        local_results: Dict[str, Optional[LocalResult]] = dict.fromkeys(models.keys())
        for model_identifier, request in requests.items():
            if request is None:
                continue
            model: Client = models[model_identifier]
            local_results[model_identifier] = model.handle_computation_request(request)
        return local_results

    def _fulfill_all_requests_global(self, local_results: Dict[str, Optional[List[LocalResult]]],
                                     models: Dict[str, Coordinator]):
        aggregated_results: Dict[str, Optional[SteppedEventBasedNewtonCgOptimizer.Resolved]] = dict.fromkeys(
            models.keys())
        for model_identifier, local_result in local_results.items():
            if local_result is None:
                continue
            model: Coordinator = models[model_identifier]
            aggregated_results[model_identifier] = model.aggregate_local_result(local_result)
        return aggregated_results

    def _inform_all(self, aggregated_results: Dict[str, Optional[SteppedEventBasedNewtonCgOptimizer.Resolved]],
                    models: Dict[str, Coordinator]):
        for model_identifier, aggregated_result in aggregated_results.items():
            if aggregated_result is None:
                continue
            model: Coordinator = models[model_identifier]
            optimizer: SteppedEventBasedNewtonCgOptimizer = model.newton_optimizer
            optimizer.resolve(aggregated_result)

    def app_flow(self):
        # This method contains a state machine for the client and coordinator instance

        # === States ===
        state_initializing = 1
        state_read_input = 2
        state_preprocessing = 3
        state_send_data_attributes = 4
        state_global_aggregation_of_data_attributes = 5
        state_global_optimization = 6
        state_local_optimization_calculation_requests_listener = 7
        state_send_global_model = 8
        state_set_global_model = 9
        state_generate_predictions = 10
        state_shutdown = 11

        # Initial state
        state = state_initializing
        self.progress = 'initializing...'

        while True:
            logging.debug(f"Current state: {state}")
            logging.debug(f"Progress: {self.progress}")

            if state == state_initializing:
                logging.info("Initializing")
                if self.id is not None:  # Test if setup has happened already
                    logging.info(f"Coordinator: {self.coordinator}")

                    state = state_read_input

            if state == state_read_input:
                logging.info('Read input and config')
                self.progress = 'reading config...'
                self.read_config()

                self.progress = 'reading data...'
                for split in self.splits.keys():
                    logging.info(f'Read {split} data')
                    if self.coordinator:
                        self.models[split] = Coordinator()
                    else:
                        self.models[split] = Client()

                    self.splits[split] = self.read_survival_data(os.path.join(split, self.train_filename))
                state = state_preprocessing

            if state == state_preprocessing:
                self.progress = 'preprocessing...'
                for split in self.splits.keys():
                    logging.info(f'Preprocess {split}')
                    model = self.models[split]
                    X, y = self.splits[split]
                    model.set_config(
                        SharedConfig(alpha=self.alpha, fit_intercept=self.fit_intercept,
                                     max_iter=self.max_iter))  # TODO: check if nodes agree on config
                    model.set_data(SurvivalData(X, y))
                    model.log_transform_times()  # run log transformation of time values for regression objective
                state = state_send_data_attributes

            if state == state_send_data_attributes:
                self.progress = "generating data descriptions..."
                data_to_send = {}
                for split in self.splits.keys():
                    logging.info(f'Generate data attributes for split {split}')
                    data_to_send[split] = self.models[split].generate_data_description()

                logging.debug(f"Data attributes:\n{data_to_send}")
                data_to_send = jsonpickle.encode(data_to_send)

                if self.coordinator:
                    self.data_incoming.append(data_to_send)
                    state = state_global_aggregation_of_data_attributes
                else:
                    self.data_outgoing = data_to_send
                    self.status_available = True
                    state = state_local_optimization_calculation_requests_listener
                    logging.info(f'[CLIENT] Sending attributes to coordinator')

            if state == state_global_aggregation_of_data_attributes:
                logging.debug(self.data_incoming)
                self.progress = f'waiting for data attributes... {len(self.data_incoming)} of {len(self.clients)}'
                if len(self.data_incoming) == len(self.clients):
                    self.progress = 'aggregating data attributes...'
                    logging.debug("Received data attributes of all clients")
                    data = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
                    self.data_incoming = []
                    for split in self.splits.keys():
                        logging.debug(f'Aggregate {split}')
                        split_data = []
                        for client in data:
                            split_data.append(client[split])
                        logging.debug(f"Data attributes at split {split}: {split_data}")
                        self.models[split].set_data_attributes(split_data)
                        self.models[split].set_initial_w_and_init_optimizer()

                    requests = self._get_all_requests(self.models)

                    logging.debug(requests)
                    data_to_send = jsonpickle.encode(requests)

                    if self.coordinator:
                        self.data_incoming.append(data_to_send)

                    self.data_outgoing = data_to_send
                    self.status_available = True
                    logging.info(f'[COORDINATOR] Sending initial calculation requests')

                    state = state_local_optimization_calculation_requests_listener

            if state == state_global_optimization:
                logging.debug(self.data_incoming)
                self.progress = f'waiting for results of calculation requests... {len(self.data_incoming)} of {len(self.clients)}'

                if len(self.data_incoming) == len(self.clients):
                    self.progress = 'starting global calculation...'
                    logging.debug("Received results of calculation requests")
                    data = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
                    self.data_incoming = []

                    local_results: Dict[str, Optional[List[LocalResult]]] = {k: None for k, v in self.splits.items()}
                    for split in self.splits.keys():
                        logging.debug(f'Aggregate {split}')
                        split_data = []
                        for client in data:
                            split_data.append(client[split])
                        logging.debug(f"Result for {split}: {split_data}")

                        local_results[split] = split_data

                    logging.debug(f"Local results: {local_results}")
                    aggregated = self._fulfill_all_requests_global(local_results, self.models)
                    logging.debug(f"Aggregated: {aggregated}")
                    self._inform_all(aggregated, self.models)

                    requests = self._get_all_requests(self.models)
                    logging.debug(f"Requests: {requests}")

                    if self._all_finished(self.models):
                        logging.info('Optimization for all splits finished')
                        state = state_send_global_model
                    else:
                        data_to_send = jsonpickle.encode(requests)

                        if self.coordinator:
                            self.data_incoming.append(data_to_send)

                        self.data_outgoing = data_to_send
                        self.status_available = True
                        logging.info(f'[COORDINATOR] Sending calculation requests')
                        state = state_local_optimization_calculation_requests_listener

            if state == state_local_optimization_calculation_requests_listener:
                self.progress = 'waiting for calculation requests...'
                if len(self.data_incoming) > 0:
                    logging.info("Received calculation requests")
                    self.progress = 'processing calculation requests...'
                    requests = jsonpickle.decode(self.data_incoming[0])
                    self.data_incoming = []

                    if isinstance(requests, OptFinished):
                        logging.debug("Received OptFinished signal")
                        state = state_set_global_model
                        self.opt_finished_signal: OptFinished = requests
                    else:
                        results = self._fulfill_all_requests_local(requests, self.models)

                        logging.debug(f"Local results:\n{results}")
                        data_to_send = jsonpickle.encode(results)

                        if self.coordinator:
                            self.data_incoming.append(data_to_send)
                            state = state_global_optimization
                        else:
                            self.data_outgoing = data_to_send
                            self.status_available = True
                            state = state_local_optimization_calculation_requests_listener
                            logging.info(f'[CLIENT] Sending attributes to coordinator')

            if state == state_send_global_model:
                self.progress = 'optimization finished...'
                opt_results: Dict[str, OptimizeResult] = {k: None for k, v in self.splits.items()}
                for split in self.splits.keys():
                    model: Coordinator = self.models[split]
                    optimizer: SteppedEventBasedNewtonCgOptimizer = model.newton_optimizer
                    opt_results[split] = optimizer.result
                logging.debug(f'Optimizers: {opt_results}')

                data_to_send = jsonpickle.encode(OptFinished(opt_results=opt_results))

                if self.coordinator:
                    self.data_incoming.append(data_to_send)

                self.data_outgoing = data_to_send
                self.status_available = True
                state = state_local_optimization_calculation_requests_listener

            if state == state_set_global_model:
                self.progress = "save global models..."
                logging.debug(f"GLOBAL MODELS {self.opt_finished_signal}")
                opt_results: Dict[str, Optional[OptimizeResult]] = self.opt_finished_signal.opt_results
                for split in self.splits.keys():
                    logging.info(f'Get and export model for {split}')
                    opt_result = opt_results.get(split, None)

                    if opt_result is None:
                        continue

                    model: Client = self.models[split]
                    sksurv_obj: FastSurvivalSVM = model.to_sksurv(opt_result)
                    self.svm[split] = sksurv_obj

                    # write pickled model
                    pickle_output_path = os.path.join(split.replace("/input", "/output"), self.model_output)
                    logging.debug(f"Writing model to {pickle_output_path}")
                    with open(pickle_output_path, "wb") as fh:
                        pickle.dump(sksurv_obj, fh)

                    # write model parameters as meta file
                    meta_output_path = os.path.join(split.replace("/input", "/output"), "meta.yml")
                    logging.debug(f"Writing metadata to {meta_output_path}")

                    # unpack coefficients
                    coefficients = {}
                    features = self.splits[split][0].columns.values
                    weights = sksurv_obj.coef_.tolist()
                    for feature_name, feature_weight in zip(features, weights):
                        coefficients[feature_name] = feature_weight
                    if self.fit_intercept is not None:
                        coefficients["intercept"] = float(sksurv_obj.intercept_)

                    metadata = {
                        "model": {
                            "name": "FederatedPureRegressionSurvivalSVM",
                            "version": "0.0.1",
                            "parameters": {
                                "alpha": sksurv_obj.alpha,
                                "rank_ratio": sksurv_obj.rank_ratio,
                                "fit_intercept": sksurv_obj.fit_intercept,
                                "coefficients": coefficients
                            },
                        },
                        "split": split,
                    }
                    with open(meta_output_path, "w") as fh:
                        yaml.dump(metadata, fh)

                state = state_generate_predictions

            if state == state_generate_predictions:
                self.progress = "generate predictions on test data..."
                for split in self.splits.keys():
                    logging.info(f'Generate predictions for {split}')
                    logging.debug(f'Load test data for {split}')
                    X_test, y_test = self.read_survival_data(os.path.join(split, self.test_filename))
                    logging.debug(f'Load model for {split}')
                    svm: FastSurvivalSVM = self.svm[split]

                    logging.debug(f'Generate predictions for {split}')
                    predictions: NDArray[float] = svm.predict(X_test)

                    # re-add tte and event column
                    X_test[self.label_time_to_event] = y_test['Survival']
                    X_test[self.label_event] = y_test['Status']
                    # add predictions to dataframe
                    X_test['predicted_tte'] = predictions.tolist()

                    pred_output_path = os.path.join(split.replace("/input", "/output"), self.pred_output)
                    logging.debug(f"Writing predictions to {pred_output_path}")
                    X_test.to_csv(pred_output_path, sep=self.sep, index=False)
                state = state_shutdown

            if state == state_shutdown:
                self.progress = "finished"
                logging.info("Finished")
                self.status_finished = True
                break

            time.sleep(1)


logic = AppLogic()
