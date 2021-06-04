import logging
import os
import shutil
import threading
import time
from typing import Any, Dict, Union, List

import joblib
import jsonpickle
import numpy as np
import pandas as pd
import yaml
from nptyping import NDArray, Bool
from scipy.optimize import OptimizeResult

from federated_pure_regression_survival_svm.model import Coordinator, Client, SurvivalData, SharedConfig, ObjectivesW, \
    ObjectivesS
from federated_pure_regression_survival_svm.stepwise_newton_cg import NewtonCgStateMachine, UpdateW, UpdateS


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

        self.splits = {}
        self.test_splits = {}
        self.requests = {}

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
            self.test_splits = dict.fromkeys(self.splits.keys())
            self.models = dict.fromkeys(self.splits.keys())
        else:
            self.splits[self.INPUT_DIR] = None
            self.test_splits[self.INPUT_DIR] = None
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

    def read_survival_data(self, path):
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

    def app_flow(self):
        # This method contains a state machine for the client and coordinator instance

        # === States ===
        state_initializing = 1
        state_read_input = 2
        state_preprocessing = 3
        state_send_data_attributes = 3.1
        state_global_aggregation_of_data_attributes = 3.2
        state_wait_for_aggregation_of_data_attributes = 3.3
        state_global_mainloop = 3.4
        state_global_stepwise_calculation_listener = 3.4
        state_local_calculation_requests_listener = 3.6
        state_local_computation = 4
        state_wait_for_aggregation = 5
        state_global_aggregation = 6
        state_writing_results = 7
        state_finishing = 8

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
                    self.test_splits[split] = self.read_survival_data(os.path.join(split, self.test_filename))
                state = state_preprocessing

            if state == state_preprocessing:
                self.progress = 'preprocessing...'
                for split in self.splits.keys():
                    logging.info(f'Preprocess {split}')
                    model = self.models[split]
                    X, y = self.splits[split]
                    model.set_config(
                        SharedConfig(alpha=self.alpha, fit_intercept=self.fit_intercept, max_iter=self.max_iter))
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
                    state = state_local_calculation_requests_listener
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
                        self.models[split].set_initial_w_and_prime_optimizer()
                        self.requests[split] = self.models[split].newton_optimizer.next(None)

                    logging.debug(self.requests)
                    data_to_send = jsonpickle.encode(self.requests)

                    if self.coordinator:
                        self.data_incoming.append(data_to_send)

                    self.data_outgoing = data_to_send
                    self.status_available = True
                    logging.info(f'[COORDINATOR] Sending initial calculation requests')

                    state = state_local_calculation_requests_listener

            if state == state_local_calculation_requests_listener:
                self.progress = 'waiting for calculation requests...'
                if len(self.data_incoming) > 0:
                    logging.info("Received calculation requests")
                    self.progress = 'processing calculation requests...'
                    requests = jsonpickle.decode(self.data_incoming[0])
                    self.data_incoming = []

                    results = {}
                    for split in self.splits:
                        logging.debug(f"Request for split {split}: {requests[split]}")
                        request = requests[split]
                        data = None

                        if isinstance(request, OptimizeResult):
                            data = None
                        elif isinstance(request, NewtonCgStateMachine.RequestWDependent):
                            request: NewtonCgStateMachine.RequestWDependent
                            data = self.models[split].get_values_depending_on_w(request.xk)
                        elif isinstance(request, NewtonCgStateMachine.RequestHessian):
                            request: NewtonCgStateMachine.RequestHessian
                            data = self.models[split].hessian_func(request.psupi)

                        results[split] = data

                    logging.debug(f"Local results:\n{results}")
                    data_to_send = jsonpickle.encode(results)

                    if self.coordinator:
                        self.data_incoming.append(data_to_send)
                        state = state_global_mainloop
                    else:
                        self.data_outgoing = data_to_send
                        self.status_available = True
                        state = state_local_calculation_requests_listener
                        logging.info(f'[CLIENT] Sending attributes to coordinator')


                    # results = {}
                    # for split in self.splits.keys():
                    #     w = requests[split]
                    #     logging.debug(f"Got {w} as initial weights for {split}")
                    #     objectivesW = self.models[split].get_values_depending_on_w(w)
                    #     results[split] = objectivesW
                    #
                    # data_to_send = jsonpickle.encode(results)
                    #
                    # if self.coordinator:
                    #     self.data_incoming.append(data_to_send)
                    #     state = state_global_start_stepwise_calculation
                    # else:
                    #     self.data_outgoing = data_to_send
                    #     self.status_available = True
                    #     state = state_local_stepwise_calculation_listener
                    #     logging.info(f'[CLIENT] Sending initial local sum of zeta squared and local gradient value')


            if state == state_global_mainloop:
                logging.debug(self.data_incoming)
                self.progress = f'waiting for results of calculation requests... {len(self.data_incoming)} of {len(self.clients)}'
                if len(self.data_incoming) == len(self.clients):
                    self.progress = 'starting global calculation...'
                    logging.debug("Received results of calculation requests")
                    data = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
                    self.data_incoming = []


                    for split in self.splits.keys():
                        logging.debug(f'Aggregate {split}')
                        split_data = []
                        for client in data:
                            split_data.append(client[split])
                        logging.debug(f"Result for {split}: {split_data}")

                        result = split_data

                        if isinstance(result[0], ObjectivesW):
                            result: List[ObjectivesW]
                            aggregated: UpdateW = self.models[split].update_fval_and_gval(result)
                            self.requests[split] = self.models[split].newton_optimizer.next(aggregated)
                        elif isinstance(result[0], ObjectivesS):
                            result: List[ObjectivesS]
                            aggregated: UpdateS = self.models[split].aggregated_hessian(result)
                            self.requests[split] = self.models[split].newton_optimizer.next(aggregated)

                    logging.debug(self.requests)
                    data_to_send = jsonpickle.encode(self.requests)

                    if self.coordinator:
                        self.data_incoming.append(data_to_send)

                    self.data_outgoing = data_to_send
                    self.status_available = True
                    logging.info(f'[COORDINATOR] Sending calculation requests')

                    state = state_local_calculation_requests_listener

            if state == state_local_computation:
                print("Perform local beta update")
                self.progress = 'local beta update'
                self.iter_counter = self.iter_counter + 1
                print(f'Iteration {self.iter_counter}')
                data_to_send = {}
                for split in self.splits.keys():
                    print(f'Compute {split}')
                    try:
                        data_to_send[split] = self.models[split].compute_derivatives(self.splits[split][0],
                                                                                     self.splits[split][1],
                                                                                     self.betas[split])
                    except FloatingPointError:
                        data_to_send[split] = "early_stop"

                data_to_send = jsonpickle.encode(data_to_send)

                if self.coordinator:
                    self.data_incoming.append(data_to_send)
                    state = state_global_aggregation
                else:
                    self.data_outgoing = data_to_send
                    self.status_available = True
                    state = state_wait_for_aggregation
                    print(f'[CLIENT] Sending computation data to coordinator', flush=True)

            if state == state_wait_for_aggregation:
                print("Wait for aggregation")
                self.progress = 'wait for aggregation'
                if len(self.data_incoming) > 0:
                    print("Received aggregation data from coordinator.")
                    data_decoded = jsonpickle.decode(self.data_incoming[0])
                    self.betas, self.betas_finished = data_decoded[0], data_decoded[1]
                    self.data_incoming = []
                    if False in self.betas_finished.values() and self.max_iter > self.iter_counter:
                        state = state_local_computation
                    else:
                        print("Beta update finished.")
                        for split in self.splits:
                            self.models[split].set_coefs(self.betas[split])
                        state = state_writing_results

            # GLOBAL PART

            if state == state_global_aggregation:
                print("Global computation")
                self.progress = 'computing...'
                if len(self.data_incoming) == len(self.clients):
                    print("Received data of all clients")
                    data = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
                    self.data_incoming = []
                    for split in self.splits.keys():
                        if not self.betas_finished[split]:
                            print(f'Aggregate {split}')
                            split_data = []
                            for client in data:
                                split_data.append(client[split])
                            beta, beta_finished = self.models[split].aggregate_beta(split_data)
                            self.betas[split] = beta
                            self.betas_finished[split] = beta_finished
                    data_to_broadcast = jsonpickle.encode([self.betas, self.betas_finished])
                    self.data_outgoing = data_to_broadcast
                    self.status_available = True
                    print(f'[COORDINATOR] Broadcasting computation data to clients', flush=True)
                    if False in self.betas_finished.values() and self.max_iter > self.iter_counter:
                        print(f'Beta update not finished for all splits.')
                        state = state_local_computation
                    else:
                        print("Beta update finished.")
                        for split in self.splits.keys():
                            self.models[split].set_coefs(self.betas[split])
                        state = state_writing_results

            if state == state_writing_results:
                print("Writing results")
                for split in self.splits.keys():
                    print(f'Write {split}')
                    model = self.models[split]

                    joblib.dump(model, split.replace("/input", "/output") + '/model.pkl')

                    y_pred = pd.DataFrame(model.predict(self.test_splits[split][0]), columns=["y_pred"])
                    y_proba = pd.DataFrame(model.predict_proba(self.test_splits[split][0]))
                    y_pred.to_csv(split.replace("/input", "/output") + "/" + self.pred_output, index=False)
                    y_proba.to_csv(split.replace("/input", "/output") + "/" + self.proba_output, index=False)

                if self.coordinator:
                    self.data_incoming = ['DONE']
                    state = state_finishing
                else:
                    self.data_outgoing = 'DONE'
                    self.status_available = True
                    break

            if state == state_finishing:
                print("Finishing")
                self.progress = 'finishing...'
                if len(self.data_incoming) == len(self.clients):
                    self.status_finished = True
                    break

            time.sleep(1)


logic = AppLogic()
