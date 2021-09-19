import _queue
import collections
import datetime
import logging
import os
import pickle
import shutil
import threading
import time
from typing import Any, Dict, Union, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from nptyping import NDArray, Bool
from scipy.optimize import OptimizeResult
from sksurv.svm import FastSurvivalSVM

from communication.silo_communication import Communication, JsonEncodedCommunication
from federated_pure_regression_survival_svm.model import Coordinator, Client, SurvivalData, SharedConfig, LocalResult, \
    OptFinished
from federated_pure_regression_survival_svm.stepwise_newton_cg import SteppedEventBasedNewtonCgOptimizer
from smpc.helper import SMPCClient, MaskedDataDescription, SMPCRequest, SMPCMasked, MAX_RAND_INT, D_TYPE



class AppLogic:

    def __init__(self):
        # === Status of this app instance ===

        # Only relevant for coordinator, will stop execution when True
        self.status_finished = False

        # === Parameters set during setup ===
        self.id = None
        self.coordinator = None
        self.clients = None

        # === Communication ===
        self.communicator: Communication = JsonEncodedCommunication()

        # === Privacy ===
        self.enable_smpc = True

        # === Internals ===
        self.thread = None
        self.iteration = 0
        self.progress = 'not started yet'

        # === Custom ===
        self.INPUT_DIR = "/mnt/input"
        self.OUTPUT_DIR = "/mnt/output"

        self.models: Dict[str, Union[Client, Coordinator]] = {}
        self.smpc_client: Optional[SMPCClient] = None

        self.train_filename = None
        self.test_filename = None

        self.model_output = None
        self.meta_output = None
        self.pred_output = None
        self.test_output = None
        self.train_output = None

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
        self.train_data_paths: Dict[str, str] = {}
        self.svm: Dict[str, FastSurvivalSVM] = {}
        self.training_states: Optional[Dict[str, Dict[str, Any]]] = None
        self.last_requests = None

        self.timings: Dict[str, float] = collections.defaultdict(float)

        self._log_f_val = collections.defaultdict(dict)

    def read_config(self):
        with open(os.path.join(self.INPUT_DIR, "config.yml")) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_survival_svm']
            self.enable_smpc = config['privacy']['enable_smpc']

            self.train_filename = config['input']['train']
            self.test_filename = config['input']['test']

            self.model_output = config['output']['model']
            self.pred_output = config['output']['pred']
            self.meta_output = config['output'].get("meta", "meta.yml")  # default value
            self.train_output = config['output'].get("train", self.train_filename)  # default value
            self.test_output = config['output'].get("test", self.test_filename)  # default value

            self.sep = config['format']['sep']
            self.label_time_to_event = config["format"]["label_survival_time"]
            self.label_event = config["format"]["label_event"]
            self.event_truth_value = config["format"].get("event_truth_value", True)  # default value
            logging.debug(f"EVENT TRUTH VALUE: {self.event_truth_value}")

            self.mode = config['split']['mode']
            self.dir = config['split']['dir']

            self.alpha = config['svm']['alpha']
            self.fit_intercept = config['svm']['fit_intercept']
            self.max_iter = config['svm']['max_iterations']

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

    @staticmethod
    def get_column(dataframe: pd.DataFrame, col_name: str) -> pd.Series:
        try:
            return dataframe[col_name]
        except KeyError as e:
            logging.error(f"Column {col_name} does not exist in the data")
            raise e

    @staticmethod
    def event_value_to_truth_array(event: NDArray[Any], truth_value: Any) -> NDArray[Bool]:
        logging.debug(f"event: {event}")
        if truth_value is True and event.dtype == np.dtype('bool'):  # nothing to do...
            logging.debug("NOTHING TO DO")
            return event

        truth_array = (event == truth_value)
        logging.debug(f"truth {truth_array}")
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
        self.communicator.init(self.coordinator, len(clients))
        logging.info(f'Received setup: {self.id} {self.coordinator} {self.clients}')

        self.thread = threading.Thread(target=self.guarded_app_flow)
        self.thread.start()

    @property
    def status_available(self):
        return self.communicator.status_available

    def handle_incoming(self, data):
        self.communicator.handle_incoming(data)

    def handle_outgoing(self):
        return self.communicator.handle_outgoing()

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

    def _get_states(self, requests: Dict[str, Optional[SteppedEventBasedNewtonCgOptimizer.Request]]):
        states = dict.fromkeys(requests.keys())
        for split_name, request in requests.items():
            if request is None:
                states[split_name] = {"state": "Finished"}
            else:
                states[split_name] = {"state": "Training"}
        return states


    def _fulfill_all_requests_local(self, requests: Dict[str, Optional[SteppedEventBasedNewtonCgOptimizer.Request]],
                                    models: Dict[str, Coordinator]):
        local_results: Dict[str, Optional[LocalResult]] = dict.fromkeys(models.keys())
        for model_identifier, request in requests.items():
            if request is None:
                continue
            model: Client = models[model_identifier]
            local_results[model_identifier] = model.handle_computation_request(request)
        return local_results

    def _fulfill_all_requests_local_smpc(self, requests: Dict[str, Optional[SteppedEventBasedNewtonCgOptimizer.Request]],
                                         models: Dict[str, Coordinator]):
        local_results: Dict[str, Optional[SMPCMasked]] = dict.fromkeys(models.keys())
        for model_identifier, request in requests.items():
            if request is None:
                continue
            model: Client = models[model_identifier]
            local_results[model_identifier] = model.handle_computation_request_smpc(request, self.smpc_client.pub_keys_of_other_parties)
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

    def _fulfill_all_requests_global_smpc(self, local_results: Dict[str, Optional[LocalResult]],
                                     models: Dict[str, Coordinator]):
        aggregated_results: Dict[str, Optional[SteppedEventBasedNewtonCgOptimizer.Resolved]] = dict.fromkeys(
            models.keys())
        for model_identifier, local_result in local_results.items():
            if local_result is None:
                continue
            model: Coordinator = models[model_identifier]
            aggregated_results[model_identifier] = model.aggregate_local_result_smpc(local_result)
        return aggregated_results

    def _inform_all(self, aggregated_results: Dict[str, Optional[SteppedEventBasedNewtonCgOptimizer.Resolved]],
                    models: Dict[str, Coordinator]):
        for model_identifier, aggregated_result in aggregated_results.items():
            if aggregated_result is None:
                continue
            model: Coordinator = models[model_identifier]
            optimizer: SteppedEventBasedNewtonCgOptimizer = model.newton_optimizer
            optimizer.resolve(aggregated_result)

    def _fulfil_smpc_sum_up_masks(self, request: SMPCRequest):
        summed_masks = dict.fromkeys(self.splits.keys())
        for split in self.splits.keys():
            if request.data[split] is not None:
                mask = request.data[split][self.id]
                summed_masks[split] = self.smpc_client.sum_encrypted_masks_up(mask)
            else:
                summed_masks[split] = None
        return summed_masks

    def guarded_app_flow(self):
        """Adds error handling to app flow."""
        try:
            self.app_flow()
        except Exception as e:  # gonna catch 'em all
            self.status_finished = True
            self.communicator.clear()
            self.communicator.broadcast(Exception(f'app_flow of client {self.id} failed'))
            raise e

    def app_flow(self):
        # This method contains a state machine for the client and coordinator instance

        # === States ===
        state_initializing = 1
        state_read_input = 2
        state_smpc_send_public_key = 3.0
        state_smpc_aggregate_public_keys = 3.1
        state_smpc_set_pubkeys = 3.2
        state_preprocessing = 4
        state_send_data_attributes = 5
        state_global_aggregation_of_data_attributes = 6
        state_global_optimization = 7
        state_local_optimization_calculation_requests_listener = 8
        state_check_success_and_try_recover = 9.0
        state_send_global_model = 9.1
        state_set_global_model = 10
        state_generate_predictions = 11
        state_shutdown = 12

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
                tic = time.perf_counter()
                self.read_config()
                toc = time.perf_counter()
                self.timings['read_config'] = toc - tic

                self.progress = 'reading data...'
                tic = time.perf_counter()
                for split in self.splits.keys():
                    logging.info(f'Read {split} data')
                    if self.coordinator:
                        self.models[split] = Coordinator()
                    else:
                        self.models[split] = Client()

                    train_data_path = os.path.join(split, self.train_filename)
                    self.splits[split] = self.read_survival_data(train_data_path)
                    self.train_data_paths[split] = train_data_path
                toc = time.perf_counter()
                self.timings['read_data'] = toc - tic

                if self.enable_smpc:
                    state = state_smpc_send_public_key
                else:
                    state = state_preprocessing

            if state == state_smpc_send_public_key:
                tic = time.perf_counter()
                logging.debug(f"SMPC SEED: {self._smpc_seed}")
                self.smpc_client = SMPCClient(self.id, random_seed=self._smpc_seed)
                toc = time.perf_counter()
                self.timings['smpc_init'] = toc - tic
                self.communicator.send_to_coordinator({'client': self.id, 'pub_key': self.smpc_client.pub_key})

                if self.coordinator:
                    state = state_smpc_aggregate_public_keys
                else:
                    state = state_smpc_set_pubkeys

            if state == state_smpc_aggregate_public_keys:
                public_keys = self.communicator.wait_for_data_from_all()
                logging.debug(public_keys)

                for pubkey_info in public_keys:
                    logging.debug(pubkey_info)
                    self.smpc_client.add_party_pub_key(pubkey_info['client'], pubkey_info['pub_key'])

                self.communicator.broadcast(self.smpc_client.pub_keys_of_other_parties)

                state = state_smpc_set_pubkeys

            if state == state_smpc_set_pubkeys:
                pub_keys = self.communicator.wait_for_data()
                logging.debug(pub_keys)

                self.smpc_client.add_party_pub_key_dict(pub_keys)

                state = state_preprocessing

            if state == state_preprocessing:
                self.progress = 'preprocessing...'
                tic = time.perf_counter()
                for split in self.splits.keys():
                    logging.info(f'Preprocess {split}')
                    model = self.models[split]
                    X, y = self.splits[split]
                    model.set_config(
                        SharedConfig(alpha=self.alpha, fit_intercept=self.fit_intercept,
                                     max_iter=self.max_iter))  # TODO: check if nodes agree on config
                    model.set_data(SurvivalData(X, y))
                    model.log_transform_times()  # run log transformation of time values for regression objective
                toc = time.perf_counter()
                self.timings['preprocessing'] = toc - tic
                state = state_send_data_attributes

            if state == state_send_data_attributes:
                self.progress = "generating data descriptions..."
                data_to_send = {}
                for split in self.splits.keys():
                    logging.info(f'Generate data attributes for split {split}')
                    data_description = self.models[split].generate_data_description()
                    logging.debug(data_description)
                    if self.enable_smpc:
                        data_to_send[split] = MaskedDataDescription().mask(data_description, self.smpc_client.pub_keys_of_other_parties)
                    else:
                        data_to_send[split] = data_description

                logging.debug(f"Data attributes:\n{data_to_send}")
                self.communicator.send_to_coordinator(data_to_send)

                if self.coordinator:
                    state = state_global_aggregation_of_data_attributes
                else:
                    state = state_local_optimization_calculation_requests_listener
                    logging.info(f'[CLIENT] Sending attributes to coordinator')

            if state == state_global_aggregation_of_data_attributes:
                self.progress = f'waiting for data attributes...'

                if self.enable_smpc:
                    data = self.communicator.wait_for_data_from_all()
                    self.progress = 'getting masked data attributes...'

                    aggregated: Dict[str, MaskedDataDescription] = dict.fromkeys(self.splits.keys())
                    masks = dict.fromkeys(self.splits.keys())
                    for split in self.splits.keys():
                        logging.debug(f'Aggregate {split}')
                        masked_result: MaskedDataDescription = data[0][split]
                        for i in range(1, len(data)):
                            masked_result += data[i][split]

                        aggregated[split] = masked_result
                        masks[split] = masked_result.encrypted_masks

                    logging.debug(aggregated)
                    logging.debug(masks)

                    # get aggregated masks
                    self.communicator.broadcast(SMPCRequest(masks))
                    # fulfill at coordinator
                    request: SMPCRequest = self.communicator.wait_for_data()
                    summed_up_masks_local = self._fulfil_smpc_sum_up_masks(request)
                    self.communicator.send_to_coordinator(summed_up_masks_local)


                    results: List[Dict[str, np.array]] = self.communicator.wait_for_data_from_all()
                    logging.debug(results)
                    masks_for_split = collections.defaultdict(list)
                    for split in self.splits.keys():
                        for local_res in results:
                            masks_for_split[split].append(local_res[split])
                    logging.debug(masks_for_split)

                    self.progress = 'setting data description...'
                    for split in self.splits.keys():
                        logging.debug(masks_for_split[split])
                        mask_sum = masks_for_split[split][0]
                        for i in range(1, len(masks_for_split[split])):
                            mask_sum += masks_for_split[split][i]
                        logging.debug(mask_sum)
                        logging.debug(aggregated[split])
                        data_description = aggregated[split].unmasked_obj(mask_sum)

                        logging.debug(f"Data description at split {split}: {data_description}")

                        self.models[split].set_data_description(data_description)
                        self.models[split].set_initial_w_and_init_optimizer()

                else:
                    data = self.communicator.wait_for_data_from_all()
                    self.progress = 'setting data description...'

                    for split in self.splits.keys():
                        logging.debug(f'Aggregate {split}')
                        split_data = []
                        for client in data:
                            split_data.append(client[split])
                        logging.debug(f"Data description at split {split}: {split_data}")
                        self.models[split].aggregate_and_set_data_descriptions(split_data)
                        self.models[split].set_initial_w_and_init_optimizer()

                requests = self._get_all_requests(self.models)

                logging.debug(requests)
                self.communicator.broadcast(requests)
                logging.info(f'[COORDINATOR] Sending initial calculation requests')

                state = state_local_optimization_calculation_requests_listener

            if state == state_global_optimization:
                self.progress = f'waiting for results of calculation requests...'
                data = self.communicator.wait_for_data_from_all()
                self.progress = 'starting global calculation...'

                if self.smpc_client:
                    local_results: Dict[str, Optional[List[SMPCMasked]]] = {k: None for k, v in self.splits.items()}
                    for split in self.splits.keys():
                        logging.debug(f'Aggregate {split}')
                        split_data = []
                        for client in data:
                            split_data.append(client[split])
                        logging.debug(f"Result for {split}: {split_data}")

                        local_results[split] = split_data

                    aggregated: Dict[str, Optional[SMPCMasked]] = dict.fromkeys(self.splits.keys())
                    masks = dict.fromkeys(self.splits.keys())
                    for split in self.splits.keys():
                        logging.debug(f'Aggregate {split}')
                        masked_result: SMPCMasked = local_results[split][0]
                        if masked_result is not None:
                            for i in range(1, len(local_results[split])):
                                masked_result += local_results[split][i]

                            aggregated[split] = masked_result
                            masks[split] = masked_result.encrypted_masks
                        else:
                            aggregated[split] = None
                            masks[split] = None

                    logging.debug(aggregated)
                    logging.debug(masks)

                    # get aggregated masks
                    self.communicator.broadcast(SMPCRequest(masks))
                    # fulfill at coordinator
                    request: SMPCRequest = self.communicator.wait_for_data()
                    summed_up_masks_local = self._fulfil_smpc_sum_up_masks(request)
                    self.communicator.send_to_coordinator(summed_up_masks_local)

                    results: List[Dict[str, np.array]] = self.communicator.wait_for_data_from_all()
                    logging.debug(f"Masking request result: {results}")
                    masks_for_split = collections.defaultdict(list)
                    for split in self.splits.keys():
                        for local_res in results:
                            masks_for_split[split].append(local_res[split])
                    logging.debug(f"Masks for splits: {masks_for_split}")

                    local_results: Dict[str, Optional[List[LocalResult]]] = dict.fromkeys(self.splits.keys())
                    for split in self.splits.keys():
                        logging.debug(masks_for_split[split])
                        mask_sum = masks_for_split[split][0]
                        if mask_sum is not None:
                            for i in range(1, len(masks_for_split[split])):
                                mask_sum += masks_for_split[split][i]
                            logging.debug(f"Mask sum: {mask_sum}")
                            logging.debug(aggregated[split])
                            unmasked = aggregated[split].unmasked_obj(mask_sum)

                            logging.debug(f"Unmasked result at split {split}: {unmasked}")
                            local_results[split] = unmasked
                        else:
                            local_results[split] = None
                else:
                    local_results: Dict[str, Optional[List[LocalResult]]] = {k: None for k, v in self.splits.items()}
                    for split in self.splits.keys():
                        logging.debug(f'Aggregate {split}')
                        split_data = []
                        for client in data:
                            split_data.append(client[split])
                        logging.debug(f"Result for {split}: {split_data}")

                        local_results[split] = split_data

                logging.debug(f"Local results: {local_results}")
                if self.enable_smpc:
                    aggregated: Dict[str, Optional[SteppedEventBasedNewtonCgOptimizer.Resolved]] = self._fulfill_all_requests_global_smpc(local_results, self.models)
                else:
                    aggregated: Dict[str, Optional[SteppedEventBasedNewtonCgOptimizer.Resolved]] = self._fulfill_all_requests_global(local_results, self.models)

                for split in self.splits.keys():
                    resolved = aggregated[split]
                    if isinstance(resolved, SteppedEventBasedNewtonCgOptimizer.ResolvedWDependent):
                        resolved: SteppedEventBasedNewtonCgOptimizer.ResolvedWDependent
                        self._log_f_val[split][datetime.datetime.now().timestamp()] = resolved.f_val

                logging.debug(f"Aggregated: {aggregated}")
                self._inform_all(aggregated, self.models)

                requests = self._get_all_requests(self.models)
                logging.debug(f"Requests: {requests}")

                if self._all_finished(self.models):
                    logging.info('Optimization for all splits finished')
                    state = state_check_success_and_try_recover
                else:
                    self.communicator.broadcast(requests)
                    logging.info(f'[COORDINATOR] Sending calculation requests')
                    state = state_local_optimization_calculation_requests_listener

            if state == state_local_optimization_calculation_requests_listener:
                self.progress = 'waiting for calculation requests...'
                requests = self.communicator.wait_for_data()
                self.progress = 'processing calculation requests...'

                self.iteration += 1

                if isinstance(requests, OptFinished):
                    logging.debug("Received OptFinished signal")
                    state = state_set_global_model
                    self.opt_finished_signal: OptFinished = requests
                elif isinstance(requests, SMPCRequest):  # decrypt and sum up masks
                    results: Dict[str, Any] = self._fulfil_smpc_sum_up_masks(requests)

                    logging.debug(f"Local results:\n{results}")
                    self.communicator.send_to_coordinator(results)

                    state = state_local_optimization_calculation_requests_listener
                    logging.info(f'[CLIENT] Sending summed masks to coordinator')
                else:
                    self.training_states = self._get_states(requests)
                    self.last_requests = requests
                    if self.smpc_client:
                        results: Dict[str, Optional[SMPCMasked]] = self._fulfill_all_requests_local_smpc(requests, self.models)
                    else:
                        results: Dict[str, Optional[LocalResult]] = self._fulfill_all_requests_local(requests, self.models)

                    logging.debug(f"Local results:\n{results}")
                    self.communicator.send_to_coordinator(results)

                    if self.coordinator:
                        state = state_global_optimization
                    else:
                        state = state_local_optimization_calculation_requests_listener
                        logging.info(f'[CLIENT] Sending attributes to coordinator')

            if state == state_check_success_and_try_recover:
                # check all successful
                if self._tries_recover > 0:
                    needs_recover = False
                    for split in self.splits.keys():
                        model: Coordinator = self.models[split]
                        optimizer: SteppedEventBasedNewtonCgOptimizer = model.newton_optimizer
                        result: OptimizeResult = optimizer.result
                        if not result.success:
                            needs_recover = True
                            self.models[split].set_initial_w_and_init_optimizer(w=result.x)

                    if needs_recover:
                        logging.debug("Trying to recover from failure!")
                        self._tries_recover -= 1
                        requests = self._get_all_requests(self.models)

                        logging.debug(requests)
                        self.communicator.broadcast(requests)
                        logging.info(f'[COORDINATOR] Sending initial calculation requests')

                        state = state_local_optimization_calculation_requests_listener
                    else:
                        state = state_send_global_model
                else:
                    state = state_send_global_model

            if state == state_send_global_model:
                self.progress = 'optimization finished...'
                opt_results: Dict[str, OptimizeResult] = {k: None for k, v in self.splits.items()}
                for split in self.splits.keys():
                    model: Coordinator = self.models[split]
                    optimizer: SteppedEventBasedNewtonCgOptimizer = model.newton_optimizer
                    opt_results[split] = optimizer.result
                logging.debug(f'Optimizers: {opt_results}')

                self.communicator.broadcast(OptFinished(opt_results=opt_results))

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
                    meta_output_path = os.path.join(split.replace("/input", "/output"), self.meta_output)
                    logging.debug(f"Writing metadata to {meta_output_path}")

                    # unpack coefficients
                    beta = {}
                    features = self.splits[split][0].columns.values
                    weights = sksurv_obj.coef_.tolist()
                    for feature_name, feature_weight in zip(features, weights):
                        beta[feature_name] = feature_weight
                    bias = None
                    if self.fit_intercept is not None:
                        bias = float(sksurv_obj.intercept_)

                    # unpack timings
                    opt_times = opt_result.timings['py/seq']  # TODO. Why is this a dict? Failed JSON parsing?
                    timings = {
                        "optimizer": {
                            "calculation_time": opt_times[0],
                            "total_time": opt_times[1],
                            "idle_time": opt_times[2],
                        }
                    }
                    for k,v in self.timings.items():
                        timings[k] = v

                    # privacy
                    privacy = {"privacy_technique": "SMPC" if self.enable_smpc else "None"}
                    if self.enable_smpc:
                        privacy["SMPC_MAX_RAND_INT"] = MAX_RAND_INT
                        privacy["MASK_DTYPE"] = D_TYPE.__name__

                    metadata = {
                        "model": {
                            "name": "FederatedPureRegressionSurvivalSVM",
                            "version": "v0.1.7-alpha",
                            "privacy": privacy,
                            "training_parameters": {
                                "alpha": sksurv_obj.alpha,
                                "rank_ratio": sksurv_obj.rank_ratio,
                                "fit_intercept": sksurv_obj.fit_intercept,
                                "max_iter": self.max_iter,
                            },
                            "optimizer": {
                                "success": opt_result.success,
                                "status": opt_result.status,
                                "message": opt_result.message,
                                "fun": float(opt_result.fun),
                                "nit": opt_result.nit,
                                "nfev": opt_result.nfev,
                                "njev": opt_result.njev,
                                "nhev": opt_result.nhev,
                                "x": opt_result.x.tolist(),
                                "jac": opt_result.jac.tolist(),
                            },
                            "coefficients": {
                                "weights": beta,
                                "intercept": bias,
                            },
                            "training_data": {
                                "file": self.train_data_paths[split].replace(self.INPUT_DIR, "."),
                                "label_survival_time": self.label_time_to_event,
                                "label_event": self.label_event,
                                "event_truth_value": self.event_truth_value,
                            },
                            "timings": timings,
                        },
                    }

                    with open(meta_output_path, "w") as fh:
                        yaml.dump(metadata, fh)

                    # write convergence info
                    if len(self._log_f_val) != 0:
                        f_val_track_output_path = os.path.join(split.replace("/input", "/output"), 'f_val_track.csv')
                        with open(f_val_track_output_path, "w") as fh:
                            fh.write(f"timestamp,f_val\n")
                            for timestamp, f_val in self._log_f_val[split].items():
                                fh.write(f"{timestamp},{f_val}\n")

                    # copy inputs to outputs
                    output_split_dir = split.replace(self.INPUT_DIR, self.OUTPUT_DIR)
                    shutil.copyfile(os.path.join(split, self.train_filename),
                                    os.path.join(output_split_dir, self.train_output))
                    shutil.copyfile(os.path.join(split, self.test_filename),
                                    os.path.join(output_split_dir, self.test_output))

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
