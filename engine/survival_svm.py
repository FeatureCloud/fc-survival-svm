import logging
import os
import pickle
import shutil
import time
from time import sleep
from typing import Type, Dict, Optional, List, Union, Tuple

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import yaml
from scipy.optimize import OptimizeResult
from sksurv.svm import FastSurvivalSVM

import logic.data
from app.survival_svm import settings
from app.survival_svm.settings import INPUT_DIR, CONFIG_FILE_NAME, OUTPUT_DIR
from engine.app import AppState, STATE_RUNNING, STATE_ACTION, STATE_ERROR, PARTICIPANT, COORDINATOR
from engine.exchanged_parameters import SyncSignalClient, SyncSignalCoordinator
from logic.config import Config, Parameters
from logic.data import read_survival_data, SurvivalData
from logic.model import Training, LocalTraining
from logic.splits import SplitManager
from optimization.stepwise_newton_cg import SteppedEventBasedNewtonCgOptimizer

jsonpickle_numpy.register_handlers()


class BlankState(AppState):

    def __init__(self, next_state=None):
        super().__init__()
        self.next_state = next_state

    def register(self):
        if self.next_state:
            self.register_transition(self.next_state, role=(self.participant, self.coordinator))

    def run(self):
        return self.next_state


class DependingOnTypeState(AppState):

    def __init__(self, next_state_participant, next_state_coordinator):
        super().__init__()
        self.next_state_participant = next_state_participant
        self.next_state_coordinator = next_state_coordinator

    def register(self):
        self.register_transition(self.next_state_participant, role=PARTICIPANT)
        self.register_transition(self.next_state_coordinator, role=COORDINATOR)

    def run(self):
        if self.app.coordinator:
            return self.next_state_coordinator
        else:
            return self.next_state_participant


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
        config_file_path: str = os.path.join(INPUT_DIR, CONFIG_FILE_NAME)
        if os.path.isfile(config_file_path):
            with open(config_file_path) as f:
                self.app.log(f'Config file at {config_file_path} exists', level=logging.DEBUG)
                config_yml = yaml.load(f, Loader=yaml.FullLoader)
                if self.section in config_yml:
                    self.app.log(f'Section {self.section} exists in config file', level=logging.WARNING)
                    return self.next_state_exists
                else:
                    self.app.log(f'Section {self.section} is missing in config file', level=logging.WARNING)
        else:
            self.app.log(f'No config file at {config_file_path} found', level=logging.WARNING)
        return self.next_state_missing


class ConfigFileState(DependingOnTypeState):

    def __init__(self, next_state_participant, next_state_coordinator, section):
        super().__init__(next_state_participant, next_state_coordinator)
        self.section = section

    def run(self):
        self.update(
            message=f'Reading config file',
            progress=0.03,
            state=STATE_RUNNING
        )

        self.app.internal['_tic_config'] = time.perf_counter()

        config_file_path: str = os.path.join(INPUT_DIR, CONFIG_FILE_NAME)
        config: Config = Config.from_file(config_file_path)
        self.app.internal['config'] = config

        # copy config file to output folder
        shutil.copyfile(config_file_path, os.path.join(OUTPUT_DIR, CONFIG_FILE_NAME))

        return super().run()


class ConfigWebState(DependingOnTypeState):

    def __init__(self, next_state_participant, next_state_coordinator, recheck_period=0.5):
        super().__init__(next_state_participant, next_state_coordinator)
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


class BroadcastConfigState(BlankState):

    def run(self):
        self.update(message='Broadcasting config', progress=0.04)
        config = self.app.internal.get('config')

        config_to_share = {
            'parameters': config.parameters,
        }
        self.app.log(f'Sharing config: {config_to_share}', level=logging.DEBUG)

        self.broadcast_data(jsonpickle.encode(config_to_share).encode())
        return super().run()


class ReceiveConfigState(BlankState):

    def run(self):
        self.update(message='Receiving shared config', progress=0.05)
        config: Config = self.app.internal.get('config')

        shared_config: Dict = self.await_data()
        shared_config = jsonpickle.decode(shared_config)

        config.parameters = shared_config.get('parameters')
        self.app.log(config.parameters, level=logging.DEBUG)

        return super().run()


class InitSplitManager(BlankState):

    def run(self):
        self.update(message='Initialize split manager', progress=0.06)
        config = self.app.internal.get('config')

        self.app.log(f'MODE: {config.mode}', level=logging.DEBUG)
        self.app.log(f'DIR: {config.dir}', level=logging.DEBUG)
        split_manager = SplitManager(mode=config.mode, directory_name=config.dir)
        self.app.internal['split_manager'] = split_manager
        self.app.log(self.app.internal.get('split_manager'), level=logging.DEBUG)

        for split in split_manager:
            split.data['tries'] = config.tries_recover

        toc = time.perf_counter()
        self.app.internal['timing_config'] = toc - self.app.internal['_tic_config']

        return super().run()


class ReadDataState(BlankState):

    def run(self):
        self.update(message='Read data', progress=0.07)

        tic = time.perf_counter()

        config: Config = self.app.internal.get('config')
        parameters: Parameters = config.parameters
        split_manager = self.app.internal.get('split_manager')
        for split in split_manager:
            training_data_path = os.path.join(split.input_dir, config.train_filename)
            if not os.path.isfile(training_data_path):
                raise RuntimeError(f'No training data at {training_data_path}')

            training_data: SurvivalData = read_survival_data(
                training_data_path, sep=config.sep,
                label_event=config.label_event, label_time_to_event=config.label_time_to_event,
                event_truth_value=config.event_truth_value)

            split.data['training_data'] = training_data

            # get some metrics about data
            n_samples = training_data.n_samples
            n_censored = training_data.n_censored
            self.app.log(split.name, level=logging.DEBUG)
            self.app.log(f'samples = {n_samples}', level=logging.DEBUG)
            self.app.log(f'censored = {n_censored}', level=logging.DEBUG)

            opt_out = n_samples < config.min_samples
            split.data['opt_out'] = opt_out
            split.data['opt_finished'] = False
            if opt_out:
                self.app.log(f'Number of samples is below safe limit for split {split.input_dir}. Opt-out.',
                             level=logging.WARNING)

        toc = time.perf_counter()
        self.app.internal['timing_read_data'] = toc - tic

        return super().run()


class PreprocessDataState(BlankState):

    def run(self):
        self.update(message='Check times for zero and neg. timepoints', progress=0.08)
        split_manager = self.app.internal.get('split_manager')
        for split in split_manager:
            if not split.data.get('opt_out'):
                training_data: SurvivalData = split.data.get('training_data')

                dropped_rows = training_data.drop_negative_and_zero_timepoints()
                if dropped_rows > 0:
                    self.update(
                        message=f'Dropped {dropped_rows} samples with zero or negative timepoints in {split.name}',
                        state=STATE_RUNNING
                    )

        self.update(message='Log-transform survival times', progress=0.08)

        tic = time.perf_counter()

        split_manager = self.app.internal.get('split_manager')
        for split in split_manager:
            if not split.data.get('opt_out'):
                training_data: SurvivalData = split.data.get('training_data')

                try:
                    training_data.log_transform_times()
                except ValueError as e:
                    self.app.log(str(e), level=logging.WARNING)
                    split.data['opt_out'] = True
                    split.data['opt_out_message'] = str(e)

        toc = time.perf_counter()
        self.app.internal['timing_preprocessing'] = toc - tic

        config: Config = self.app.internal.get('config')
        parameters: Parameters = config.parameters
        for split in split_manager:
            if not split.data.get('opt_out'):
                # generate model attribute
                if self.app.coordinator:
                    model_cls = Training
                else:
                    model_cls = LocalTraining

                model = model_cls(data=split.data['training_data'],
                                  alpha=parameters.alpha,
                                  fit_intercept=parameters.fit_intercept,
                                  max_iter=parameters.max_iter)
                split.data['model'] = model

                del split.data['training_data']

        return super().run()


class SendDataAttributes(DependingOnTypeState):

    def run(self):
        self.update(message='Sending data attributes', progress=0.09)
        config: Config = self.app.internal.get('config')

        split_manager: SplitManager = self.app.internal.get('split_manager')
        data_attributes = {}
        for split in split_manager:

            if not split.data.get('opt_out'):
                model: LocalTraining = split.data.get('model')
                training_data: SurvivalData = model.data

                data_attributes[split.name] = {
                    'n_samples': training_data.n_samples,
                    'n_censored': training_data.n_censored,
                    'sum_of_times': training_data.sum_of_times,
                }
            else:
                data_attributes[split.name] = None

        json_encoded_str = jsonpickle.encode(data_attributes)
        json_encoded_bytes = json_encoded_str.encode()
        self.app.log(json_encoded_str, level=logging.DEBUG)
        self.send_data_to_coordinator(json_encoded_bytes, use_smpc=config.enable_smpc)

        if self.app.coordinator:
            return self.next_state_coordinator
        else:
            self.update(message='Waiting for initial weights', progress=0.10)
            return self.next_state_participant


class InitOptimizer(BlankState):

    def run(self):
        self.update(message='Waiting for aggregated data attributes', progress=0.10)
        config: Config = self.app.internal.get('config')
        parameters: Parameters = config.parameters
        fit_intercept = parameters.fit_intercept

        split_manager: SplitManager = self.app.internal.get('split_manager')

        if config.enable_smpc:
            smpc_data_attributes = jsonpickle.decode(self.await_data())
        else:
            data: List[Tuple[str, str]] = self.gather_data()
            self.app.log(f'data: {data}')
            data: List[Dict[str, Dict[str, Union[int, float]]]] = [jsonpickle.decode(d[0]) for d in data]
            self.app.log(f'data_decoded: {data}')
            smpc_data_attributes = split_manager.add_int_values_in_split_dictionaries(data)

        self.app.log(smpc_data_attributes, level=logging.DEBUG)

        self.update(message='Initialize optimization', progress=0.10)

        for split in split_manager:
            attributes = smpc_data_attributes.get(split.name)

            # check if all clients opted-out
            if attributes is None:
                split.data['opt_out'] = True
                split.data['opt_out_message'] = 'All clients opted-out.'
                continue

            # check if any client has uncensored data
            aggregated_n_samples = round(attributes.get('n_samples'))
            aggregated_n_censored = round(attributes.get('n_censored'))
            if aggregated_n_samples - aggregated_n_censored <= 0:  # none of the clients has uncensored samples
                split.data['opt_out'] = True
                split.data['opt_out_message'] = 'None of the clients has uncensored data for this split.'
                continue
            else:
                split.data['n_samples'] = aggregated_n_samples
                split.data['n_censored'] = aggregated_n_censored

            # getting initial guess for weights
            model: Training = split.data.get('model')
            training_data: SurvivalData = model.data
            local_n_features = training_data.n_features
            split.data['n_features'] = local_n_features

            aggregated_sum_of_times = attributes.get('sum_of_times')
            mean_time = aggregated_sum_of_times / aggregated_n_samples

            initial_weight = model.get_initial_w(n_features=local_n_features,
                                                 mean_time_to_event=mean_time)

            optimizer: SteppedEventBasedNewtonCgOptimizer = SteppedEventBasedNewtonCgOptimizer(
                x0=initial_weight,
                maxiter=parameters.max_iter
            )
            split.data['optimizer'] = optimizer

        return super().run()


class SendAggregatedDataAttributes(BlankState):

    def run(self):
        self.update(message=f'Sending aggregated data attributes')
        aggregated_attributes = {}

        split_manager: SplitManager = self.app.internal.get('split_manager')
        for split in split_manager:
            aggregated_attributes[split.name] = {
                'n_samples': split.data.get('n_samples'),
                'n_censored': split.data.get('n_censored'),
            }

        self.broadcast_data(jsonpickle.encode(aggregated_attributes).encode())
        return super().run()


class SaveAggregatedDataAttributes(DependingOnTypeState):

    def run(self):
        self.update(message=f'Waiting for aggregated data attributes')
        aggregated_attributes = jsonpickle.decode(self.await_data())
        self.update(message=f'Setting aggregated data attributes')

        split_manager: SplitManager = self.app.internal.get('split_manager')
        for split in split_manager:
            attributes = aggregated_attributes[split.name]

            split.data['n_samples'] = attributes.get('n_samples')
            split.data['n_censored'] = attributes.get('n_censored')

        return super().run()


class SendRequest(BlankState):

    def run(self):
        config: Config = self.app.internal.get('config')

        n_exchange = self.app.internal.get('round')
        self.update(message=f'Getting computation requests [{n_exchange}]')

        requests = {}

        split_manager: SplitManager = self.app.internal.get('split_manager')
        for split in split_manager:
            self.app.log(split)
            if split.data.get('opt_finished'):
                self.app.log(f'continue')
                continue

            optimizer: SteppedEventBasedNewtonCgOptimizer = split.data['optimizer']
            if not optimizer.finished:
                self.app.log(f'check pending')
                requests[split.name] = optimizer.check_pending_requests()
                self.app.log(f'after check pending')
            else:
                self.app.log(f'result')
                result: OptimizeResult = optimizer.result
                self.app.log(f'{result}')
                if not result.success:  # trying to recover from failure
                    self.app.log(f'failure')
                    parameters: Parameters = config.parameters
                    if split.data.get('tries', 0) > 0 and result.nit < parameters.max_iter:
                        self.app.log(f'recover')
                        split.data['tries'] -= 1
                        self.app.log(f'Trying to recover {split.name}')
                        optimizer: SteppedEventBasedNewtonCgOptimizer = SteppedEventBasedNewtonCgOptimizer(
                            x0=result.x,
                            maxiter=parameters.max_iter - result.nit
                        )

                        self.app.log(f'get timings')
                        opt_times = result.timings
                        self.app.log(opt_times)
                        opt_times_list = [opt_times[0], opt_times[1], opt_times[2]]
                        self.app.log(opt_times_list)
                        old_times = split.data.get('timings_from_recovered_runs', (0, 0, 0))
                        opt_times_list[0] += old_times[0]
                        opt_times_list[1] += old_times[1]
                        opt_times_list[2] += old_times[2]
                        split.data['timings_from_recovered_runs'] = opt_times_list
                        self.app.log(f'after get timings')

                        self.app.log(f'recover check pending')
                        requests[split.name] = optimizer.check_pending_requests()
                        split.data['optimizer'] = optimizer
                        self.app.log(f'after recover check pending')
                    else:
                        self.app.log(f'else01')
                        requests[split.name] = optimizer.result
                else:
                    self.app.log(f'else02')
                    requests[split.name] = optimizer.result

        self.app.log(requests, level=logging.DEBUG)
        self.update(message=f'Sending computation requests [{n_exchange}]')
        self.broadcast_data(jsonpickle.encode(requests).encode())

        return super().run()


class ListenRequest(DependingOnTypeState):

    def __init__(self, next_state_participant, next_state_coordinator, next_state_opt_finished):
        super().__init__(next_state_participant, next_state_coordinator)
        self.next_state_opt_finished = next_state_opt_finished
        self.fed_round = 0

    def register(self):
        super().register()
        self.register_transition(self.next_state_opt_finished)

    def run(self):
        n_exchange = self.app.internal.get('round')
        self.update(message=f'Awaiting computation requests [{n_exchange}]')
        self.app.internal['round'] += 1

        config: Config = self.app.internal.get('config')

        requests: Dict[str, Optional[Dict[str, List[float]]]] = self.await_data()
        requests = jsonpickle.decode(requests)
        self.app.log(requests, level=logging.DEBUG)

        self.update(message=f'Fulfilling computation requests [{n_exchange}]')
        responses = {}

        all_finished = True
        split_manager: SplitManager = self.app.internal.get('split_manager')
        for split in split_manager:
            if split.data.get('opt_out'):  # do not generate response when we have opt-out
                continue
            if split.data.get('opt_finished'):  # ignore split when optimization for this split already finished
                continue
            else:
                all_finished = False

            model: LocalTraining = split.data.get('model')

            request = requests.get(split.name)
            if request is None:
                continue
            self.app.log(request, level=logging.DEBUG)

            if isinstance(request,
                          SteppedEventBasedNewtonCgOptimizer.RequestWDependent):  # request for calculating functions depending on w
                responses[split.name] = model._get_values_depending_on_w(request)
            elif isinstance(request, SteppedEventBasedNewtonCgOptimizer.RequestHessp):
                responses[split.name] = model._hessp_update(request)
            elif isinstance(request, OptimizeResult):
                split.data['opt_finished'] = True
                split.data['opt_result'] = request
                print('finished')

        if all_finished:
            return self.next_state_opt_finished

        self.app.log(responses, level=logging.DEBUG)
        self.update(message=f'Sending responses [{n_exchange}]')
        self.send_data_to_coordinator(jsonpickle.encode(responses).encode(), use_smpc=config.enable_smpc)
        return super().run()


class SetResponse(BlankState):

    def run(self):
        n_exchange = self.app.internal.get('round')
        self.update(message=f'Waiting for local updates [{n_exchange}]')

        config: Config = self.app.internal.get('config')
        split_manager: SplitManager = self.app.internal.get('split_manager')

        if config.enable_smpc:
            local_results = jsonpickle.decode(self.await_data())
        else:
            data: List[Tuple[str, str]] = self.gather_data()
            self.app.log(f'data: {data}')
            data: List[Dict[str, Tuple[float, List[float]]]] = [jsonpickle.decode(d[0]) for d in data]
            self.app.log(f'data_decoded: {data}')
            local_results = split_manager.add_answer_values_in_split_dictionaries(data)

        self.app.log(f'local_results: {local_results}')

        self.update(message=f'Aggregate [{n_exchange}]')
        for split in split_manager:
            result = local_results.get(split.name)
            if result is None:
                continue

            model: Training = split.data.get('model')
            optimizer: SteppedEventBasedNewtonCgOptimizer = split.data.get('optimizer')

            if isinstance(result, tuple):  # values dependent on w
                zeta_squared_sum, gradient_update = result
                resolved: SteppedEventBasedNewtonCgOptimizer.ResolvedWDependent = model.aggregate_f_val_and_g_val(
                    zeta_sq_sum=zeta_squared_sum, gradient_update=gradient_update)
                optimizer.resolve(resolved)
            elif isinstance(result, list):  # hessp
                hessp_update = result
                resolved: SteppedEventBasedNewtonCgOptimizer.ResolvedHessp = model.aggregated_hessp(
                    hessp_update=hessp_update)
                optimizer.resolve(resolved)

        return super().run()


class GeneratePredictions(BlankState):

    def run(self):
        self.update(message=f'Generate predictions on test data', progress=0.95)

        tic = time.perf_counter()

        config: Config = self.app.internal.get('config')

        split_manager: SplitManager = self.app.internal.get('split_manager')
        for split in split_manager:
            model: LocalTraining = split.data.get('model')
            opt_result: Optional[OptimizeResult] = split.data.get('opt_result')

            if opt_result is None:
                continue

            sksurv_obj: FastSurvivalSVM = model.to_sksurv(opt_result)
            split.data['svm'] = sksurv_obj
            svm: FastSurvivalSVM = sksurv_obj

            if svm is None:
                continue

            test_data_path = os.path.join(split.input_dir, config.test_filename)
            X_test, y_test = logic.data.read_survival_data_np(
                test_data_path, sep=config.sep,
                label_event=config.label_event, label_time_to_event=config.label_time_to_event,
                event_truth_value=config.event_truth_value)

            predictions = svm.predict(X_test)

            # re-add tte and event column
            X_test[config.label_time_to_event] = y_test['event_indicator']
            X_test[config.label_event] = y_test['time_to_event']
            # add predictions to dataframe
            X_test['predicted_tte'] = predictions.tolist()

            pred_output_path = os.path.join(split.name.replace(settings.INPUT_DIR, settings.OUTPUT_DIR),
                                            config.pred_output)
            self.app.log(f"Writing predictions to {pred_output_path}")
            X_test.to_csv(pred_output_path, sep=config.sep, index=False)

        toc = time.perf_counter()
        self.app.internal['timing_generate_predictions'] = toc - tic

        return super().run()


class WriteResult(BlankState):

    def run(self):
        self.update(message=f'Optimization finished. Writing results.', progress=0.975)
        config: Config = self.app.internal.get('config')

        # copy config file to outputs
        shutil.copyfile(os.path.join(settings.INPUT_DIR, settings.CONFIG_FILE_NAME),
                        os.path.join(settings.OUTPUT_DIR, settings.CONFIG_FILE_NAME))

        split_manager: SplitManager = self.app.internal.get('split_manager')
        for split in split_manager:
            # copy inputs to outputs
            shutil.copyfile(os.path.join(split.input_dir, config.train_filename),
                            os.path.join(split.output_dir, config.train_output))
            shutil.copyfile(os.path.join(split.input_dir, config.test_filename),
                            os.path.join(split.output_dir, config.test_output))

            # get model
            model: LocalTraining = split.data.get('model')
            opt_result: Optional[OptimizeResult] = split.data.get('opt_result')

            if opt_result is None:
                continue

            sksurv_obj: FastSurvivalSVM = split.data.get('svm')

            # write pickled model
            pickle_output_path = os.path.join(split.output_dir, config.model_output)
            self.app.log(f"Writing model to {pickle_output_path}")
            with open(pickle_output_path, "wb") as fh:
                pickle.dump(sksurv_obj, fh)

            # unpack coefficients
            beta = {}
            training_data: SurvivalData = model.data
            features = training_data.feature_names
            weights = sksurv_obj.coef_.tolist()
            for feature_name, feature_weight in zip(features, weights):
                beta[feature_name] = feature_weight
            bias = None
            if sksurv_obj.fit_intercept:
                bias = float(sksurv_obj.intercept_)

            # unpack timings
            opt_times = opt_result.timings['py/seq']  # TODO. Why is this a dict? Failed JSON parsing?
            timings_from_recovered_runs = split.data.get('timings_from_recovered_runs', (0, 0, 0))
            timings = {
                "total_until_meta_file_write": time.perf_counter() - self.app.internal['_tic_total'],
                "config": self.app.internal['timing_config'],
                "read_data": self.app.internal['timing_read_data'],
                "preprocessing": self.app.internal['timing_preprocessing'],
                "optimizer": {
                    "calculation_time": opt_times[0] + timings_from_recovered_runs[0],
                    "total_time": opt_times[1] + timings_from_recovered_runs[1],
                    "idle_time": opt_times[2] + timings_from_recovered_runs[2],
                },
                "generate_predictions": self.app.internal['timing_generate_predictions'],
            }

            # privacy
            privacy = {"privacy_technique": "SMPC" if config.enable_smpc else "None"}

            metadata = {
                "model": {
                    "name": "FederatedPureRegressionSurvivalSVM",
                    "version": "v1.0.0-alpha",
                    "privacy": privacy,
                    "training_parameters": {
                        "alpha": sksurv_obj.alpha,
                        "rank_ratio": sksurv_obj.rank_ratio,
                        "fit_intercept": sksurv_obj.fit_intercept,
                        "max_iter": sksurv_obj.max_iter,
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
                        "split": split.name,
                        "local_file": os.path.join(
                            split.input_dir, config.train_filename).replace(settings.INPUT_DIR, '.'),
                        "label_survival_time": config.label_time_to_event,
                        "label_event": config.label_event,
                        "event_truth_value": config.event_truth_value,
                        "n_samples": split.data.get('n_samples'),
                        "n_censored": split.data.get('n_censored'),
                    },
                    "timings": timings,
                },
            }

            # write model parameters as meta file
            meta_output_path = os.path.join(split.output_dir, config.meta_output)
            self.app.log(f"Writing metadata to {meta_output_path}", level=logging.DEBUG)

            with open(meta_output_path, "w") as fh:
                yaml.dump(metadata, fh)

        return super().run()
