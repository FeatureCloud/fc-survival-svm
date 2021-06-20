import _queue
import unittest
from typing import Tuple, Dict, List, Union, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sksurv.column import encode_categorical
from sksurv.datasets import load_veterans_lung_cancer, load_aids, load_whas500
from sksurv.svm import FastSurvivalSVM

from federated_pure_regression_survival_svm.model import Coordinator, Client, SurvivalData, SharedConfig
from federated_pure_regression_survival_svm.stepwise_newton_cg import SteppedEventBasedNewtonCgOptimizer


RANDOM_STATE = 0

class TestFederatedPureRegressionSurvivalSVM(unittest.TestCase):
    @staticmethod
    def preprocessing(X, y) -> Tuple[pd.DataFrame, np.array]:
        """
        Common data preprocessing.
        """
        X_encoded = encode_categorical(X)
        return X_encoded, y

    @staticmethod
    def global_model(X, y, alpha, fit_intercept, max_iter=1000) -> FastSurvivalSVM:
        """
        Return a fitted global model.
        """
        fast_survival_svm = FastSurvivalSVM(rank_ratio=0, alpha=alpha, fit_intercept=fit_intercept,
                                            max_iter=max_iter, random_state=RANDOM_STATE)
        fast_survival_svm.fit(X, y)

        return fast_survival_svm

    @staticmethod
    def _generate_silos(n_silos):
        coordinator = Coordinator()
        clients = [Client() for _ in range(n_silos - 1)]

        silos: List[Client] = [coordinator, *clients]
        return silos

    @staticmethod
    def fed_model(X, y, alpha, fit_intercept, max_iter=1000) -> FastSurvivalSVM:
        """
        Return a fitted federalized model.
        """
        N_CLIENTS = 3
        silos = TestFederatedPureRegressionSurvivalSVM._generate_silos(N_CLIENTS)

        kf = KFold(n_splits=N_CLIENTS, shuffle=True, random_state=RANDOM_STATE)
        data_attributes = []
        for i, splits in enumerate(kf.split(X)):
            _, split = splits

            silos[i].set_config(SharedConfig(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter))
            silos[i].set_data(SurvivalData(X.iloc[split], y[split]))
            silos[i].log_transform_times()

            data_attributes.append(silos[i].generate_data_description())

        coordinator: Coordinator = silos[0]
        coordinator.set_data_attributes(data_attributes)
        coordinator.set_initial_w_and_init_optimizer()
        coordinator.newton_optimizer: SteppedEventBasedNewtonCgOptimizer

        while not coordinator.newton_optimizer.finished:
            try:
                request = coordinator.newton_optimizer.check_pending_requests(block=False, timeout=3)
            except _queue.Empty:
                continue

            # generate local results
            local_results = [silo.handle_computation_request(request) for silo in silos]

            # aggregate results
            aggregated_result = coordinator.aggregate_local_result(local_results)

            # resolve
            coordinator.newton_optimizer.resolve(aggregated_result)

        optimize_result = coordinator.newton_optimizer.result
        return coordinator.to_sksurv(optimize_result)

    @staticmethod
    def generate_datasets():
        loaders = [load_veterans_lung_cancer, load_aids, load_whas500]
        for loader in loaders:
            yield loader.__name__, loader()

    class HyperParameters(object):
        def __init__(self, alpha, fit_intercept, max_iter):
            self.alpha = alpha
            self.fit_intercept = fit_intercept
            self.max_iter = max_iter

        def __repr__(self):
            return f"{self.__class__.__name__}" \
                   f"({', '.join('{}={}'.format(key, val) for key, val in self.__dict__.items())})"

        @classmethod
        def hyper_parameter_iteration(cls, alpha_step=0.25):
            for alpha in np.arange(alpha_step, 1, alpha_step):
                for fit_intercept in [False, True]:
                    yield cls(alpha, fit_intercept, 1000)

    def setUp(self) -> None:
        """
        Calculate models.
        """
        self.global_models: Dict[str, Dict[TestFederatedPureRegressionSurvivalSVM.HyperParameters, FastSurvivalSVM]] = {}
        self.federated_models: Dict[str, Dict[TestFederatedPureRegressionSurvivalSVM.HyperParameters, FastSurvivalSVM]] = {}
        self.test_sets: Dict[str, Tuple[pd.DataFrame, np.array]] = {}

        for dataset_name, dataset in self.generate_datasets():
            X_raw, y_raw = dataset
            X, y = TestFederatedPureRegressionSurvivalSVM.preprocessing(X_raw, y_raw)

            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE)

            self.test_sets[dataset_name] = (X_test, y_test)
            self.global_models[dataset_name] = {}
            self.federated_models[dataset_name] = {}
            for hyper_parameters in self.HyperParameters.hyper_parameter_iteration(0.25):
                self.global_models[dataset_name][hyper_parameters] = self.global_model(X_train, y_train,
                                                                         hyper_parameters.alpha,
                                                                         hyper_parameters.fit_intercept,
                                                                         hyper_parameters.max_iter)
                self.federated_models[dataset_name][hyper_parameters] = self.fed_model(X_train, y_train,
                                                                         hyper_parameters.alpha,
                                                                         hyper_parameters.fit_intercept,
                                                                         hyper_parameters.max_iter)

    def test_federalized_models_nearly_equal(self):
        for dataset_name, test_set in self.test_sets.items():
            for glo, fed in zip(self.global_models[dataset_name].values(),
                                self.federated_models[dataset_name].values()):
                print(glo.optimizer_result_)
                print(fed.optimizer_result_)

                print(glo.coef_)
                print(fed.coef_)

                print(glo.predict(test_set[0]))
                print(fed.predict(test_set[0]))
                print(test_set[1])

                print()


if __name__ == "__main__":
    unittest.main()
