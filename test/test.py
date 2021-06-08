import unittest
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sksurv.column import encode_categorical
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.svm import FastSurvivalSVM


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
        X, y = TestFederatedPureRegressionSurvivalSVM.preprocessing(X, y)
        fast_survival_svm = FastSurvivalSVM(rank_ratio=0, alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter)
        fast_survival_svm.fit(X, y)

        return fast_survival_svm


    @staticmethod
    def generate_datasets():
        loaders = [load_veterans_lung_cancer]
        for loader in loaders:
            yield loader()

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
        Calculate global model.
        """
        self.global_models: Dict[TestFederatedPureRegressionSurvivalSVM.HyperParameters, FastSurvivalSVM] = {}
        for dataset in self.generate_datasets():
            X, y = dataset
            for hyper_parameters in self.HyperParameters.hyper_parameter_iteration(0.25):
                self.global_models[hyper_parameters] = self.global_model(X, y,
                                                                         hyper_parameters.alpha,
                                                                         hyper_parameters.fit_intercept,
                                                                         hyper_parameters.max_iter)
        print(self.global_models)

    def test_federalized_models_nearly_equal(self):
        pass


if __name__ == "__main__":
    unittest.main()
