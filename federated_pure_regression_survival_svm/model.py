import logging
import warnings
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
from nptyping import NDArray, Bool, Float64
from scipy.optimize import OptimizeResult
from sklearn.exceptions import ConvergenceWarning
from sksurv.util import check_arrays_survival
from sksurv.svm import FastSurvivalSVM

from federated_pure_regression_survival_svm.stepwise_newton_cg import SteppedEventBasedNewtonCgOptimizer


@dataclass
class SurvivalTuple:
    event_indicator: NDArray[Bool]
    time_to_event: NDArray[Float64]


@dataclass
class SurvivalData:
    features: NDArray
    survival: SurvivalTuple

    def __init__(self, X: pd.DataFrame, y: NDArray):
        x_array, event, time = check_arrays_survival(X, y)

        self.features = x_array
        self.survival = SurvivalTuple(
            event_indicator=event,
            time_to_event=time
        )


@dataclass
class SharedConfig:
    alpha: float
    fit_intercept: bool
    max_iter: int


@dataclass
class DataDescription:
    n_samples: int
    n_features: int
    sum_of_times: float


class Signal:
    pass


@dataclass
class OptFinished(Signal):
    opt_results: Dict[str, OptimizeResult]


class LocalResult:
    pass


@dataclass
class ObjectivesW(LocalResult):
    local_sum_of_zeta_squared: float
    local_gradient: NDArray[Float64]


@dataclass
class ObjectivesS(LocalResult):
    local_hessian: NDArray[Float64]


class Client(object):
    def __init__(self):
        self.data: Optional[SurvivalData] = None

        self.alpha: Optional[float] = None
        self.fit_intercept: Optional[bool] = None
        self.max_iter = None
        self._regr_penalty: Optional[float] = None

        self._zeta_sq_sum: Optional[float] = None
        self._regr_mask: Optional[NDArray[Bool]] = None
        self._y_compressed: Optional[NDArray[Float64]] = None

    def set_data(self, data: SurvivalData):
        self.data = data
        self._check_times()

    def _put_data(self, X, y):
        self.set_data(SurvivalData(X, y))

    def _check_times(self):
        logging.debug("Check observed time does not contain values smaller or equal to zero")
        time = self.data.survival.time_to_event
        if (time <= 0).any():
            raise ValueError("observed time contains values smaller or equal to zero")
        return True

    @property
    def data_loaded(self):
        return self.data is not None

    def set_config(self, config: SharedConfig):
        # set values for alpha and fit_intercept
        self.alpha = config.alpha
        self.fit_intercept = config.fit_intercept
        self.max_iter = config.max_iter

        # check alpha
        if not self.alpha > 0:
            return ValueError(f"Expected alpha to be greater than zero; is {self.alpha}.")

        # set regression penalty
        self._regr_penalty = 1.0 * self.alpha
        logging.debug(f"Regression penalty was set to {self._regr_penalty}")

    def log_transform_times(self):
        logging.info("Log transform times for regression objective")
        self.data.survival.time_to_event = np.log(self.data.survival.time_to_event)
        assert np.isfinite(self.data.survival.time_to_event).all()

    @property
    def shape(self):
        return self.data.features.shape

    @property
    def n_samples(self):
        n = self.shape[0]
        return n

    @property
    def n_features(self):
        n = self.shape[1]
        return n

    @property
    def time_sum(self) -> float:
        return np.sum(self.data.survival.time_to_event)

    def generate_data_description(self):
        return DataDescription(
            n_samples=self.n_samples,
            n_features=self.n_features,
            sum_of_times=self.time_sum,
        )

    def _split_coefficients(self, w: NDArray[Float64]) -> Tuple[float, NDArray[Float64]]:
        """Split into intercept/bias and feature-specific coefficients"""
        if self.fit_intercept:
            bias = w[0]
            wf = w[1:]
        else:
            bias = 0.0
            wf = w
        return bias, wf

    @staticmethod
    def _zeta_function(x: NDArray, time: Float64, event: Bool, bias: float, beta: NDArray[Float64]):
        weighted = time - np.dot(beta.T, x) - bias

        # where data is censored use the maximum between the weighted results and 0
        if not event:
            return max(0, weighted)

        return weighted

    def _calc_zeta_squared_sum(self, bias: float, beta: NDArray[Float64]):
        number_of_samples = self.data.features.shape[0]

        inner_result = np.zeros(number_of_samples)

        for i in range(number_of_samples):
            inner_result[i] = self._zeta_function(self.data.features[i], self.data.survival.time_to_event[i],
                                                  self.data.survival.event_indicator[i], bias, beta) ** 2

        zeta_sq_sum = np.sum(inner_result)
        logging.debug(f"local zeta_sq_sum: beta={beta}, bias={bias}, result={zeta_sq_sum}")
        return zeta_sq_sum

    def _update_constraints(self, bias: float, beta: NDArray[Float64]):
        xw = np.dot(self.data.features, beta)
        pred_time = self.data.survival.time_to_event - xw - bias
        self._regr_mask = (pred_time > 0) | self.data.survival.event_indicator
        self._y_compressed = self.data.survival.time_to_event.compress(self._regr_mask, axis=0)

    def _gradient_func(self, bias: float, beta: NDArray[Float64]) -> NDArray[Float64]:

        grad = beta.copy()

        xc = self.data.features.compress(self._regr_mask, axis=0)
        xcs = np.dot(xc, beta)
        grad += self._regr_penalty * (np.dot(xc.T, xcs) + xc.sum(axis=0) * bias - np.dot(xc.T, self._y_compressed))

        # intercept
        if self.fit_intercept:
            grad_intercept = self._regr_penalty * (xcs.sum() + xc.shape[0] * bias - self._y_compressed.sum())
            grad = np.r_[grad_intercept, grad]

        logging.debug(f"local gradient: beta={beta}, bias={bias}, result={grad}")
        return grad

    def _hessian_func(self, req: SteppedEventBasedNewtonCgOptimizer.RequestHessp) -> ObjectivesS:
        s_bias, s_feat = self._split_coefficients(req.psupi)

        hessp = s_feat.copy()
        xc = self.data.features.compress(self._regr_mask, axis=0)
        hessp += self._regr_penalty * np.dot(xc.T, np.dot(xc, s_feat))

        # intercept
        if self.fit_intercept:
            xsum = xc.sum(axis=0)
            hessp += self._regr_penalty * xsum * s_bias
            hessp_intercept = (self._regr_penalty * xc.shape[0] * s_bias
                               + self._regr_penalty * np.dot(xsum, s_feat))
            hessp = np.r_[hessp_intercept, hessp]

        logging.debug(f"local hessian: s={req.psupi}, result={hessp}")
        return ObjectivesS(
            local_hessian=hessp
        )

    def _get_values_depending_on_w(self, req: SteppedEventBasedNewtonCgOptimizer.RequestWDependent) -> ObjectivesW:
        bias, beta = self._split_coefficients(req.xk)

        self._update_constraints(bias, beta)

        return ObjectivesW(
            local_sum_of_zeta_squared=self._calc_zeta_squared_sum(bias, beta),
            local_gradient=self._gradient_func(bias, beta),
        )

    def handle_computation_request(self, request: SteppedEventBasedNewtonCgOptimizer.Request) -> LocalResult:
        if isinstance(request, SteppedEventBasedNewtonCgOptimizer.RequestWDependent):
            request: SteppedEventBasedNewtonCgOptimizer.RequestWDependent
            return self._get_values_depending_on_w(request)
        elif isinstance(request, SteppedEventBasedNewtonCgOptimizer.RequestHessp):
            request: SteppedEventBasedNewtonCgOptimizer.RequestHessp
            return self._hessian_func(request)

    def to_sksurv(self, optimize_result: OptimizeResult) -> FastSurvivalSVM:
        """
        Export model to a FastSurvivalSVM in the fitted state.
        :param optimize_result:
        :return:
        """
        fss = FastSurvivalSVM(
            alpha=self.alpha,
            rank_ratio=0,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
        )

        # set attributes in order to get a FastSurvivalSVM in the fitted state
        coef = optimize_result.x
        if fss.fit_intercept:
            fss.coef_ = coef[1:]
            fss.intercept_ = coef[0]
        else:
            fss.coef_ = coef

        if not optimize_result.success:
            warnings.warn(('Optimization did not converge: ' + optimize_result.message),
                          category=ConvergenceWarning,
                          stacklevel=2)
        fss.optimizer_result_ = optimize_result

        return fss


class Coordinator(Client):
    def __init__(self):
        super().__init__()
        self.total_n_features: Optional[int] = None  # number of features (equal at each client)
        self.total_n_samples: Optional[int] = None  # summed up number of samples over all clients
        self.total_time_sum: Optional[float] = None  # summed up time over all clients

        self.w: Optional[NDArray] = None  # weights vector
        self.total_zeta_sq_sum: Optional[float] = None

        self.newton_optimizer = None

    def set_data_attributes(self, data_descriptions: List[DataDescription]):
        # unwrap
        aggregated_n_features = []
        aggregated_n_samples = np.zeros(len(data_descriptions))
        aggregated_sum_of_times = np.zeros(len(data_descriptions))
        for i, local_data in enumerate(data_descriptions):
            aggregated_n_features.append(local_data.n_features)
            aggregated_n_samples[i] = local_data.n_samples
            aggregated_sum_of_times[i] = local_data.sum_of_times

        # check all clients have reported the same number of features
        if len(set(aggregated_n_features)) > 1:
            raise ValueError("Clients reported a differing number of features")
        self.total_n_features = aggregated_n_features[0]
        logging.debug(f"Clients agree on having {self.total_n_features} features")

        # get total number of samples
        self.total_n_samples = np.sum(aggregated_n_samples)
        logging.debug(f"Clients have {self.total_n_samples} samples in total")

        # get summed up times
        self.total_time_sum = np.sum(aggregated_sum_of_times)
        logging.debug(f"Clients have a summed up time of {self.total_time_sum}")

    @property
    def n_coefficients(self):
        n = self.total_n_features
        if self.fit_intercept:
            n += 1
        return n

    @property
    def time_mean(self):
        return self.total_time_sum / self.total_n_samples

    def set_initial_w_and_init_optimizer(self):
        w = np.zeros(self.n_coefficients)
        self._last_w = w.copy()

        n = w.shape[0]
        if self.fit_intercept:
            time_mean = self.time_mean
            w[0] = time_mean
            n -= 1

        logging.debug(f"Initial w: {w}")
        self.w = w
        self.newton_optimizer = SteppedEventBasedNewtonCgOptimizer(w)
        return w

    def aggregated_hessp(self, results: List[ObjectivesS]) -> SteppedEventBasedNewtonCgOptimizer.ResolvedHessp:
        # unwrap
        aggregated_hessp = results[0].local_hessian
        for i in range(1, len(results)):
            aggregated_hessp += results[i].local_hessian
        logging.debug(f"Aggregated hessian: {aggregated_hessp}")

        hess_p = aggregated_hessp
        return SteppedEventBasedNewtonCgOptimizer.ResolvedHessp(aggregated_hessp=hess_p)

    def _calc_fval(self, total_zeta_sq_sum):
        bias, beta = self._split_coefficients(self.w)
        val = 0.5 * np.dot(beta.T, beta) + (self.alpha / 2) * total_zeta_sq_sum
        return val

    def aggregate_fval_and_gval(self,
                                results: List[ObjectivesW]) -> SteppedEventBasedNewtonCgOptimizer.ResolvedWDependent:
        # unwrap
        aggregated_zeta_sq_sums = results[0].local_sum_of_zeta_squared
        aggregated_gradients = results[0].local_gradient
        for i in range(1, len(results)):
            aggregated_zeta_sq_sums += results[i].local_sum_of_zeta_squared
            aggregated_gradients += results[i].local_gradient

        fval = self._calc_fval(aggregated_zeta_sq_sums)
        gval = aggregated_gradients
        logging.debug(f"fval={fval} and gval={gval}")
        return SteppedEventBasedNewtonCgOptimizer.ResolvedWDependent(
            aggregated_objective_function_value=fval,
            aggregated_gradient=gval
        )

    def aggregate_local_result(self, local_results: List[LocalResult]) -> SteppedEventBasedNewtonCgOptimizer.Resolved:
        if isinstance(local_results[0], ObjectivesW):
            local_results: List[ObjectivesW]
            return self.aggregate_fval_and_gval(local_results)
        elif isinstance(local_results[0], ObjectivesS):
            local_results: List[ObjectivesS]
            return self.aggregated_hessp(local_results)
