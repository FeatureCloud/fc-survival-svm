import logging
import warnings
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import rsa
from nptyping import NDArray, Bool, Float64
from scipy.optimize import OptimizeResult
from sklearn.exceptions import ConvergenceWarning
from sksurv.svm import FastSurvivalSVM
from sksurv.util import check_arrays_survival

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
    local_gradient_update: NDArray[Float64]


@dataclass
class ObjectivesS(LocalResult):
    local_hessp_update: NDArray[Float64]


from smpc.helper import MaskedObjectivesW, SMPCMasked, MaskedObjectivesS  # noqa: Import late to avoid circular import problem


class Client(object):
    def __init__(self):
        self.data: Optional[SurvivalData] = None

        self.alpha: Optional[float] = None
        self.fit_intercept: Optional[bool] = None
        self.max_iter = None
        self._regr_penalty: Optional[float] = None

        self.last_request: Optional[SteppedEventBasedNewtonCgOptimizer.Request] = None

        self._regr_mask: Optional[NDArray[Bool]] = None
        self._y_compressed: Optional[NDArray[Float64]] = None

        # helpers for zeta function evaluation that only need to be calculated once
        self._zeros = None
        self._censored = None

    def set_data(self, data: SurvivalData):
        self.data = data
        self._check_times()

        # calculate these once for zeta function evaluation
        number_of_samples = self.data.features.shape[0]
        self._zeros = np.zeros(number_of_samples)
        self._censored = (self.data.survival.event_indicator == False)  # noqa

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
        return float(np.sum(self.data.survival.time_to_event, axis=None))

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

    def _calc_zeta_squared_sum(self, bias: float, beta: NDArray[Float64]) -> float:
        dot_product = np.sum(np.multiply(beta.T, self.data.features),
                             axis=1)  # equal to dot product of beta.T with each feature row
        weighted = self.data.survival.time_to_event - dot_product - bias
        np.maximum(weighted, self._zeros, out=weighted,
                   where=self._censored)  # replaces values of censored entries inplace with 0 if weighted is below 0

        zeta_sq_sum = float(np.sum(np.square(weighted), axis=None))
        logging.debug(f"local zeta_sq_sum: beta={beta}, bias={bias}, result={zeta_sq_sum}")
        return zeta_sq_sum

    def _update_constraints(self, bias: float, beta: NDArray[Float64]):
        xw = np.dot(self.data.features, beta)
        pred_time = self.data.survival.time_to_event - xw - bias
        self._regr_mask = (pred_time > 0) | self.data.survival.event_indicator
        self._y_compressed = self.data.survival.time_to_event.compress(self._regr_mask, axis=0)

    def _gradient_update(self, bias: float, beta: NDArray[Float64]) -> NDArray[Float64]:
        xc = self.data.features.compress(self._regr_mask, axis=0)
        xcs = np.dot(xc, beta)
        grad_update = self._regr_penalty * (
                np.dot(xc.T, xcs) + xc.sum(axis=0) * bias - np.dot(xc.T, self._y_compressed))

        # intercept
        if self.fit_intercept:
            grad_intercept = self._regr_penalty * (xcs.sum() + xc.shape[0] * bias - self._y_compressed.sum())
            grad_update = np.hstack([grad_intercept, grad_update])

        logging.debug(f"local gradient: beta={beta}, bias={bias}, result={grad_update}")
        return grad_update

    def _hessp_update(self, req: SteppedEventBasedNewtonCgOptimizer.RequestHessp) -> ObjectivesS:
        self.last_request = req

        s_bias, s_feat = self._split_coefficients(req.psupi)

        xc = self.data.features.compress(self._regr_mask, axis=0)
        hessp_update = self._regr_penalty * np.dot(xc.T, np.dot(xc, s_feat))

        # intercept
        if self.fit_intercept:
            xsum = xc.sum(axis=0)
            hessp_update += self._regr_penalty * xsum * s_bias
            hessp_intercept = (self._regr_penalty * xc.shape[0] * s_bias
                               + self._regr_penalty * np.dot(xsum, s_feat))
            hessp_update = np.hstack([hessp_intercept, hessp_update])

        logging.debug(f"local hessian: s={req.psupi}, result={hessp_update}")
        return ObjectivesS(
            local_hessp_update=hessp_update
        )

    def _get_values_depending_on_w(self, req: SteppedEventBasedNewtonCgOptimizer.RequestWDependent) -> ObjectivesW:
        self.last_request = req

        bias, beta = self._split_coefficients(req.xk)

        self._update_constraints(bias, beta)

        return ObjectivesW(
            local_sum_of_zeta_squared=self._calc_zeta_squared_sum(bias, beta),
            local_gradient_update=self._gradient_update(bias, beta),
        )

    def handle_computation_request(self, request: SteppedEventBasedNewtonCgOptimizer.Request) -> LocalResult:
        if isinstance(request, SteppedEventBasedNewtonCgOptimizer.RequestWDependent):
            request: SteppedEventBasedNewtonCgOptimizer.RequestWDependent
            return self._get_values_depending_on_w(request)
        elif isinstance(request, SteppedEventBasedNewtonCgOptimizer.RequestHessp):
            request: SteppedEventBasedNewtonCgOptimizer.RequestHessp
            return self._hessp_update(request)

    def handle_computation_request_smpc(self, request: SteppedEventBasedNewtonCgOptimizer.Request,
                                        pub_keys_of_other_parties: Dict[int, rsa.PublicKey]) -> SMPCMasked:
        if isinstance(request, SteppedEventBasedNewtonCgOptimizer.RequestWDependent):
            request: SteppedEventBasedNewtonCgOptimizer.RequestWDependent
            response: ObjectivesW = self._get_values_depending_on_w(request)
            return MaskedObjectivesW().mask(response, pub_keys_of_other_parties)
        elif isinstance(request, SteppedEventBasedNewtonCgOptimizer.RequestHessp):
            request: SteppedEventBasedNewtonCgOptimizer.RequestHessp
            response: ObjectivesS = self._hessp_update(request)
            return MaskedObjectivesS().mask(response, pub_keys_of_other_parties)

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
        self._last_w: Optional[NDArray] = None
        self.total_zeta_sq_sum: Optional[float] = None

        self.newton_optimizer = None

    def set_data_description(self, data_descriptions: DataDescription):
        self.total_n_features = data_descriptions.n_features
        logging.debug(f"Clients have {self.total_n_features} features")

        self.total_n_samples = data_descriptions.n_samples
        logging.debug(f"Clients have {self.total_n_samples} samples in total")

        self.total_time_sum = data_descriptions.sum_of_times
        logging.debug(f"Clients have a summed up time of {self.total_time_sum}")

    def aggregate_and_set_data_descriptions(self, data_descriptions: List[DataDescription]):
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
        w = np.zeros(self.n_coefficients, dtype=float)
        self._last_w = w.copy()

        n = w.shape[0]
        if self.fit_intercept:
            time_mean = self.time_mean
            w[0] = time_mean
            n -= 1

        logging.debug(f"Initial w: {w}")
        self._update_w(w)
        self.newton_optimizer = SteppedEventBasedNewtonCgOptimizer(w)  # noqa
        return w

    def _update_w(self, new_w):
        self.w = new_w.copy()

    def _get_values_depending_on_w(self, req: SteppedEventBasedNewtonCgOptimizer.RequestWDependent) -> ObjectivesW:
        self._update_w(req.xk)
        return super()._get_values_depending_on_w(req)

    def _apply_hessp_update(self, aggregated_hessp_update):
        self.last_request: SteppedEventBasedNewtonCgOptimizer.RequestHessp
        s_bias, s_feat = self._split_coefficients(self.last_request.psupi)
        return np.hstack([0, s_feat]) + aggregated_hessp_update

    def aggregated_hessp(self, results: List[ObjectivesS]) -> SteppedEventBasedNewtonCgOptimizer.ResolvedHessp:
        # unwrap
        aggregated_hessp_update = results[0].local_hessp_update
        for i in range(1, len(results)):
            aggregated_hessp_update += results[i].local_hessp_update
        logging.debug(f"Aggregated hessp update: {aggregated_hessp_update}")

        hess_p = self._apply_hessp_update(aggregated_hessp_update)
        return SteppedEventBasedNewtonCgOptimizer.ResolvedHessp(hessp_val=hess_p)

    def aggregated_hessp_smpc(self, result: ObjectivesS) -> SteppedEventBasedNewtonCgOptimizer.ResolvedHessp:
        # unwrap
        aggregated_hessp_update = result.local_hessp_update
        logging.debug(f"Aggregated hessp update: {aggregated_hessp_update}")

        hess_p = self._apply_hessp_update(aggregated_hessp_update)
        return SteppedEventBasedNewtonCgOptimizer.ResolvedHessp(hessp_val=hess_p)

    def _calc_f_val(self, total_zeta_sq_sum):
        bias, beta = self._split_coefficients(self.w)
        val = 0.5 * np.dot(beta.T, beta) + (self.alpha / 2) * total_zeta_sq_sum
        return val

    def _calc_g_val(self, aggregated_gradient_update):
        bias, beta = self._split_coefficients(self.w)
        return np.hstack([0, beta]) + aggregated_gradient_update

    def aggregate_f_val_and_g_val(self,
                                  results: List[ObjectivesW]) -> SteppedEventBasedNewtonCgOptimizer.ResolvedWDependent:
        # unwrap
        aggregated_zeta_sq_sums = results[0].local_sum_of_zeta_squared
        aggregated_gradient_update = results[0].local_gradient_update
        for i in range(1, len(results)):
            aggregated_zeta_sq_sums += results[i].local_sum_of_zeta_squared
            aggregated_gradient_update += results[i].local_gradient_update

        f_val = self._calc_f_val(aggregated_zeta_sq_sums)
        g_val = self._calc_g_val(aggregated_gradient_update)

        logging.debug(f"f_val={f_val} and g_val={g_val}")
        return SteppedEventBasedNewtonCgOptimizer.ResolvedWDependent(
            f_val=f_val,
            g_val=g_val,
        )

    def aggregate_f_val_and_g_val_smpc(self,
                                       result: ObjectivesW) -> SteppedEventBasedNewtonCgOptimizer.ResolvedWDependent:
        # unwrap
        aggregated_zeta_sq_sums = result.local_sum_of_zeta_squared
        aggregated_gradients = result.local_gradient_update

        f_val = self._calc_f_val(aggregated_zeta_sq_sums)
        g_val = aggregated_gradients
        logging.debug(f"f_val={f_val} and g_val={g_val}")
        return SteppedEventBasedNewtonCgOptimizer.ResolvedWDependent(
            f_val=f_val,
            g_val=g_val,
        )

    def aggregate_local_result(self, local_results: List[LocalResult]) -> SteppedEventBasedNewtonCgOptimizer.Resolved:
        if isinstance(local_results[0], ObjectivesW):
            local_results: List[ObjectivesW]
            return self.aggregate_f_val_and_g_val(local_results)
        elif isinstance(local_results[0], ObjectivesS):
            local_results: List[ObjectivesS]
            return self.aggregated_hessp(local_results)

    def aggregate_local_result_smpc(self, local_result: LocalResult) -> SteppedEventBasedNewtonCgOptimizer.Resolved:
        if isinstance(local_result, ObjectivesW):
            local_result: ObjectivesW
            return self.aggregate_f_val_and_g_val_smpc(local_result)
        elif isinstance(local_result, ObjectivesS):
            local_result: ObjectivesS
            return self.aggregated_hessp_smpc(local_result)
