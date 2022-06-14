import logging
import warnings
from typing import Optional, Tuple, List

import numpy as np
from nptyping import NDArray, Float64, Bool
from sklearn.exceptions import ConvergenceWarning
from sksurv.svm import FastSurvivalSVM

from optimization.stepwise_newton_cg import SteppedEventBasedNewtonCgOptimizer, OptimizeResult
from logic.data import SurvivalData
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


class LocalTraining:

    def __init__(self, data: SurvivalData, alpha: float = 1, fit_intercept: bool = True, max_iter: int = 1000):
        self.data: SurvivalData = data

        self.alpha: float = alpha
        # check alpha
        if not self.alpha > 0:
            raise ValueError(f"Expected alpha to be greater than zero; is {self.alpha}.")
        self.fit_intercept: bool = fit_intercept
        self.max_iter = max_iter
        self._regr_penalty = 1.0 * self.alpha

        self._regr_mask: Optional[NDArray[Bool]] = None
        self._y_compressed: Optional[NDArray[Float64]] = None

        # helpers for zeta function evaluation that only need to be calculated once
        self._zeros = np.zeros(self.data.n_samples)
        self._censored = (self.data.survival.event_indicator == False)  # noqa

        self._last_request = None

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

    def _gradient_update(self, bias: float, beta: NDArray[Float64]):
        xc = self.data.features.compress(self._regr_mask, axis=0)
        xcs = np.dot(xc, beta)
        grad_update = self._regr_penalty * (
                np.dot(xc.T, xcs) + xc.sum(axis=0) * bias - np.dot(xc.T, self._y_compressed))

        # intercept
        if self.fit_intercept:
            grad_intercept = self._regr_penalty * (xcs.sum() + xc.shape[0] * bias - self._y_compressed.sum())
            grad_update = np.hstack([grad_intercept, grad_update])

        logging.debug(f"local gradient update: beta={beta}, bias={bias}, result={grad_update}")
        return grad_update.tolist()

    def _hessp_update(self, req: SteppedEventBasedNewtonCgOptimizer.RequestHessp):
        self._last_request = req

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

        logging.debug(f"local hessp update: s={req.psupi}, result={hessp_update}")
        return hessp_update.tolist()

    def _get_values_depending_on_w(self, req: SteppedEventBasedNewtonCgOptimizer.RequestWDependent) -> Tuple[
        float, List]:
        self._last_request = req

        bias, beta = self._split_coefficients(req.xk)

        self._update_constraints(bias, beta)

        return self._calc_zeta_squared_sum(bias, beta), self._gradient_update(bias, beta)

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


class Training(LocalTraining):

    @staticmethod
    def get_n_coefficients(n_features, fit_intercept=True):
        n_coefficients = n_features
        if fit_intercept:
            n_coefficients += 1
        return n_coefficients

    def get_initial_w(self, n_features: int, mean_time_to_event: float):
        n_coefficients = self.get_n_coefficients(n_features=n_features, fit_intercept=self.fit_intercept)

        # empty array in the size of the coefficients
        # if we want to fit an intercept, set the mean tte as an initial guess
        w = np.zeros(n_coefficients, dtype=float)
        if self.fit_intercept:
            w[0] = mean_time_to_event

        return w

    def _update_w(self, new_w):
        self.w = new_w.copy()

    def _get_values_depending_on_w(self, req: SteppedEventBasedNewtonCgOptimizer.RequestWDependent):
        self._update_w(req.xk)
        return super()._get_values_depending_on_w(req)

    def _apply_hessp_update(self, aggregated_hessp_update):
        self._last_request: SteppedEventBasedNewtonCgOptimizer.RequestHessp
        s_bias, s_feat = self._split_coefficients(self._last_request.psupi)
        if self.fit_intercept:
            return np.hstack([0, s_feat]) + aggregated_hessp_update
        else:
            return s_feat + aggregated_hessp_update

    def aggregated_hessp(self, hessp_update) -> SteppedEventBasedNewtonCgOptimizer.ResolvedHessp:
        hess_p = self._apply_hessp_update(np.array(hessp_update))
        return SteppedEventBasedNewtonCgOptimizer.ResolvedHessp(hessp_val=hess_p)

    def _calc_f_val(self, total_zeta_sq_sum):
        bias, beta = self._split_coefficients(self.w)
        val = 0.5 * np.dot(beta.T, beta) + (self.alpha / 2) * total_zeta_sq_sum
        return val

    def _calc_g_val(self, aggregated_gradient_update):
        bias, beta = self._split_coefficients(self.w)
        if self.fit_intercept:
            return np.hstack([0, beta]) + aggregated_gradient_update
        else:
            return beta + aggregated_gradient_update

    def aggregate_f_val_and_g_val(self, zeta_sq_sum,
                                  gradient_update) -> SteppedEventBasedNewtonCgOptimizer.ResolvedWDependent:
        f_val = self._calc_f_val(zeta_sq_sum)
        g_val = self._calc_g_val(np.array(gradient_update))

        logging.debug(f"f_val={f_val} and g_val={g_val}")
        return SteppedEventBasedNewtonCgOptimizer.ResolvedWDependent(
            f_val=f_val,
            g_val=g_val,
        )


def create_feature_importance(classifier, feature_names: list, top_features: int = 10):
    def colors_from_values(values, palette_name):
        # normalize the values to range [0, 1]
        normalized = (values - min(values)) / (max(values) - min(values))
        # convert to indices
        indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
        # use the indices to get the colors
        palette = sns.color_palette(palette_name, len(values))
        return np.array(palette).take(indices, axis=0)

    coef = classifier.coef_.ravel()
    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coef}).sort_values(by="Coefficient",
                                                                                        ascending=False)

    coef_df = coef_df.iloc[np.r_[0:top_features, -top_features:0]]

    sns.set("talk", font_scale=1.2)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(top_features, round(top_features / 2, 1)))
    sns.barplot(data=coef_df, x="Feature", y="Coefficient", ax=ax,
                palette=colors_from_values(coef_df['Coefficient'].abs(), "flare"))
    ax.legend([], [], frameon=False)
    ax.xaxis.set_tick_params(rotation=90)

    return fig, coef_df
