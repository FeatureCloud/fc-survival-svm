import contextlib
from unittest import TestCase

import numpy
import numpy as np
from hypothesis import given, settings
import hypothesis.extra.numpy as hxn

import scipy
from hypothesis.strategies import floats
from scipy.optimize import rosen, rosen_der, rosen_hess_prod, OptimizeResult

from federated_pure_regression_survival_svm.stepwise_newton_cg import SteppedEventBasedNewtonCgOptimizer

objective_function = rosen
jacobian_function = rosen_der
hessp_function = rosen_hess_prod


def calc_objective_function_and_gradient_at(req: SteppedEventBasedNewtonCgOptimizer.RequestWDependent):
    return SteppedEventBasedNewtonCgOptimizer.ResolvedWDependent(f_val=objective_function(req.xk),
                                                                 g_val=jacobian_function(req.xk))


def calc_hessp_at(req: SteppedEventBasedNewtonCgOptimizer.RequestHessp):
    return SteppedEventBasedNewtonCgOptimizer.ResolvedHessp(hessp_val=hessp_function(req.xk, req.psupi))


def handle_computation_request(
        request: SteppedEventBasedNewtonCgOptimizer.Request
) -> SteppedEventBasedNewtonCgOptimizer.Resolved:
    if isinstance(request, SteppedEventBasedNewtonCgOptimizer.RequestWDependent):
        request: SteppedEventBasedNewtonCgOptimizer.RequestWDependent
        return calc_objective_function_and_gradient_at(request)
    elif isinstance(request, SteppedEventBasedNewtonCgOptimizer.RequestHessp):
        request: SteppedEventBasedNewtonCgOptimizer.RequestHessp
        return calc_hessp_at(request)


class TestSteppedEventBasedNewtonCgOptimizer(TestCase):
    def test_finishes(self):
        eventMinimizer = SteppedEventBasedNewtonCgOptimizer([1, 1])
        eventMinimizer.solve(handle_computation_request)
        self.assertTrue(eventMinimizer.finished)

    def test_max_iterations_exceeded(self):
        eventMinimizer = SteppedEventBasedNewtonCgOptimizer([2, -1], options={'maxiter': 20})
        eventMinimizer.solve(handle_computation_request)
        self.assertTrue(eventMinimizer.finished)
        self.assertFalse(eventMinimizer.result.success)

    @settings(deadline=None)
    @given(x0=hxn.arrays(np.float, 2, elements=floats(allow_infinity=False, allow_nan=False)))
    def test_hypothesis_equal_results_to_standard_method(self, x0):
        with contextlib.suppress(RuntimeWarning):
            eventMinimizer = SteppedEventBasedNewtonCgOptimizer(x0)
            eventMinimizer.solve(handle_computation_request)

            normalMinimizer: OptimizeResult = scipy.optimize.minimize(
                fun=objective_function,
                x0=x0,
                method='newton-cg',
                jac=jacobian_function,
                hessp=hessp_function
            )

            np.testing.assert_almost_equal(eventMinimizer.result.x, normalMinimizer.x)

    def test_has_pending(self):
        eventMinimizer = SteppedEventBasedNewtonCgOptimizer([2, -1])

        self.assertTrue(eventMinimizer.has_pending())
        request = eventMinimizer.check_pending_requests()
        self.assertFalse(eventMinimizer.has_pending())

    def test_finished_before_end(self):
        eventMinimizer = SteppedEventBasedNewtonCgOptimizer([2, -1])
        _ = eventMinimizer.check_pending_requests()
        self.assertFalse(eventMinimizer.finished)

    def test_wrong_type_resolve(self):
        eventMinimizer = SteppedEventBasedNewtonCgOptimizer([2, -1])

        with self.assertRaises(SteppedEventBasedNewtonCgOptimizer.UnexpectedResolveType):
            eventMinimizer.resolve(SteppedEventBasedNewtonCgOptimizer.ResolvedHessp(hessp_val=[0, 0]))
