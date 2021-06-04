# The following is a derived work of the
# optimize.py module by Travis E. Oliphant
# https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py
# which is part of the scipy package
import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from nptyping import NDArray, Float64
from scipy.optimize import minpack2
from scipy.optimize.optimize import _LineSearchError, OptimizeResult, rosen_hess_prod, rosen_der, rosen


class Stats(object):
    def __init__(self):
        self.iteration_counter = 0

        self.nfev = 0
        self.njev = 0
        self.nhev = 0


class Optimizer(object):
    pass


class StepwiseWolfeLineSearch(object):
    @dataclass
    class Result:
        stp: float
        converged: bool
        iter: int

    def __init__(self, maxiter=100):
        self.isave = np.zeros((2,), np.intc)
        self.dsave = np.zeros((13,), float)
        self.task = b'START'

        self.alpha1 = 1.0
        self.c1 = 1e-4
        self.c2 = 0.9
        self.amin = 1e-8
        self.amax = 50
        self.xtol = 1e-14

        self.iter = 0
        self.maxiter = maxiter

        self.line_search_converged = False

    def get_dephi(self, gval, pk):
        dephi0 = np.dot(gval, pk)
        return dephi0

    def next_step_needed(self):
        if self.iter >= self.maxiter:
            raise _LineSearchError()
        return not self.line_search_converged

    def line_search_step(self, phi1, derphi1):
        self.iter += 1
        stp, phi1, derphi1, self.task = minpack2.dcsrch(self.alpha1, phi1, derphi1,
                                                        self.c1, self.c2, self.xtol, self.task,
                                                        self.amin, self.amax, self.isave, self.dsave)

        if self.task[:2] == b'FG':
            self.alpha1 = stp
            return StepwiseWolfeLineSearch.Result(stp, converged=False, iter=self.iter)
        elif self.task[:5] == b'ERROR' or self.task[:4] == b'WARN':
            raise _LineSearchError(self.task.decode('utf-8').strip())  # failed
        else:
            self.line_search_converged = True
            return StepwiseWolfeLineSearch.Result(stp, converged=True, iter=self.iter)


class StepwiseTruncatedConjugateGradient(Optimizer):
    @dataclass
    class Result:
        psupi: float
        xsupi: float
        ri: float
        dri0: float
        iter: int

    float64eps = np.finfo(np.float64).eps

    def __init__(self, psupi, n_coefficients, maxiter):
        self.iter = 0
        self.finished = False

        self.psupi = psupi

        # calculate termination condition
        maggrad = np.add.reduce(np.abs(self.psupi))
        eta = np.min([0.5, np.sqrt(maggrad)])
        self.termcond = eta * maggrad

        self.xsupi = np.zeros(n_coefficients, dtype=np.float64)

        self.maxiter = maxiter
        self.b = psupi
        self.ri = -psupi
        self.dri0 = np.dot(self.ri, self.ri)

    def cg_next_round_needed(self):
        if self.finished:
            return False

        if self.iter >= self.maxiter:
            raise Exception("Warning: CG iterations didn't converge. The Hessian is not positive definite.")

        if np.add.reduce(np.abs(self.ri)) <= self.termcond:
            return False

        return True

    def _break_cg(self):
        self.finished = True

    def _do_cg_next(self, hval, psupi, xsupi, dri0, ri, i):
        # check curvature
        Ap = np.asarray(hval).squeeze()  # get rid of matrices...
        _curv = np.dot(psupi, Ap)
        if 0 <= _curv <= 3 * self.float64eps:
            self._break_cg()
        elif _curv < 0:
            if (i > 0):
                self._break_cg()
            else:
                # fall back to steepest descent direction
                xsupi = dri0 / (-_curv) * self.b
                self._break_cg()
        _alphai = dri0 / _curv
        xsupi = xsupi + _alphai * psupi
        ri = ri + _alphai * Ap
        _dri1 = np.dot(ri, ri)
        _betai = _dri1 / dri0
        psupi = -ri + _betai * psupi
        i += 1
        dri0 = _dri1  # update np.dot(ri,ri) for next time.

        return StepwiseTruncatedConjugateGradient.Result(psupi=psupi, xsupi=xsupi, dri0=dri0, ri=ri, iter=i)

    def cg_next(self, hval):
        result: StepwiseTruncatedConjugateGradient.Result = self._do_cg_next(hval, self.psupi, self.xsupi, self.dri0,
                                                                             self.ri,
                                                                             self.iter)
        self.psupi = result.psupi
        self.xsupi = result.xsupi
        self.dri0 = result.dri0
        self.ri = result.ri
        self.iter = result.iter

        return result


class StepwiseNewtonCgOptimizer(Optimizer):
    def __init__(self, n_coefficients: int, avextol=1e-5, maxiter=None):
        self.n_coefficients = n_coefficients

        self.xtol = self.n_coefficients * avextol
        self.update = np.array([2 * self.xtol])

        self.maxiter = self.n_coefficients * 200 if maxiter is None else maxiter
        self.cg_maxiter = 20 * self.n_coefficients

        self.iter = 0

    def newton_next_round_needed(self):
        if np.add.reduce(np.abs(self.update)) <= self.xtol:
            if np.isnan(self.update).any():
                raise Exception("NaN result encountered.")
            return False
        if self.iter >= self.maxiter:
            raise Exception("Maximum number of iterations has been exceeded.")
        return True

    def newton_prime_inner_optimizer(self, gval: NDArray[float]):
        psupi = -gval
        self.truncated_solver = StepwiseTruncatedConjugateGradient(psupi=psupi,
                                                                   n_coefficients=self.n_coefficients,
                                                                   maxiter=self.cg_maxiter)
        return psupi, self.truncated_solver

    def newton_update_xk(self, alphak, xsupi, xk):
        self.iter += 1

        self.update = alphak * xsupi
        xk = xk + self.update
        return xk


@dataclass
class UpdateW:
    aggregated_objective_function_value: float
    aggregated_gradient: NDArray[Float64]


@dataclass
class UpdateS:
    aggregated_hessian: NDArray[Float64]


class NewtonCgStateMachine(object):
    class Output:
        pass

    @dataclass
    class RequestWDependent(Output):
        xk: NDArray

    @dataclass
    class RequestHessian(Output):
        xk: NDArray
        psupi: NDArray

    def __init__(self, x0):
        self._state = 'needs_initialization'

        self.xk = np.array(x0)
        self.fval = None
        self.gval = None
        self.hval = None

        self.stat = Stats()
        self.newton_optimizer = StepwiseNewtonCgOptimizer(self.xk.shape[0])

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        logging.debug(state)
        self._state = state

    @property
    def finished(self):
        return self.state == 'finished' or self.state == 'line_search_failed'

    def next(self, data):
        while True:
            if self.state == 'needs_initialization':
                self.state = 'initialization'
                return self.RequestWDependent(self.xk)

            if self.state == 'initialization':
                if not isinstance(data, UpdateW):
                    raise Exception
                else:
                    data: UpdateW
                    self.fval = data.aggregated_objective_function_value
                    self.gval = data.aggregated_gradient
                    self.stat.nfev += 1
                    self.stat.njev += 1
                    data = None

                    self.state = 'newton_optimization_next_round'

            if self.state == 'newton_optimization_next_round':
                if not self.newton_optimizer.newton_next_round_needed():
                    self.state = 'finished'
                else:
                    self.psupi, self.trunc_cg_optimizer = self.newton_optimizer.newton_prime_inner_optimizer(self.gval)
                    self.state = 'inner_optimization_start'

                    if not self.trunc_cg_optimizer.cg_next_round_needed():
                        self.state = 'inner_cg_finished'

            if self.state == 'inner_optimization_start':
                self.state = 'inner_optimization_proceed'
                return self.RequestHessian(self.xk, self.psupi)

            if self.state == 'inner_optimization_proceed':
                if not isinstance(data, UpdateS):
                    raise Exception
                else:
                    data: UpdateS
                    self.trunc_nc_result = self.trunc_cg_optimizer.cg_next(data.aggregated_hessian)
                    self.stat.nhev += 1
                    data = None
                    self.psupi = self.trunc_nc_result.psupi

                    if not self.trunc_cg_optimizer.cg_next_round_needed():
                        self.state = 'inner_cg_finished'
                    else:
                        return self.RequestHessian(self.xk, self.psupi)

            if self.state == 'inner_cg_finished':
                self.state = 'line_search_start'

            if self.state == 'line_search_start':
                self.line_searcher = StepwiseWolfeLineSearch()

                if not self.line_searcher.next_step_needed():
                    self.state = 'line_search_finished'
                else:
                    phi = self.fval
                    derphi = self.line_searcher.get_dephi(self.gval, self.trunc_nc_result.xsupi)
                    try:
                        line_search_result = self.line_searcher.line_search_step(phi, derphi)
                        if line_search_result.converged:
                            self.state = 'line_search_finished'
                        else:
                            evaluate_for = self.xk + line_search_result.stp * self.trunc_nc_result.xsupi
                            self.state = 'line_search_proceed'
                            return self.RequestWDependent(evaluate_for)
                    except _LineSearchError:
                        self.state = 'line_search_failed'

            if self.state == 'line_search_proceed':
                if not isinstance(data, UpdateW):
                    raise Exception
                else:
                    data: UpdateW
                    self.fval = data.aggregated_objective_function_value
                    self.gval = data.aggregated_gradient
                    self.stat.nfev += 1
                    self.stat.njev += 1
                    data = None

                    if not self.line_searcher.next_step_needed():
                        self.state = 'line_search_finished'
                    else:
                        phi = self.fval
                        derphi = self.line_searcher.get_dephi(self.gval, self.trunc_nc_result.xsupi)
                        self.line_search_result = self.line_searcher.line_search_step(phi, derphi)

                        if self.line_search_result.converged:
                            self.state = 'line_search_finished'
                        else:
                            evaluate_for = self.xk + self.line_search_result.stp * self.trunc_nc_result.xsupi
                            self.state = 'line_search_proceed'
                            return self.RequestWDependent(evaluate_for)

            if self.state == 'line_search_finished':
                self.state = 'update_xk'

            if self.state == 'update_xk':
                self.xk = self.newton_optimizer.newton_update_xk(self.line_search_result.stp,
                                                                 self.trunc_nc_result.xsupi,
                                                                 self.xk)
                self.stat.iteration_counter += 1
                self.state = 'newton_optimization_next_round'

            if self.state == 'finished':
                return OptimizeResult(fun=self.fval, jac=self.gval, nfev=self.stat.nfev,
                                      njev=self.stat.njev, nhev=self.stat.nhev, status=0,
                                      success=True, message='Optimization terminated successfully.', x=self.xk,
                                      nit=self.stat.iteration_counter)

            if self.state == 'line_search_failed':
                return OptimizeResult(fun=self.fval, jac=self.gval, nfev=self.stat.nfev,
                                      njev=self.stat.njev, nhev=self.stat.nhev, status=1,
                                      success=False, message='Line search failed.', x=self.xk,
                                      nit=self.stat.iteration_counter)


if __name__ == '__main__':
    def calc_objective_function_and_gradient_at(req: NewtonCgStateMachine.RequestWDependent):
        return UpdateW(aggregated_objective_function_value=rosen(req.xk), aggregated_gradient=rosen_der(req.xk))


    def calc_hessp_at(req: NewtonCgStateMachine.RequestHessian):
        return UpdateS(aggregated_hessian=rosen_hess_prod(req.xk, req.psupi))


    def handle_computation_request(request):
        if isinstance(request, OptimizeResult):
            request: OptimizeResult
            return request
        elif isinstance(request, NewtonCgStateMachine.RequestWDependent):
            request: NewtonCgStateMachine.RequestWDependent
            return calc_objective_function_and_gradient_at(request)
        elif isinstance(request, NewtonCgStateMachine.RequestHessian):
            request: NewtonCgStateMachine.RequestHessian
            return calc_hessp_at(request)


    x0 = [2, -1]
    statemachine = NewtonCgStateMachine(x0)
    data = None
    while True:
        request = statemachine.next(data)
        data = handle_computation_request(request)
        if isinstance(data, OptimizeResult):
            print(data)
            break


    def all_finished(multiple: List[NewtonCgStateMachine]):
        for statemachine in multiple:
            if not statemachine.finished:
                return False
        return True


    multiple = [NewtonCgStateMachine([2, -1]), NewtonCgStateMachine([0.90, 1]), NewtonCgStateMachine([0, 0])]
    data = [None] * len(multiple)
    c = 0
    while not all_finished(multiple):
        c += 1
        for i, statemachine in enumerate(multiple):
            if not statemachine.finished:
                data[i] = handle_computation_request(statemachine.next(data[i]))
    print([statemachine.state for statemachine in multiple])
    print(data)
    print(c)
