# The following is a derived work of the
# optimize.py module by Travis E. Oliphant
# https://github.com/scipy/scipy/blob/master/scipy/optimize/optimize.py
# which is part of the scipy package
import _queue
import threading
import time
from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache, wraps
from queue import Queue
from typing import List, Optional, Union, Callable, Type

import numpy as np
import scipy.optimize
from nptyping import NDArray, Float64
from numpy import ndarray
from scipy.optimize.optimize import OptimizeResult


class Optimizer(object):
    pass


def np_cache(fn):
    """Cache results for a method called with one or more numpy vectors.
    Derived from work of StackOverflow user Martijn Pieters available at
    https://stackoverflow.com/questions/52331944/cache-decorator-for-numpy-arrays/52332109#52332109.
    Adapted for multiple vectors.
    """

    @lru_cache(maxsize=1)
    def cached_wrapper(*hashable_arrays):
        arrays = [np.array(hashable_array) for hashable_array in hashable_arrays]
        return fn(*arrays)

    @wraps(fn)
    def wrapper(*arrays):
        hashable_arrays = [tuple(array) for array in arrays]
        return cached_wrapper(*hashable_arrays)

    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


class SteppedEventBasedNewtonCgOptimizer(Optimizer):
    """Performs the minimization of the objective function value using the Newton-CG method of
    `scipy.optimize.minimize`, but allows for a federated calculation by using an event-based system that interrupts
    the calculation of the inner optimization function till results from the clients are available.

    The Newton-CG method is dependent on the objective function value and gradient value (Jacobian) for a given weight,
    and the product of the Hessian matrix of the objective function times an arbitrary vector (called Hessp).

    This class exposes `Request`s for the calculation of values dependent on a new weight vector (both the objective
    function value and its gradient) as an instance of the `RequestWDependent` class and requests for the calculation
    of the Hessp as an instance of the `RequestHessp` class.
    Pending requests are retrievable using the `check_pending_requests` method in a blocking or unblocking way.

    Wrapping logic should mandate requests to the clients of the federated calculation. Responses are returned to the
    optimizer in the form of either a `ResolvedWDependent` instance when fulfilling `RequestWDependent` requests or
    analogous an instance of `ResolvedHessp` in the case of a `RequestHessp` request.
    Submitting the resolve of a request is done by calling the `resolve` method. On offering a wrong type of
    Resolve to this method, an `UnexpectedResolveType` Exception will be raised.

    When the optimization is finished (you can check this using the finished property of the instance), the result is
    made available as the result property in the form of a `scipy.optimize.optimize.OptimizeResult` instance.
    Timings of the calculation tracking calculation and idle times are attached to this object.
    """

    class Request:
        """Request class for dispatching calculation requests from the optimizer."""
        pass

    @dataclass
    class RequestWDependent(Request):
        """Request for the calculation of the objective function value and gradient vector at the point xk.
        Should be resolved with an ResolveWDependent response."""
        xk: NDArray

    @dataclass
    class RequestHessp(Request):
        """Request for the calculation of the Hessian matrix times the psupi vector.
        Should be resolved with an ResolveWDependent response."""
        xk: NDArray
        psupi: NDArray

    class Resolved:
        """Class for responses to requests."""
        pass

    @dataclass
    class ResolvedWDependent(Resolved):
        """Response to a RequestWDependent. Contains the value of the objective function and gradient vector at the
        point given in the request."""
        f_val: float
        g_val: NDArray[Float64]

    @dataclass
    class ResolvedHessp(Resolved):
        """Response to a RequestHessp. Contains the value of the Hessian matrix times the vector given in the
        request."""
        hessp_val: NDArray[Float64]

    class UnexpectedResolveType(Exception):
        """Exception that is raised if the wrong type of Resolve is provided after a request.
        Requests of the type `RequestWDependent` should be fulfilled with a `ResolvedWDependent` response while
        requests of the type `RequestHessp` should be fulfilled with a `ResolvedHessp` response."""
        pass

    def __init__(self, x0: Union[NDArray[float], List[Union[float, int]]], **kwargs):
        """Init instance and start minimization."""
        self._incoming: Queue[SteppedEventBasedNewtonCgOptimizer.Resolved] = Queue(maxsize=1)
        self._outgoing: Queue[SteppedEventBasedNewtonCgOptimizer.Request] = Queue(maxsize=1)
        self._expected_resolve_type: Optional[Type[SteppedEventBasedNewtonCgOptimizer.Resolved]] = None

        self._thread: threading.Thread
        self._result: Optional[OptimizeResult] = None

        self._total_time = 0
        self._idle_time = 0

        self._start_minimize(x0, **kwargs)

    @property
    def finished(self) -> bool:
        """Check if optimization is finished."""
        return self._result is not None

    @property
    def result(self) -> Optional[OptimizeResult]:
        """Get result of the optimization. If available, timings are attached to the OptimizeResult."""
        optimize_result = self._result
        if optimize_result is None:
            return optimize_result
        optimize_result['timings'] = self.timings
        return optimize_result

    @property
    def calculation_time(self):
        """Calculate the time for the calculation."""
        return self._total_time - self._idle_time

    @property
    def timings(self):
        """Generate a timings tuple."""
        Timings = namedtuple('Timings', ['calculation_time', 'total_time', 'idle_time'])
        return Timings(self.calculation_time, self._total_time, self._idle_time)

    def _register_request(self, request: Request):
        """Put a request to the outgoing queue."""
        self._outgoing.put_nowait(request)

    def _request_calculations_depending_on_w(self, x: NDArray):
        """Given an updated x generate a request for f_val and g_val and expose it."""
        self._expected_resolve_type = self.ResolvedWDependent
        self._register_request(self.RequestWDependent(xk=x))

    def _request_calculations_depending_on_s(self, x: NDArray, p: NDArray):
        """Given an updated x and p generate a request for hessp_val and expose it."""
        self._expected_resolve_type = self.ResolvedHessp
        self._register_request(self.RequestHessp(xk=x, psupi=p))

    def has_pending(self) -> bool:
        """Check if requests are available."""
        return not self._outgoing.empty()

    def check_pending_requests(self, block: bool = True, timeout: int = None) -> Request:
        """Get request."""
        return self._outgoing.get(block=block, timeout=timeout)

    def resolve(self, resolved: Resolved):
        """Resolve a request."""
        if isinstance(resolved, self._expected_resolve_type):
            self._incoming.put_nowait(resolved)
        else:
            raise self.UnexpectedResolveType

    def _get_resolved(self, block: bool = True):
        """Return resolved for a dispatched request."""
        tic = time.perf_counter()
        resolved = self._incoming.get(block=block)
        toc = time.perf_counter()
        self._idle_time += toc - tic
        return resolved

    def _updateWDependent(self, w):
        """Request calculations depending on an updated w and read back results which update f_val and g_val."""
        self._request_calculations_depending_on_w(w)
        result: SteppedEventBasedNewtonCgOptimizer.Resolved = self._get_resolved()

        if not isinstance(result, SteppedEventBasedNewtonCgOptimizer.ResolvedWDependent):
            raise self.UnexpectedResolveType()
        result: SteppedEventBasedNewtonCgOptimizer.ResolvedWDependent

        self.f_val = result.f_val
        self.g_val = result.g_val

    def _updateHessian(self, x, p):
        """Request calculations depending on an updated w and p and read back results which update hessp_val."""
        self._request_calculations_depending_on_s(x, p)
        result: SteppedEventBasedNewtonCgOptimizer.Resolved = self._get_resolved()

        if not isinstance(result, SteppedEventBasedNewtonCgOptimizer.ResolvedHessp):
            raise self.UnexpectedResolveType()
        result: SteppedEventBasedNewtonCgOptimizer.ResolvedHessp

        self.hessp_val = result.hessp_val

    def _start_minimize(self, x0: Union[NDArray[float], List[Union[float, int]]], **kwargs):
        """Start minimization of the weight vector x0."""

        @np_cache
        def _do_objective_func(w: NDArray):
            self._updateWDependent(w)
            return self.f_val

        @np_cache
        def _do_gradient_func(w: NDArray):
            return self.g_val  # value was already updated in the _do_objective_function_call

        @np_cache
        def _do_hess_prod(x, p):
            self._updateHessian(x, p)
            return self.hessp_val

        def _time_and_save_optimizer_result(**kwargs):
            """Wrapper for calling the scipy minimizer with timings."""
            tic = time.perf_counter()
            self._result = scipy.optimize.minimize(
                fun=_do_objective_func,
                x0=x0,
                method='newton-cg',
                jac=_do_gradient_func,
                hessp=_do_hess_prod,
                **kwargs
            )
            toc = time.perf_counter()
            self._total_time = toc - tic

        self._thread = threading.Thread(target=_time_and_save_optimizer_result, kwargs=kwargs)
        self._thread.start()

    def solve(self, handler: Callable[[Request], Resolved], recheck_timeout=1):
        """Solve method for convenience. Provide a request handler to get the result of the optimization."""
        while not self.finished:
            request: SteppedEventBasedNewtonCgOptimizer.Request
            try:
                # if no pending request is found after timeout, the finished property is rechecked
                request = self.check_pending_requests(block=False, timeout=recheck_timeout)
            except _queue.Empty:
                continue

            self.resolve(handler(request))

        return self.result
