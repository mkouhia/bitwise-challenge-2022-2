"""Optimization implementation"""

import multiprocessing
import os

import numpy as np
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.core.duplicate import (
    ElementwiseDuplicateElimination,
    DefaultDuplicateElimination,
)
from pymoo.core.problem import ElementwiseProblem, starmap_parallelized_eval
from pymoo.core.result import Result
from pymoo.optimize import minimize
from pymoo.util.termination.default import SingleObjectiveDefaultTermination

from .network import BaseNetwork


class MyProblem(ElementwiseProblem):

    """Network optimization problem"""

    def __init__(self, network_json: os.PathLike, **kwargs):
        network_json = BaseNetwork.from_json(network_json)
        self.base_network = network_json

        super().__init__(
            n_var=len(self.base_network.edges),
            n_obj=1,
            n_constr=1,
            xl=0,
            xu=1,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        x_binary = np.round(x).astype(int)
        del_edges = (x_binary == 0).nonzero()[0]

        new_net = self.base_network.as_graph(remove_edges=del_edges.tolist())

        out["F"] = new_net.evaluate()
        out["G"] = -1 if new_net.is_connected else 1
        out["pheno"] = x_binary
        out["hash"] = hash(str(del_edges))


class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):

    """Eliminate duplicates based on result hash"""

    def __init__(self, cmp_func=None, **kwargs) -> None:
        super().__init__(self.is_equal, **kwargs)

    def is_equal(self, a, b):
        return a.get("hash")[0] == b.get("hash")[0]


def optimize(
    network_json: os.PathLike, termination: dict | None = None, **kwargs
) -> Result:
    """Main optimization method

    Args:
        network_json (os.PathLike): Location to network specification
        termination (dict | None, optional): Keyword arguments
          for termination criteria. See
          :func:`~pymoo.util.termination.default.SingleObjectiveDefaultTermination`.
          Defaults to None.
        kwargs: keyword arguments to minimization problem

    Returns:
        Result: pymoo optimization result
    """
    n_threads = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(n_threads)

    problem = MyProblem(
        network_json, runner=pool.starmap, func_eval=starmap_parallelized_eval
    )

    algorithm = BRKGA(
        n_elites=200,
        n_offsprings=700,
        n_mutants=100,
        bias=0.7,
        eliminate_duplicates=MyElementwiseDuplicateElimination(),
    )

    termination = termination = SingleObjectiveDefaultTermination(**termination)

    res = minimize(problem, algorithm, termination, **kwargs)

    return res
