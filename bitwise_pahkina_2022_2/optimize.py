"""Optimization implementation"""

import numpy as np

from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.result import Result
from pymoo.optimize import minimize
from pymoo.util.termination.default import SingleObjectiveDefaultTermination

from .network import BaseNetwork


class MyProblem(ElementwiseProblem):

    """Network optimization problem"""

    def __init__(self, base_network: BaseNetwork, **kwargs):
        self.base_network = base_network
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

    def is_equal(self, a, b):
        return a.get("hash")[0] == b.get("hash")[0]


def optimize(
    base_network: BaseNetwork, termination: dict | None = None, **kwargs
) -> Result:
    """Main optimization method

    Args:
        base_network (BaseNetwork): Base network for optimization
        termination (dict | None, optional): Keyword arguments
          for termination criteria. See
          :func:`~pymoo.util.termination.default.SingleObjectiveDefaultTermination`.
          Defaults to None.
        kwargs: keyword arguments to minimization problem

    Returns:
        Result: pymoo optimization result
    """
    problem = MyProblem(base_network)

    population_size = 2 * problem.n_var

    algorithm = BRKGA(
        n_elites=int(population_size * 0.2),
        n_offsprings=int(population_size * 0.7),
        n_mutants=int(population_size * 0.1),
        bias=0.7,
        eliminate_duplicates=MyElementwiseDuplicateElimination(),
    )

    termination = termination = SingleObjectiveDefaultTermination(**termination)

    res = minimize(problem, algorithm, termination, **kwargs)

    return res
