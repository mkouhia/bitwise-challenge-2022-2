"""Optimization implementation"""

import os
from pathlib import Path
import time
import numpy as np

from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.core.callback import Callback
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.result import Result
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.display import SingleObjectiveDisplay
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


class MyDisplay(SingleObjectiveDisplay):  # pylint: disable=too-few-public-methods

    """Modified display

    Include evaluations per second
    """

    def __init__(self, favg=True, **kwargs):
        super().__init__(favg, **kwargs)
        self._prev_time = time.time()

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        time_now = time.time()

        self.output.append(
            "eval_per_s", algorithm.pop.size / (time_now - self._prev_time)
        )

        self._prev_time = time_now


class MyCallback(Callback):

    """Callback on each generation"""

    def __init__(
        self,
        x_path: os.PathLike | None = None,
        metric_log: os.PathLike | None = None,
    ) -> None:
        super().__init__()

        self.x_path = Path(x_path)
        self.metric_log = Path(metric_log)

        for key in ["n_gen", "time", "fopt"]:
            self.data[key] = []
        self._init_log()

    def _init_log(self):
        if self.metric_log.exists():
            self.metric_log.unlink()
        else:
            self.metric_log.parent.mkdir(parents=True, exist_ok=True)

        with open(self.metric_log, "w", encoding="utf8") as metric_file:
            metric_file.write(",".join(self.data.keys()) + "\n")

    def notify(self, algorithm, **kwargs):
        self.data["n_gen"].append(algorithm.n_gen)
        self.data["time"].append(time.time())
        self.data["fopt"].append(algorithm.pop.get("F").min())

        if self.x_path is not None:
            population_x = algorithm.pop.get("X")
            np.save(self.x_path, population_x)

        if self.metric_log is not None:
            with open(self.metric_log, "a", encoding="utf8") as metric_file:
                line_vals = (str(self.data[key][-1]) for key in self.data)
                metric_file.write(",".join(line_vals) + "\n")


def optimize(
    base_network: BaseNetwork,
    termination: dict | None = None,
    x_path: os.PathLike | None = None,
    metric_log: os.PathLike | None = None,
    resume: bool = False,
    **kwargs
) -> Result:
    """Main optimization method

    Args:
        base_network (BaseNetwork): Base network for optimization
        termination (dict | None, optional): Keyword arguments
          for termination criteria. See
          :func:`~pymoo.util.termination.default.SingleObjectiveDefaultTermination`.
          Defaults to None.
        x_path (os.PathLike | None): location for saving intermediate
          x values for population.
        metric_log (os.PathLike | None): location for logging
          optimization running metrics.
        resume (bool): continue from previous result in x_path.
          Defaults to False.
        kwargs: keyword arguments to minimization problem

    Returns:
        Result: pymoo optimization result
    """
    problem = MyProblem(base_network)

    sampling = np.load(x_path) if resume else FloatRandomSampling()

    population_size = 2 * problem.n_var

    algorithm = BRKGA(
        n_elites=int(population_size * 0.2),
        n_offsprings=int(population_size * 0.7),
        n_mutants=int(population_size * 0.1),
        bias=0.7,
        eliminate_duplicates=MyElementwiseDuplicateElimination(),
        sampling=sampling,
        callback=MyCallback(x_path=x_path, metric_log=metric_log),
        display=MyDisplay(),
    )

    termination = termination = SingleObjectiveDefaultTermination(**termination)

    res = minimize(problem, algorithm, termination, **kwargs)

    return res
