"""Optimization implementation"""

import logging
import os
from pathlib import Path
import shutil
import sys
import time

import networkx as nx
import numba
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import optuna
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.core.algorithm import Algorithm
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.core.result import Result
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.display import SingleObjectiveDisplay
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
from pyrecorder.recorder import Recorder
from pyrecorder.writers.streamer import Streamer

from .network import BaseNetwork, NetworkGraph, evaluate_many


class MyProblem(Problem):

    """Network optimization problem"""

    def __init__(self, network_json: os.PathLike, **kwargs):
        self.base_network = BaseNetwork.from_json(network_json)
        self.edges = self.base_network.get_edge_matrix()
        self.weights = self.base_network.get_weight_vector()

        super().__init__(
            n_var=len(self.base_network.edges),
            n_obj=1,
            n_constr=0,
            xl=0,
            xu=1,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        base_mat = self.base_network.to_adjacency_matrix()
        x_boolean = np.round(x).astype(bool)

        objective, hash_, x_final = evaluate_many(
            base_mat, x_boolean, self.edges, self.weights
        )

        out["F"] = objective
        out["hash"] = hash_
        out["x_final"] = x_final


class MyDuplicateElimination(DefaultDuplicateElimination):

    """Eliminate duplicates by comparing result hashes"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _do(self, pop: Population, other: Population, is_duplicate: np.ndarray):
        return self._do_compare(pop.get("hash"), other.get("hash"), is_duplicate)

    @staticmethod
    @numba.njit(parallel=True)
    def _do_compare(
        arr_a: np.ndarray, arr_b: np.ndarray | None, is_duplicate: np.ndarray
    ):
        # pylint: disable=not-an-iterable
        if arr_b is None:
            for i in numba.prange(len(arr_a)):
                for j in numba.prange(i + 1, len(arr_a)):
                    if arr_a[i] == arr_b[j]:
                        is_duplicate[i] = True
                        break
        else:
            for i in numba.prange(len(arr_a)):
                for j in numba.prange(len(arr_b)):
                    if arr_a[i] == arr_b[j]:
                        is_duplicate[i] = True
                        break

        return is_duplicate


class MyDisplay(SingleObjectiveDisplay):  # pylint: disable=too-few-public-methods

    """Modified display

    Include evaluations per second
    """

    def __init__(self, favg=True, n_gen_offset: int = 0, **kwargs):
        super().__init__(favg, **kwargs)
        self.n_gen_offset = n_gen_offset
        self._prev_time = time.time()

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        time_now = time.time()

        # Offset n_gen in printout
        self.output.attrs[0][1] += self.n_gen_offset

        self.output.append("x_opt", algorithm.opt[0].get("x_final").mean())
        self.output.append("x_avg", algorithm.pop.get("x_final").mean())
        self.output.append(
            "eval_per_s", algorithm.pop.size / (time_now - self._prev_time)
        )

        self._prev_time = time_now


class MyCallback(Callback):  # pylint: disable=too-few-public-methods

    """Callback on each generation

    Attributes:
        x_path (Path): location for X array storage file
        metric_log (Path): location for metric log file
    """

    def __init__(
        self,
        plot: bool = False,
        network_json: os.PathLike | None = None,
        x_path: os.PathLike | None = None,
        metric_log: os.PathLike | None = None,
        resume: bool = False,
        n_gen_offset: int = 0,
    ) -> None:
        super().__init__()

        self.x_path = Path(x_path)
        self.metric_log = Path(metric_log)

        self.base_network = (
            None if network_json is None else BaseNetwork.from_json(network_json)
        )
        self.rec = Recorder(Streamer()) if plot else None

        self._current_best_x = np.empty(0)
        self._current_obj_dist = np.empty(0)
        self._current_gen = 0
        self._obj_history = []
        self._f_max = 0.0

        for key in ["n_gen", "f_opt", "f_avg", "x_opt", "x_avg", "eval_per_s"]:
            self.data[key] = []
        self.n_gen_offset = n_gen_offset
        self._init_log(resume=resume)
        self._prev_time = time.time()

    def _init_log(self, resume=False):
        if not resume and self.metric_log.exists():
            self.metric_log.unlink()

        if not self.metric_log.exists():
            self.metric_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metric_log, "w", encoding="utf8") as metric_file:
                metric_file.write(",".join(self.data.keys()) + "\n")

    def notify(self, algorithm, **kwargs):
        time_now = time.time()

        opt = algorithm.opt[0]

        self._current_best_x = opt.get("x_final")
        self._current_obj_dist = algorithm.pop.get("F")
        self._current_gen = algorithm.n_gen
        self._obj_history.append(opt.F[0])

        self.data["n_gen"].append(algorithm.n_gen + self.n_gen_offset)
        self.data["f_opt"].append(opt.F[0])
        self.data["f_avg"].append(algorithm.pop.get("F").mean())
        self.data["x_opt"].append(self._current_best_x.mean())
        self.data["x_avg"].append(algorithm.pop.get("x_final").mean())
        self.data["eval_per_s"].append(
            algorithm.pop.size / (time_now - self._prev_time)
        )

        if self.x_path is not None:
            self._write_x_file(algorithm.pop.get("X"))

        if self.metric_log is not None:
            with open(self.metric_log, "a", encoding="utf8") as metric_file:
                line_vals = (str(val[-1]) for val in self.data.values())
                metric_file.write(",".join(line_vals) + "\n")

        self._prev_time = time_now

        if self.rec is not None:
            t0 = time.time()
            self._plot_results()
            self.rec.record()
            t1 = time.time()
            print(f"Plotting overhead: {t1-t0:.3f} s")

    def _write_x_file(self, x_array):
        """Write x array to file, using temporary .bak file"""
        handle_bak = self.x_path.exists()
        if handle_bak:
            bak_path = self.x_path.parent / (self.x_path.name + ".bak")
            shutil.copy(self.x_path, bak_path)

        np.save(self.x_path, x_array)

        if handle_bak:
            os.unlink(bak_path)

    def _plot_results(self):
        """Create matplotlib plot of the results, leave out plt.show()"""
        graph = nx.Graph()
        edges = [
            (u, v, self.base_network.weights[id_])
            for id_, (u, v) in self.base_network.edges.items()
        ]
        edges_actual = [
            (u, v)
            for id_, (u, v) in self.base_network.edges.items()
            if self._current_best_x[id_]
        ]
        edges_missing = [
            (u, v)
            for id_, (u, v) in self.base_network.edges.items()
            if not self._current_best_x[id_]
        ]
        graph.add_weighted_edges_from(edges)
        graph.add_nodes_from(self.base_network.nodes.keys())

        plt.figure(figsize=(14, 8))

        ax0: Axes = plt.subplot2grid((2, 5), (0, 0), 2, 3)
        ax1: Axes = plt.subplot2grid((2, 5), (0, 3), 1, 2)
        ax2: Axes = plt.subplot2grid((2, 5), (1, 3), 1, 2)

        # ax0 - graph
        ax0.set_title(f"Generation {self._current_gen}", y=0.97)
        nx.draw_networkx_nodes(
            graph, pos=self.base_network.nodes, node_size=50, node_color="gold", ax=ax0
        )
        nx.draw_networkx_edges(
            graph,
            pos=self.base_network.nodes,
            edgelist=edges_missing,
            edge_color="#C14242",
            alpha=0.3,
            ax=ax0,
        )
        nx.draw_networkx_edges(
            graph,
            pos=self.base_network.nodes,
            edgelist=edges_actual,
            ax=ax0,
        )
        ax0.set_aspect("equal")
        ax0.set_axis_off()

        # ax1 - distribution of objective values as histogram
        ax1.set_ylabel("Count")
        ax1.set_xlabel("Objective value")

        f_dist = self._current_obj_dist[~np.isinf(self._current_obj_dist)]
        min_val = (self._obj_history[-1] // 100) * 100
        self._f_max = max((f_dist.max() // 100 + 1) * 100, self._f_max)

        ax1.hist(
            f_dist, bins=min_val + np.arange((self._f_max - min_val) // 50 + 1) * 50
        )
        ax1.set_xlim(min_val, self._f_max)

        # ax2 - objective vs generations
        ax2.set_ylabel("Objective value")
        ax2.set_xlabel("Number of generations")
        ax2.plot(self._obj_history)

        plt.subplots_adjust(0.02, 0.06, 0.98, 0.98, 0.04, 0.15)


class BRKGAOptimization:

    """Optimization wrapper class

    Attributes:
        network_json (os.PathLike): Location to network specification
    """

    def __init__(self, network_json: os.PathLike) -> None:
        self.network_json = network_json

    def optimize(
        self, n_trials: int | None = None, verbose=True, optuna_prune=True, **kwargs
    ):

        if n_trials is None:
            res = self.optimize_single(verbose=verbose, **kwargs)
            if verbose:
                self._print_single_report(res)
        else:
            study = self.optimize_optuna(n_trials, optuna_prune, **kwargs)
            if verbose:
                self._print_optuna_report(study)

    def optimize_optuna(self, n_trials: int, optuna_prune=True, **kwargs):
        """Optimize using optuna hyperparameter optimization

        Args:
            n_trials (int): number of Optuna trials
        """

        def _optuna_objective(trial: optuna.trial.Trial):
            params = {
                "elite_frac": trial.suggest_float("elite_frac", 0.15, 0.25),
                "mutant_frac": trial.suggest_float("mutant_frac", 0.05, 0.15),
                "elite_bias": trial.suggest_float("elite_bias", 0.55, 0.75),
                "seed": trial.suggest_int("seed", 1, 10000),
            }

            res = self.optimize_single(
                trial=trial,
                verbose=False,
                optuna_prune=optuna_prune,
                **(params | kwargs),
            )
            return res.F[0]

        study_name = "bitwise-challenge-2022-2"
        storage_name = f"sqlite:///{study_name}.db"
        if not kwargs.get("resume", False):
            Path(storage_name).unlink(missing_ok=True)
        study = optuna.create_study(
            study_name=study_name, storage=storage_name, load_if_exists=True
        )
        study.optimize(_optuna_objective, n_trials=n_trials)

        return study

    def _print_optuna_report(self, study):
        print(f"Best value: {study.best_trial.value}")
        print(f"Best parameters: {study.best_trial.params}")

    def optimize_single(
        self,
        termination: dict | None = None,
        x_path: os.PathLike | None = None,
        metric_log: os.PathLike | None = None,
        resume: bool = False,
        plot: bool = False,
        seed: int | None = None,
        elite_frac: float = 0.2,
        mutant_frac: float = 0.1,
        elite_bias: float = 0.7,
        **kwargs,
    ) -> Result:
        """Main optimization method

        Args:
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
        problem = MyProblem(self.network_json)

        population_size = 2 * problem.n_var
        initial_nonrandom_count = 5

        if resume:
            sampling = np.load(x_path)
        else:
            rng = np.random.default_rng(seed)
            initial_feasible = _create_feasible_solutions(
                self.network_json, rng, initial_nonrandom_count
            )
            initial_random = rng.random(
                (population_size - initial_nonrandom_count, problem.n_var)
            )
            sampling = np.concatenate((initial_feasible, initial_random), axis=0)

        n_gen_offset = _get_n_gen_offset(metric_log=metric_log) if resume else 0

        algorithm = BRKGA(
            n_elites=int(population_size * elite_frac),
            n_offsprings=int(population_size * (1 - elite_frac - mutant_frac)),
            n_mutants=int(population_size * mutant_frac),
            bias=elite_bias,
            eliminate_duplicates=MyDuplicateElimination(),
            sampling=sampling,
            callback=MyCallback(
                plot=plot,
                network_json=self.network_json,
                x_path=x_path,
                metric_log=metric_log,
                resume=resume,
                n_gen_offset=n_gen_offset,
            ),
            display=MyDisplay(n_gen_offset=n_gen_offset),
        )

        termination = SingleObjectiveDefaultTermination(**termination)

        res = self.minimize(problem, algorithm, termination, seed=seed, **kwargs)

        return res

    def minimize(self, problem, algorithm: Algorithm, termination, **kwargs):
        algorithm.setup(problem, termination=termination, **kwargs)

        while algorithm.has_next():
            algorithm.next()

            optuna_trial = kwargs.get("trial")
            optuna_prune = kwargs.get("optuna_prune")
            if optuna_trial is not None:
                intermediate_value = algorithm.opt[0].F[0]
                optuna_trial.report(intermediate_value, step=algorithm.n_gen)
                if optuna_prune and optuna_trial.should_prune():
                    raise optuna.TrialPruned()

        res = algorithm.result()
        res.algorithm = algorithm
        return res

    def _print_single_report(self, res: Result):
        base_net = BaseNetwork.from_json(self.network_json)

        x_binary = res.opt.get("x_final")[0]
        del_edges = (x_binary == 0).nonzero()[0]
        remove_edges = np.array([base_net.edges[i] for i in del_edges.tolist()])

        base_mat = base_net.to_adjacency_matrix()
        new_net = NetworkGraph(base_mat)
        new_net.remove_edges(remove_edges)

        score = base_net.comparison_score(new_net)

        print(
            f"""
    Binary random key genetic algorithm, with hyperparameter optimization
    - Random initialization, with some pre-generated feasible results
    - Minimize score of modified network graph
    - Correction of infeasible solutions
    - Arguments: {' '.join(sys.argv[1:])}
    - {res.algorithm.n_gen} generations
    - Best objective value: {res.F[0]:.2f}
    - Best score: {score:.3f}
    - Execution time: {res.exec_time:.2f} s
    """
        )
        print("Solution:")
        print(", ".join(del_edges.astype(str)))


def _create_feasible_solutions(
    network_json: os.PathLike, rng: np.random.Generator, count: int
) -> np.ndarray:
    """Create some feasible solutions using heuristics"""
    base_network = BaseNetwork.from_json(network_json)
    arr = base_network.to_adjacency_matrix()
    edges = base_network.get_edge_matrix()
    random_orders = rng.permuted(
        np.tile(np.arange(len(edges)), count).reshape(count, len(edges)), axis=1
    )
    return _make_feasible_sols(arr, edges, random_orders)


@numba.njit(parallel=True)
def _make_feasible_sols(
    arr: np.ndarray, edges: np.ndarray, random_orders: np.ndarray
) -> np.ndarray:
    result = np.empty(random_orders.shape, np.float64)
    for i in numba.prange(len(random_orders)):  # pylint: disable=not-an-iterable
        result[i] = _make_feasible_sol(arr, edges, random_orders[i])
    return result


@numba.njit
def _make_feasible_sol(
    arr: np.ndarray, edges: np.ndarray, random_order: np.ndarray
) -> np.ndarray:
    """Generate feasible solution

    Args:
        arr (np.ndarray): base adjacency matrix
        edges (np.ndarray): matrix of all edges
        random_order (np.ndarray): order, in which edges are tried to be
          removed

    Returns:
        np.ndarray: 1/0 float array of same length as edges
    """
    net = NetworkGraph(arr)
    score = net.evaluate()
    x_created = np.full(len(edges), 1.0, np.float64)
    for i in random_order:
        original_weight = net.adjacency_matrix[edges[i, 0], edges[i, 1]]
        net.adjacency_matrix[edges[i, 0], edges[i, 1]] = np.inf
        net.adjacency_matrix[edges[i, 1], edges[i, 0]] = np.inf

        new_score = net.evaluate()
        if new_score < score:
            score = new_score
            x_created[i] = 0.0
        else:
            net.adjacency_matrix[edges[i, 0], edges[i, 1]] = original_weight
            net.adjacency_matrix[edges[i, 1], edges[i, 0]] = original_weight

    return x_created


def _get_n_gen_offset(metric_log: os.PathLike | None):
    n_gen = 0

    if metric_log is None or not Path(metric_log).exists():
        return n_gen

    with open(metric_log, "r", encoding="utf-8") as metric_file:
        line = ""
        for line in metric_file:
            pass  # skip to last line
        try:
            parts = line.strip().split(",")
            n_gen = int(parts[0])
        except (IndexError, ValueError):
            pass

    return n_gen
