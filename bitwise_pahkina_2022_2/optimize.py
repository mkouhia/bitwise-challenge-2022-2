"""Optimization implementation"""

import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.optimize import minimize

from .network import BaseNetwork


class MyProblem(ElementwiseProblem):
    
    """Network optimization problem"""

    def __init__(self, base_network: BaseNetwork):
        self.base_network = base_network
        
        super().__init__(
            n_var=len(self.base_network.edges),
            n_obj=1, n_constr=0, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        x_binary = np.round(x).astype(int)
        del_edges = (x_binary==0).nonzero()[0]

        new_net = self.base_network.as_graph(remove_edges=del_edges.tolist())
        
        out["F"] = -self.base_network.comparison_score(new_net)
        out["pheno"] = x_binary
        out["hash"] = hash(str(del_edges))


class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):

    """Eliminate duplicates based on result hash"""

    def is_equal(self, a, b):
        return a.get("hash")[0] == b.get("hash")[0]
    

def optimize(base_network: BaseNetwork):
    """Main optimization method"""
    np.random.seed(2)
    problem = MyProblem(base_network)

    algorithm = BRKGA(
        n_elites=200,
        n_offsprings=700,
        n_mutants=100,
        bias=0.7,
        eliminate_duplicates=MyElementwiseDuplicateElimination())

    res = minimize(problem,
                algorithm,
                ("n_gen", 75),
                seed=1,
                verbose=True)

    print(f"Best solution found: \nF = {res.F}")
    
    best_x_binary = res.opt.get("pheno")[0]
    removed_edges = (best_x_binary==0).nonzero()[0]
    print("Solution:",)
    print(', '.join(removed_edges.astype(str)))
