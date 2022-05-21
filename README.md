# Network optimizer with genetic algorithm

This repository is in response to [Bitwise code challenge 2022 2/3](https://bitwise.fi/koodipahkina/).
There is a network of nodes that are connected with some edges.
The goal is to remove as much edges as possible, while maximizing total score P on

$$ P = (P_{og} / P_r - 1) \cdot 1000 \,,$$

where $P_{og}$ is the original score, $P_r$ new score of a network, where edges are removed, and $P_i$ are calculated by

$$ P_i = A W_t + B D_{avg} \, . $$

Here, $A=0.1$, $B=2.1$, $W_t$ is the sum of length of all arcs and $D_{avg}$ average shortest path length from each individual point to any other point.

Original data is included in this repository at [bitwise_challenge_2022_2/koodipahkina-data.json](bitwise_challenge_2022_2/koodipahkina-data.json).


## Quickstart

Requirements:
- Python 3.10
- Poetry (see [installation instructions][poetry-install])


```sh
poetry install  # Install environment and dependencies
poetry shell    # Enter environment
python -m bitwise_challenge_2022_2  # Run optimization
```

Please see program help for options:

```
$ python -m bitwise_challenge_2022_2 --help
usage: __main__.py [-h] [--n-trials N_TRIALS] [--optuna-prune]
                   [--restart RESTART] [--metric-log METRIC_LOG]
                   [--xpath XPATH] [--resume] [--multiprocessing] [--quiet]
                   [--plot] [--seed SEED] [--elite-frac ELITE_FRAC]
                   [--mutant-frac MUTANT_FRAC] [--elite-bias ELITE_BIAS]

Optimize liana network.

options:
  -h, --help            show this help message and exit
  --n-trials N_TRIALS   Number of trials for hyperparameter optimization.
                        Default: None
  --optuna-prune        Use pruning in hyperparameter optimization. Default:
                        no pruning
  --restart RESTART     Restart specification in format
                        key1:value1,key2:value2. Available values: see keyword
                        arguments at
                        https://pymoo.org/interface/termination.html .
                        Default: n_max_evals:1000000,n_last:500,n_max_gen:1000
  --metric-log METRIC_LOG
                        Location for logging optimization running metrics.
                        Default: opt_log.txt
  --xpath XPATH         Location for saving intermediate x values for
                        population. Default: opt_X_latest.npy
  --resume              Resume from xfile
  --quiet, -q           Suppress output
  --plot                Display progress plot during calculation
  --seed SEED           Random seed. Default: 1
  --elite-frac ELITE_FRAC
                        Elite fraction of population. Default: 0,2
  --mutant-frac MUTANT_FRAC
                        Mutant fraction of population. Default: 0.1
  --elite-bias ELITE_BIAS
                        Probability of elite gene transfer. Default: 0.7
```


## Implementation

The optimization is built on [pymoo] optimization framework, more specifically its [biased random key genetic algorithm][pymoo-brkga] implementation.
A population of vectors is initialized, where vector length is equal to the number of original edges in the network.
These vectors represent, whether the edge is to be removed ($x_i < 0.5$) or kept.
Some initial vectors are constructed with a greedy algorithm, others generated randomly.
All vectors are repaired so, that they produce a feasible solution:

1. find edge from set of not-connected edges that has minimum weight, and has one end in the graph and the other end in a not-connected node.
2. Add the edge to the graph.
3. Continue until graph is connected.

The genetic algorithm then is used to minimize the network score $P_r$.

A hyperparameter optimization framework [optuna] is also included, but tuning
the BRKGA parameters did not result in any significant findings.
For hyperparameter optimization, use flag `--n-trials`.
By default, it is not used.

[Numba][numba] Python compiler is used to make calculations faster.
On an Intel i5-8350U CPU (Lenovo T480), the program runs at 600&ndash;800 evaluations per second, which results in a reasonable runtime.

For competition result, the optimizer was run on [DataCrunch.io][datacrunch.io] server with flag `--restart n_max_evals:10000000,n_last:5000,n_max_gen:5000`.
A final score $P=1285.075$ was achieved.
For the competition entry details, see also output from

    git tag v1.4


## Extras

If installed with `poetry install --dev`, a library for plotting is included.
Thus intermediate logs by the optimizer can be visualized with [scripts/plot_optimization_log.py](scripts/plot_optimization_log.py); see help with

    python scripts/plot_optimization_log.py --help


[datacrunch.io]: https://datacrunch.io/
[numba]: https://numba.pydata.org/
[optuna]: https://optuna.org/
[poetry-install]: https://python-poetry.org/docs/#installation
[pymoo]: https://pymoo.org/
[pymoo-brkga]: https://pymoo.org/algorithms/soo/brkga.html
