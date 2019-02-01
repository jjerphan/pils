# pils : Propulsing Iterated Local Search 💊

**Automatic Calibration of Local Search Algorithms Hyper-parameters (featuring Bayesian Hyper-Optimization)**

## Use cases

You are designing a [local search algorithm](https://en.wikipedia.org/wiki/Local_search_%28optimization%29) to try to find a relatively good solution to a (generally NP-hard) 
problem, like TSP.
Since you are clever, you have used your favorite set of heuristics.
Such heuristics generally come with *hyperparameters*.

Your algorithm is working and is yielding quite good solution.
However you know that *you can greatly improve it calibrating the set of hyperparameters*.

Tuning those parameters by hand is highly boring and misleading.
So you would like those parameters to be tuned it automatically, but you already spent all your time and energy on the design of heuristics and thus don't want to spend some more time on the design of *hyperparameters optimizers*.

*Here is where `pils` comes* into play.

## Quick example with TSP

You have designed an algorithm for TSP and you wrote a C++ implementation of it in [`tsp.cpp`](./examples/tsp.cpp).

This algorithm is parametrized by:
 - `T_0`: the initial temperature for simulated annealing ;
 - `ALPHA`: the decay rate for simulated annealing ;
 - `TT_ADJUST` : the adjustment for taboo tenure.
 
You have some prior knowledge of the range of values of those parameters.

### Optimizing locally

You would like to test your algorithm locally, but you don't have any instances of the problem. 

Here is [a snippet](examples/example_local.py) you can write using `pils` to calibrate your algorithm
(using [Optunity](http://optunity.readthedocs.io/)) on provided instances:

```python
from pils.problems.tsp.optimizers import LocalOptunityTSPOptimizer

if __name__ == "__main__":

    # Optimizing locally
    optimizer = LocalOptunityTSPOptimizer(algo_path="tsp.cpp")

    # Hyperparameters of the algorithm
    optimizer.register_hyperparameter(T_0=[10000, 60000])
    optimizer.register_hyperparameter(ALPHA=[0.00000001, 0.00001])
    optimizer.register_hyperparameter(TT_ADJUST=[0.01, 15])

    # Magic happens here
    best_hyperparameters = optimizer.optimize(number_evals=100,
                                              results_file="tsp.csv",
                                              verbose=True)
                                              
    # 🚀
    print("Best parameters:", best_hyperparameters)
```

### Optimizing against Kattis
You would like to optimize your algorithm against [Kattis](https://open.kattis.com/), an online Judge.

Here is [a snippet](examples/example.py) you can write using `pils` to calibrate your algorithm (using [Optunity](http://optunity.readthedocs.io/)):
```python
from pils.optimizers import KattisOptunityOptimizer

if __name__ == "__main__":

    # Optimizing against Kattis
    optimizer = KattisOptunityOptimizer(algo_path="tsp.cpp",
                                       id_problem="tsp",
                                       credential_file=".credentials")

    # Hyperparameters of the algorithm
    optimizer.register_hyperparameter(T_0=[10000, 60000])
    optimizer.register_hyperparameter(ALPHA=[0.00000001, 0.00001])
    optimizer.register_hyperparameter(TT_ADJUST=[0.01, 15])

    # Magic happens here
    best_hyperparameters = optimizer.optimize(number_evals=100,
                                              results_file="tsp.csv",
                                              verbose=True)

    # 🚀
    print("Best parameters:", best_hyperparameters)
```

## Installing

As `pils` is still in development, is it for now available on [TestPypi](https://test.pypi.org/).
You can install it using this:

```bash
$ pip install --index-url https://test.pypi.org/project/ pils
```

## About `Optimizers`

This optimizers are built using *Bayesian Optimization* libraries:
 - [HyperOpt](https://github.com/hyperopt/hyperopt/) that proposes an implementation of the [Tree-Structured Parzen 
 Estimator](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) (TPE), that shows 
 relatively good performances.
 - [Optunity](http://optunity.readthedocs.io/) that wraps HyperOpt's TPE implementation and that proposes other
 optimizers.

 
The different optimizer implementations (whose name is self documenting) available are:
 - `LocalOptunityTSPOptimizer`
 - `LocalHyperOptTSPOptimizer`
 - `KattisOptunityOptimizer`
 - `KattisHyperOptOptimizer`


## License

[This project license](./LICENSE) is MIT.