# pils : Propulsing Iterated Local Search 💊

**Automatic Calibration of Local Search Algorithms Hyper-parameters (featuring Bayesian Hyper-Optimization)**

```python
from pils.optimizers import KattisOptunityOptimizer

if __name__ == "__main__":

    # Optimizing using an online judge
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

# 🚧 WIP : more to come soon!