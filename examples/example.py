from pils.optimizers import KatisOptunityOptimizer

if __name__ == "__main__":

    optimizer = KatisOptunityOptimizer(algo_path="tsp.cpp",
                                       credential_file=".credentials")

    optimizer.register_hyperparameter(T_0=[10000, 60000])
    optimizer.register_hyperparameter(ALPHA=[0.00000001, 0.00001])
    optimizer.register_hyperparameter(TT_ADJUST=[0.01, 15])

    best_hyperparameters = optimizer.optimize(number_evals=100,
                                              results_file="tsp.csv",
                                              verbose=True)

    print("Best parameters:", best_hyperparameters)
