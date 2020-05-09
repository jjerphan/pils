import unittest
import os

from pils.problems.tsp.optimizers import LocalHyperOptTSPOptimizer


class Functional_Tests(unittest.TestCase):

    def setUp(self):
        self.cpp = os.path.abspath(os.path.join("fixtures", "tsp.cpp"))

    def test_LocalHyperOptTSPOptimizer(self):
        optimizer = LocalHyperOptTSPOptimizer(algo_path=self.cpp)

        optimizer.register_hyperparameter(T_0=[10000, 60000])
        optimizer.register_hyperparameter(ALPHA=[0.00000001, 0.00001])
        optimizer.register_hyperparameter(TT_ADJUST=[0.01, 15])

        best_hyperparameters = optimizer.optimize(number_evals=1,
                                                  verbose=True)

        print("Best parameters:", best_hyperparameters)
