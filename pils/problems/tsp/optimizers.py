import os

import optunity
from hyperopt import hp, tpe, fmin

from pils.optimizers import Optimizer
from pils.settings import BIN_FOLDER
from pils.problems.tsp.settings import TSP_INSTANCES_FOLDER, TSP_INSTANCES, NAIVE_COST_CSV, OPT_COST_CSV, \
    TSP_OPTS_FOLDER
from pils.problems.tsp.instance_handler import TSPInstanceHandler


class LocalTSPOptimizerTrait(Optimizer):
    """
    An abstract class to evaluate problem on local tsp directly.

    """

    def __init__(self, algo_path, opt_cost_csv_file=OPT_COST_CSV,
                 naive_cost_csv_file=NAIVE_COST_CSV,
                 tsp_instances=TSP_INSTANCES,
                 tsp_instances_folder=TSP_INSTANCES_FOLDER,
                 tsp_opts_folder=TSP_OPTS_FOLDER):
        super().__init__(algo_path)
        self._handler = TSPInstanceHandler(opt_cost_csv_file=opt_cost_csv_file,
                                           naive_cost_csv_file=naive_cost_csv_file,
                                           tsp_instances=tsp_instances,
                                           tsp_instances_folder=tsp_instances_folder,
                                           tsp_opts_folder=tsp_opts_folder)

    def _get_final_score(self, hyperparameters, verbose):
        """
        Compile the algo, run it return the final score
        """
        binary = os.path.join(BIN_FOLDER, self._algo_name) + self._generate_id()
        macros = " ".join(list(map(lambda key: '-D{}="{}"'.format(key, hyperparameters[key]), hyperparameters)))
        compile_command = 'g++ {} {} -o {}'.format(self._algo_path, macros, binary)
        if verbose:
            print("Compiling with")
            print(compile_command)

        os.system(compile_command)

        # TODO : deleguate to handler ?
        final_score = 0
        nb_cases = len(TSP_INSTANCES)
        for i, case in enumerate(TSP_INSTANCES):
            dist_matrix = self._dist_matrices[case]
            tour = self.get_tour(binary, os.path.join(TSP_INSTANCES_FOLDER, case))
            cost = get_cost(tour, dist_matrix)
            final_score += self.score(cost, case, self._opt_costs, self._naive_costs)
            if verbose:
                print("{:3.2f} % Done {:>15s}".format((i + 1) / nb_cases * 100, case))

        return final_score


class LocalOptunityTSPOptimizer(LocalTSPOptimizerTrait):
    """
    A local hyper-parameters optimizer using the `optunity` library.

    """

    def __init__(self, algo_path):
        super().__init__(algo_path)

    def _get_best_hyperparameters(self, objective_function, method, number_evaluations):
        if method is None:
            method = "tpe"

        best_hyperparameters, extra_info, solver_info = optunity.minimize(objective_function,
                                                                          solver_name=method,
                                                                          num_evals=number_evaluations,
                                                                          **self._hyperparameters)
        return best_hyperparameters


class LocalHyperOptTSPOptimizer(LocalTSPOptimizerTrait):
    """
    A local hyper-parameters optimizer using the `hyperopt` library.

    See `hyper_opt_ex.py` and https://github.com/hyperopt/hyperopt/wiki/FMin for usage

    """

    def __init__(self, algo_path):
        super().__init__(algo_path)

    def _get_best_hyperparameters(self, objective_function, method, number_evaluations):
        hyper_opt_hp = dict(map(lambda key: (key, hp.uniform(key, *self._hyperparameters[key])), self._hyperparameters))

        if method is None:
            method = tpe.suggest

        # A tweak
        def unfold_positional_args(x):
            return objective_function(**x)

        best_hyperparameters = fmin(unfold_positional_args,
                                    hyper_opt_hp,
                                    algo=method,
                                    max_evals=number_evaluations)

        return best_hyperparameters
