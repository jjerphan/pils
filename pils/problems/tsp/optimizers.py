import os
import csv
import numpy as np

import optunity
from hyperopt import hp, tpe, fmin

from pils.optimizers import Optimizer
from pils.settings import BIN_FOLDER, clean_lines
from pils.problems.tsp.settings import TSP_INSTANCES_FOLDER, TSP_INSTANCES, NAIVE_COST_CSV, OPT_COST_CSV, \
    TSP_OPTS_FOLDER


class LocalTSPOptimizerTrait(Optimizer):
    """
    An abstract class to evaluate problem on local tsp directly.

    """

    def __init__(self, algo_path, opt_cost_csv_file: str=OPT_COST_CSV,
                 naive_cost_csv_file: str=NAIVE_COST_CSV,
                 tsp_instances: str=TSP_INSTANCES,
                 tsp_instances_folder: str=TSP_INSTANCES_FOLDER,
                 tsp_opts_folder: str=TSP_OPTS_FOLDER,
                 bin_folder: str=BIN_FOLDER):
        super().__init__(algo_path)

        self._opt_cost_csv_file = opt_cost_csv_file
        self._naive_cost_csv_file = naive_cost_csv_file
        self._tsp_instances = tsp_instances
        self._tsp_instances_folder = tsp_instances_folder
        self._tsp_opts_folder = tsp_opts_folder
        self._bin_folder = bin_folder
        if not(os.path.exists(self._bin_folder)):
            os.makedirs(self._bin_folder)

        self._dist_matrices = {
            instance: self._distance_matrix(os.path.join(self._tsp_instances_folder, instance)) \
            for instance in self._tsp_instances
        }
        self._opt_costs = self._csv_to_dict(self._opt_cost_csv_file)
        self._naive_costs = self._csv_to_dict(self._naive_cost_csv_file)

    @staticmethod
    def _parse_str(str, type):
        coords = tuple(map(type, str.split()))
        return coords

    @staticmethod
    def _parse_floats(str):
        return LocalTSPOptimizerTrait._parse_str(str, float)

    @staticmethod
    def _parse_ints(str):
        return LocalTSPOptimizerTrait._parse_str(str, int)

    @staticmethod
    def _get_opt_tour(opt_file):
        """
        Returns the optimal solution for a given file.

        :param opt_file: file containing the optimal solution
        :return:
        """
        with open(opt_file, "r") as f:
            lines = clean_lines(f.readlines())

        # 1 indexing to 0 indexing
        tour = np.array(list(map(LocalTSPOptimizerTrait._parse_ints, lines))).flatten() - 1
        return tour

    @staticmethod
    def _distance_matrix(file_case):
        """
        Return the distance matrix for a specific test case

        :param file_case:
        :return:
        """
        with open(file_case, "r") as f:
            lines = clean_lines(f.readlines())

        N = int(lines[0])
        del lines[0]

        points = np.array(list(map(LocalTSPOptimizerTrait._parse_floats, lines)))
        dist_matrix = np.zeros((N, N))

        for i in range(N):
            for j in range(i, N):
                dist_ij = np.sqrt(np.sum((points[i] - points[j]) ** 2))
                dist_matrix[i, j] = dist_ij
                dist_matrix[j, i] = dist_ij

        return dist_matrix

    @staticmethod
    def _get_cost(tour, dist_matrix):
        """
        Return the cost of the solution of a test case using an algo.

        :param tour: a list on int
        :param dist_matrix: the distance matrix
        """
        tour_edges = list(zip(tour[:-1], tour[1:]))
        tour_edges.append(([tour[-1]], tour[0]))

        # Compute the length of the tour
        cost = sum(list(map(lambda edge: dist_matrix[edge[0], edge[1]], tour_edges)))[0]
        return cost

    @staticmethod
    def _csv_to_dict(file):
        """
        Construct map from info present in a csv file of the form:
             case,          score
            a.tsp,           42.0
            b.tsp,         1337.0
                …,              …

        :param file: path to the csv file
        :return: a default
        """

        with open(file, "r") as f:
            lines = clean_lines(f.readlines())

        # Skip headers
        del lines[0]

        def make_key_value(line):
            parsed = line.split(",")
            return parsed[0], float(parsed[1])

        return dict(map(make_key_value, lines))

    @staticmethod
    def _get_tour(binary_path, test_case: str):
        """
        Returns the tour of a given test case for a binary

        :param binary_path:
        :param test_case:
        :return:
        """
        out_put = os.popen("{} < {}".format(binary_path, test_case)).read()
        tour = list(map(int, filter(lambda x: x != "", out_put.split("\n"))))
        return tour

    @staticmethod
    def _score(cost, instance, opt_costs, naive_costs):
        """
        Compute the score of a solution of an algo for a given instance
        :param cost: the cost of a solution
        :param instance: the name of the test case
        :return: the score of this test case
        """
        opt = opt_costs[instance]
        naive = naive_costs[instance]
        return 0.02 ** ((cost - opt) / (naive - opt))

    def save_opt_costs(self, verbose):
        with open(file=self._opt_cost_csv_file, mode="w+") as f:
            writer = csv.DictWriter(f=f, fieldnames=["test_case", "opt_cost"])
            writer.writeheader()
            for instance in self._tsp_instances:
                dist_mat = self._distance_matrix(os.path.join(self._tsp_instances_folder, instance))
                tour = self._get_opt_tour(os.path.join(self._tsp_opts_folder, instance.replace(".tsp", ".opt.tour")))
                cost = self._get_cost(tour, dist_mat)
                if verbose:
                    print("Case {} of cost {}".format(instance, cost))
                writer.writerow({"test_case": instance, "opt_cost": cost})

    def _get_final_score(self, hyperparameters, verbose):
        """
        Compile the algo, run it return the final score
        """
        binary = os.path.join(self._bin_folder, self._algo_name) + self._generate_id()
        macros = " ".join(list(map(lambda key: '-D{}="{}"'.format(key, hyperparameters[key]), hyperparameters)))
        compile_command = 'g++ {} {} -o {}'.format(self._algo_path, macros, binary)
        if verbose:
            print("Compiling with:")
            print(compile_command)

        os.system(compile_command)
        if verbose:
            print("Done Compiling")

        final_score = 0
        nb_cases = len(self._tsp_instances)
        for i, instance in enumerate(self._tsp_instances):
            dist_matrix = self._dist_matrices[instance]
            tour = self._get_tour(binary, os.path.join(self._tsp_instances_folder, instance))
            cost = self._get_cost(tour, dist_matrix)
            final_score += self._score(cost, instance, self._opt_costs, self._naive_costs)
            if verbose:
                print("{:3.2f} % Done {:>15s}".format((i + 1) / nb_cases * 100, instance))

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

        print(self._hyperparameters)
        best_hyperparameters, extra_info, solver_info = optunity.minimize(objective_function,
                                                                          solver_name=method,
                                                                          num_evals=number_evaluations,
                                                                          **self._hyperparameters)
        return best_hyperparameters


class LocalHyperOptTSPOptimizer(LocalTSPOptimizerTrait):
    """
    A local hyper-parameters optimizer using the `hyperopt` library.

    See https://github.com/hyperopt/hyperopt/wiki/FMin for usage

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
