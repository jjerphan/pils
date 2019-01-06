import csv

import numpy as np
import os

from pils.problems.tsp.settings import TSP_OPTS_FOLDER, TSP_INSTANCES_FOLDER, TSP_INSTANCES, OPT_COST_CSV, \
    NAIVE_COST_CSV
from pils.settings import clean_lines


class TSPInstanceHandler:

    def __init__(self, opt_cost_csv_file: str, naive_cost_csv_file: str, tsp_instances: list,
                 tsp_instances_folder: str, tsp_opts_folder: str):

        self._opt_cost_csv_file = opt_cost_csv_file
        self._naive_cost_csv_file = naive_cost_csv_file
        self._tsp_instances = tsp_instances
        self._tsp_instances_folder = tsp_instances_folder
        self._tsp_opts_folder = tsp_opts_folder

        # Static attributes for local instances
        self._dist_matrices = {instance: self._distance_matrix(os.path.join(self._tsp_instances_folder, instance)) \
                               for instance in self._tsp_instances}
        self._opt_costs = self._csv_to_dict(self._opt_cost_csv_file)
        self._naive_costs = self._csv_to_dict(self._naive_cost_csv_file)

    @staticmethod
    def _parse_str(str, type):
        coords = tuple(map(type, str.split()))
        return coords

    @staticmethod
    def _parse_floats(str):
        return TSPInstanceHandler._parse_str(str, float)

    @staticmethod
    def _parse_ints(str):
        return TSPInstanceHandler._parse_str(str, int)

    @staticmethod
    def _get_opt_tour(opt_file):
        """
        Returns the cost of the optimal solution for a given file.

        :param opt_file: file containing the optimal solution
        :return:
        """
        with open(opt_file, "r") as f:
            lines = clean_lines(f.readlines())

        # 1 indexing to 0 indexing
        tour = np.array(list(map(TSPInstanceHandler._parse_ints, lines))).flatten() - 1
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

        points = np.array(list(map(TSPInstanceHandler._parse_floats, lines)))
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
    def get_tour(binary_path, test_case: str):
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
    def score(cost, case, opt_costs, naive_costs):
        """
        Compute the score of a solution of an algo for a given instance
        :param cost: the cost of a solution
        :param case: the name of the test case
        :return: the score of this test case
        """
        opt = opt_costs[case]
        naive = naive_costs[case]
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


if __name__ == "__main__":
    handler = TSPInstanceHandler(opt_cost_csv_file=OPT_COST_CSV,
                                 naive_cost_csv_file=NAIVE_COST_CSV,
                                 tsp_instances=TSP_INSTANCES,
                                 tsp_instances_folder=TSP_INSTANCES_FOLDER,
                                 tsp_opts_folder=TSP_OPTS_FOLDER)

    handler.save_opt_costs(verbose=True)
