import os

from pils.settings import PROBLEMS_FOLDER, clean_lines

TSP_FOLDER = os.path.join(PROBLEMS_FOLDER, "tsp")
TSP_OPTS_FOLDER = os.path.join(TSP_FOLDER, "opt")
TSP_INSTANCES_FOLDER = os.path.join(TSP_FOLDER, "instances")
TEST_CASES_FILE = os.path.join(TSP_FOLDER, "to_use.txt")

# Cases to test : they are 2D Euclidian and their optimal
# solution is available
with open(TEST_CASES_FILE, "r") as f:
    lines = f.readlines()

TEST_CASES = list(filter(lambda name: '#' not in name, clean_lines(lines)))
TSP_INSTANCES = list(map(lambda name: "{}.tsp".format(name), TEST_CASES))
TSP_OPT_SOLS = list(map(lambda name: "{}.opt.tour".format(name), TEST_CASES))

# Cost stored to persist data
COST_FOLDER = os.path.join(TSP_FOLDER, "cost")
NAIVE_COST_CSV = os.path.join(COST_FOLDER, "naive_cost.csv")
OPT_COST_CSV = os.path.join(COST_FOLDER, "opt_cost.csv")