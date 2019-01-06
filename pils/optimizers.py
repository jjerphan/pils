import csv
import os
import random
import re
import string
import time
from abc import ABC, abstractmethod

import optunity
from hyperopt import tpe, hp, fmin
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from .settings import HERE, TEMP_FOLDER
from pils.settings import clean_lines


class Optimizer(ABC):
    """
    Abstract class for Optimizer.

    TODO : document

    """

    @staticmethod
    def _generate_id(id_length=10, chars=string.ascii_uppercase + string.digits):
        """
        Generate a id for binaries to avoid clashes when compiling or running.
        """
        return ''.join(random.choice(chars) for _ in range(id_length))

    def __init__(self, algo_path):
        # "/path/to/the_algo.cpp"
        self._algo_path = algo_path
        # "the_algo.cpp"
        self._algo_basename = self._algo_path.split(os.sep)[-1]
        # "the_algo"
        self._algo_name = self._algo_basename.split(".")[0]

        self._hyperparameters = {}

    def register_hyperparameter(self, **args):
        """
        Register an hyperparameter
        :return:
        """
        for k, v in args.items():
            self._hyperparameters[k] = v
        return self

    @abstractmethod
    def _get_best_hyperparameters(self, objective_function, method, number_evaluations):
        """
        Return the best hyperparameters for the given objective function using a specific method
        for a given number of evaluations.

        :param objective_function: the objective function
        :param method: the method to use (library dependent
        :param number_evaluations: the number of search to perform
        :return:
        """
        pass

    @abstractmethod
    def _get_final_score(self, hyperparameters, verbose):
        """
        Return for a given algo the final score using given hyperparameters

        :param hyperparameters: a dict of parameters
        :param verbose: to use verbose output
        :return:
        """
        pass

    def optimize(self, number_evals: int = 10, verbose: bool = True, results_file: str = None, method=None):
        """
        Optimize registered hyper-parameters for the given algorithm

        :param number_evals: the number of search to perform
        :param verbose: to have verbose outputs
        :param results_file: the file to use for results
        :param method: the method to use to optimize (library dependent)
        :return:
        """
        csv_header = ["final_score", *sorted(self._hyperparameters.keys())]

        if results_file is not None:
            if not (os.path.exists(results_file)):
                with open(results_file, "w") as file:
                    writer = csv.DictWriter(f=file, fieldnames=csv_header)
                    writer.writeheader()

        def _objective(**hyperparameters):
            """
            Objective function to optimize with `hyperopt.fmin`

            :param hyperparameters:
            :return:
            """
            if verbose:
                print("Trying {}".format(hyperparameters))

            final_score = self._get_final_score(hyperparameters, verbose)

            if verbose:
                print("Final score : {:>12} ; Parameters : {}".format(final_score, hyperparameters))
            if results_file is not None:
                with open(results_file, "a") as file:
                    writer = csv.DictWriter(f=file, fieldnames=csv_header)
                    hyperparameters["final_score"] = final_score
                    writer.writerow(hyperparameters)

            # Flipping the sign to maximize with `fmin`
            return - final_score

        best_hyperparamters = self._get_best_hyperparameters(_objective,
                                                             method=method,
                                                             number_evaluations=number_evals)

        return best_hyperparamters


class KatisOptimizerTrait(Optimizer):
    """
    An abstract class to evaluate problem on Katis directly.

    Can authenticate with email or with Google Auth.

    For email place credentials in the `.credentials` (see `.credentials.env`)
    for the structure.

    """
    _PROBLEM_SUBMIT_URL = "https://open.kattis.com/problems/{}/submit"
    _LOG_EMAIL_URL = "https://open.kattis.com/login/email"
    _LOG_URL = "https://open.kattis.com/login"

    def __init__(self, algo_path, id_problem, credential_file, use_google_auth=False):
        super().__init__(algo_path=algo_path)
        self._id_problem = id_problem
        self._login = self._login_with_google if use_google_auth else self._login_with_email
        self._credentials_file = credential_file

        # Little hack to have the driver work ; put the driver in the `HERE` folder
        os.environ["PATH"] = os.environ["PATH"] + ":" + HERE
        self._driver = webdriver.Firefox()
        self._login()

    def _get_final_score(self, hyperparameters, verbose):
        """
        Get the algo, replace macros run it online, return the final score
        """
        # Get the original code
        with open(self._algo_path, "r") as f:
            code = f.read()

        # Modify the value of the macro
        for k, v in hyperparameters.items():
            code = code.replace(str(k), str(v))

        # Saving new code in a temp file
        temp_file = os.path.join(TEMP_FOLDER, self._generate_id() + self._algo_name)
        with open(temp_file, "w") as f:
            f.write(code)

        # Submitting this code online with Katis
        final_score = self._submit_code(temp_file)
        return final_score

    def _submit_code(self, algo: str):
        """
        Submit code on Katis and return final_score

        :param algo: the .cpp file to use
        :return:
        """
        self._driver.get(self._PROBLEM_SUBMIT_URL.format(self._id_problem))

        # Uploading code
        sub_button = self._driver.find_element_by_id("sub_files_input")
        sub_button.send_keys(algo)

        # Submitting
        submit_button = self._driver.find_elements_by_name("submit")[0]
        submit_button.click()

        # Sleeping will executing remotely
        final_score = -1
        while final_score == -1:
            self._driver.refresh()
            results_elements = self._driver.find_elements_by_class_name("accepted")
            if len(results_elements) > 0:
                status = results_elements[0].text
                if "Accepted" in status or "Wrong" in status:
                    final_score = float(re.search("\d*\.\d*", status)[0])
                if "Error" in status or "Time" in status:
                    final_score = 0
                else:
                    time.sleep(5)

        return final_score

    def _get_login_and_pass_user(self):
        """
        :return: login and pass from credential file
        """
        with open(self._credentials_file, "r") as f:
            lines = clean_lines(f.readlines())

        content = dict(map(lambda line: line.split("="), lines))

        return content["LOGIN"], content["PASSWORD"]

    def _login_with_google(self):
        self._driver.get(self._LOG_URL)
        print("Please log using Google Auth")
        input("Once done Press [Enter]")

    def _login_with_email(self):
        self._driver.get(self._LOG_EMAIL_URL)
        login, password = self._get_login_and_pass_user()

        input_login = self._driver.find_element_by_id("user_input")
        input_password = self._driver.find_element_by_id("password_input")
        input_login.clear()
        input_password.clear()
        input_login.send_keys(login)
        input_password.send_keys(password)
        input_password.send_keys(Keys.RETURN)
        time.sleep(2)

# Concrete classes bellow


class KatisOptunityOptimizer(KatisOptimizerTrait):
    """
    An online hyper-parameters optimizer using the `optunity` library.

    """

    def __init__(self, algo_path, id_problem, credential_file, use_google_auth=False):
        super().__init__(algo_path=algo_path, id_problem=id_problem,
                         credential_file=credential_file, use_google_auth=use_google_auth)

    def _get_best_hyperparameters(self, objective_function, method, number_evaluations):
        if method is None:
            method = "particle swarm"

        best_hyperparameters, extra_info, solver_info = optunity.minimize(objective_function,
                                                                          solver_name=method,
                                                                          num_evals=number_evaluations,
                                                                          **self._hyperparameters)
        return best_hyperparameters


class KatisHyperOptOptimizer(KatisOptimizerTrait):
    """
    An online hyper-parameters optimizer using the `hyperopt` library.

    See https://github.com/hyperopt/hyperopt/wiki/FMin for usage

    """

    def __init__(self, algo_path, id_problem, credential_file, use_google_auth=False):
        super().__init__(algo_path=algo_path, id_problem=id_problem,
                         credential_file=credential_file, use_google_auth=use_google_auth)

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