# ==============================================================================
# File: trainer.py
# Project: allison
# File Created: Monday, 6th March 2023 8:30:17 am
# Author: Dillon Koch
# -----
# Last Modified: Monday, 6th March 2023 8:30:18 am
# Modified By: Dillon Koch
# -----
#
# -----
# facilitating the training of multiple hyperparameter sets on multiple algos
# ==============================================================================


import copy
import pickle
import random
import sys
import warnings
from operator import itemgetter
from os.path import abspath, dirname

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.models.logistic_regression import LogisticRegressionModel

warnings.filterwarnings('ignore')


class Trainer:
    def __init__(self, league, bet_type):
        self.league = league
        self.bet_type = bet_type

        self.lr_param_sets = [[n_games, ] for n_games in [3, 5, 10, 15, 25]]
        self.param_sets = {"logistic regression": self.lr_param_sets}
        self.algo_dict = {"logistic regression": LogisticRegressionModel}

    def save_model(self, model):  # Global Helper
        with open(ROOT_PATH + f"/models/{self.league}/{self.bet_type}_best.pickle", 'wb') as f:
            pickle.dump(model, f)

    def grid_search(self, algo, param_sets=None, save_best=True):  # Run
        """
        loops through all param sets, training models (of "algo" type)
        - returns all (model, val_acc, params) combos, sorted by val_acc
        """
        model = self.algo_dict[algo]
        param_sets = self.param_sets[algo] if param_sets is None else param_sets

        model_acc_params = []
        for param_set in param_sets:
            cur_model = model(self.league, self.bet_type, param_set)
            val_acc = cur_model.train()
            model_acc_params.append([cur_model, val_acc, param_set])

        model_acc_params.sort(key=itemgetter(1))

        if save_best:
            self.save_model(model_acc_params[-1][0])

        return model_acc_params

    def random_search(self, algo, n, save_best=True):  # Run
        """
        selects n random parameter sets for "algo", trains all,
        returns (model, val_acc) tuples sorted by val_acc
        """
        param_sets = self.param_sets[algo]
        param_sets = random.sample(param_sets, min(n, len(param_sets)))
        model_acc_params = self.grid_search(algo, param_sets=param_sets, save_best=save_best)
        return model_acc_params

    def neighbors(self, algo, param_set):   # Top Level
        """
        given an algorithm type and set of hyperparameters, this finds all neighbors
        """
        neighbors = []

        for i, value in enumerate(param_set):
            eligible_values = list(set([item[i] for item in self.param_sets[algo] if item[i] != value]))

            for neighbor_value in eligible_values:
                new_set = param_set.copy()
                new_set[i] = neighbor_value
                neighbors.append(new_set)

        return neighbors

    def coarse_to_fine_search(self, algo, n, m, save_best=True):  # Run
        """
        performs random search with n parameter sets,
        then grid searches neighbors of top m parameter sets from the random search
        """
        random_model_acc_params = self.random_search(algo, n, save_best=False)

        neighbors = []
        for _, _, params in random_model_acc_params:
            neighbors += self.neighbors(algo, params)
        neighbors = list(set([tuple(item) for item in neighbors]))
        neighbors = [list(item) for item in neighbors]

        self.grid_search(algo, neighbors, save_best=save_best)

    def run_all_algos(self, method, n=None, m=None):  # Run
        """
        performs one of grid/random/ctf search across all algorithms
        """
        pass


if __name__ == '__main__':
    league = "NBA"
    bet_type = "Spread"
    algo = "logistic regression"
    x = Trainer(league, bet_type)
    self = x
    # x.grid_search(algo)
    # x.random_search(algo, 3)
    x.coarse_to_fine_search(algo, 4, 2)
