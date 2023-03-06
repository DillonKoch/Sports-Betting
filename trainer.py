# ==============================================================================
# File: trainer.py
# Project: allison
# File Created: Tuesday, 28th February 2023 7:05:29 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 7:05:29 am
# Modified By: Dillon Koch
# -----
#
# -----
# using model classes to train models with hyperparameters and save the best one
# ==============================================================================


import pickle
import sys
import warnings
from os.path import abspath, dirname
import numpy as np
from operator import itemgetter

warnings.filterwarnings('ignore')

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel


class Trainer:
    def __init__(self, league, bet_type):
        self.league = league
        self.bet_type = bet_type

        self.algos = [LogisticRegressionModel, RandomForestModel]
        # * param sets
        self.lr_hyperparams = [(n_games,) for n_games in [3, 5, 10, 15, 25]]
        self.rf_hyperparams = [(n_games, n_estimators, max_features, max_depth)
                               for n_games in [3, 5, 10, 15, 25]
                               for n_estimators in [10, 50, 100]
                               for max_features in ['sqrt', 'log2', 0.5]
                               for max_depth in [None, 5, 10, 20]]

        self.param_sets = [self.lr_hyperparams, self.rf_hyperparams]

    def save_model(self, model):  # Top Level
        with open(ROOT_PATH + f"/models/{self.league}/{self.bet_type}_best.pickle", 'wb') as f:
            pickle.dump(model, f)

    def grid_search(self, param_sets=None):  # Run
        """
        searching through all hyperparameters and saving the best model
        """
        param_sets = self.param_sets[i] if param_sets is None else param_sets
        # best_model = None
        # best_acc = 0
        model_accs = []

        for i, (algo, param_set) in enumerate(zip(self.algos, param_sets)):
            for j, params in enumerate(param_set):
                print(f"Algorithm {i}/{len(self.algos)}, Hyperparameters {j}/{len(param_set)}")
                model = algo(self.league, self.bet_type, params)
                acc = model.train()
                model_accs.append((model, acc))
                # if acc > best_acc:
                #     best_model = model
                #     best_acc = acc
                print(f"Best accuracy: {max([item[1] for item in model_accs])}")

        model_accs.sort(key=itemgetter(1))
        best_model = model_accs[-1][0]
        self.save_model(best_model)
        return model_accs

    def random_search(self, n):  # Run
        """
        searching through n random sets of parameters, and saving the best model
        """
        param_sets = np.random.shuffle(self.param_sets)[:n]
        self.grid_search(param_sets)

    def neighbors(self, param_set):
        pass

    def coarse_to_fine_search(self, n, m):  # Run
        """
        searching through n random sets of hyperparameters, then neighboring sets for the top m
        """
        param_sets = np.random.shuffle(self.param_sets)[:n]
        best_m = self.grid_search(param_sets)
        neighbors = []
        self.grid_search(neighbors)


if __name__ == '__main__':
    bet_type = "Spread"
    league = "NBA"
    x = Trainer(league, bet_type)
    self = x
    x.run()
