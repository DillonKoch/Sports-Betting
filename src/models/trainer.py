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


from os.path import abspath, dirname
import sys
import warnings

warnings.filterwarnings('ignore')

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.models.logistic_regression import LogisticRegressionModel
from src.models.random_forest import RandomForestModel


class Trainer:
    def __init__(self, bet_type):
        self.bet_type = bet_type

        self.algos = [LogisticRegressionModel, RandomForestModel]
        #   RandomForestModel]
        # * param sets
        self.lr_hyperparams = [(n_games,) for n_games in [3, 5, 10, 15, 25]]
        self.rf_hyperparams = [(n_games, n_estimators, max_features, max_depth)
                               for n_games in [3, 5, 10, 15, 25]
                               for n_estimators in [10, 50, 100, 500, 1000]
                               for max_features in ['sqrt', 'log2', 0.5, 0.7, 1]
                               for max_depth in [None, 5, 10, 20, 30]]

        self.param_sets = [self.lr_hyperparams, self.rf_hyperparams]

    def run(self):  # Run
        best_model = None
        best_acc = 0

        for algo, param_set in zip(self.algos, self.param_sets):
            for params in param_set:
                model = algo(self.bet_type, params)
                acc = model.train()
                if acc > best_acc:
                    best_model = model
                    best_acc = acc

        print(f"Best accuracy: {best_acc}")
        # TODO save best model


if __name__ == '__main__':
    bet_type = "Spread"
    x = Trainer(bet_type)
    self = x
    x.run()
