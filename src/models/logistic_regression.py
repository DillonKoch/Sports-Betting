# ==============================================================================
# File: logistic_regression.py
# Project: allison
# File Created: Tuesday, 28th February 2023 7:10:05 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 7:10:08 am
# Modified By: Dillon Koch
# -----
#
# -----
# logistic regression model for predicting bets
# ==============================================================================


import sys
from os.path import abspath, dirname
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pandas as pd

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.models.model_parent import Model_Parent


class LogisticRegressionModel(Model_Parent):
    def __init__(self, bet_type, hyperparameters):
        super().__init__(bet_type)
        self.hyperparameters = hyperparameters
        self.n_games = hyperparameters[0]  # TODO unpack more if more hypers added
        self.model = LogisticRegression()

    def train(self):  # Run
        train, val, test = self.load_data()
        train_X, train_y = self.separate_X_y(train, balance_classes=True)
        val_X, val_y = self.separate_X_y(val)
        self.model.fit(train_X, train_y)

        val_preds = self.model.predict(val_X)
        val_acc = accuracy_score(val_preds, val_y)
        print(val_acc)
        return val_acc

    def predict(self, X):  # Run
        return self.model.predict(X)


if __name__ == '__main__':
    bet_type = "Spread"
    n_games = 10
    hyperparameters = [n_games]
    x = LogisticRegressionModel(bet_type, hyperparameters)
    self = x
    x.train()
