# ==============================================================================
# File: logistic_regression.py
# Project: allison
# File Created: Monday, 6th March 2023 8:26:33 am
# Author: Dillon Koch
# -----
# Last Modified: Monday, 6th March 2023 8:26:34 am
# Modified By: Dillon Koch
# -----
#
# -----
# modeling bets with logistic regression
# ==============================================================================

import sys
from os.path import abspath, dirname

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.models.model_parent import Model_Parent


class LogisticRegressionModel(Model_Parent):
    def __init__(self, league, bet_type, hyperparameters):
        super().__init__(league, bet_type)
        self.n_games = hyperparameters[0]
        print(f"N games: {self.n_games}")
        self.model = LogisticRegression()

    def train(self):  # Run
        train, val, test = self.load_data()
        train_X, train_y = self.separate_X_y(train, balance_classes=True)
        val_X, val_y = self.separate_X_y(val)
        self.model.fit(train_X, train_y)

        val_preds = self.model.predict(val_X)
        val_acc = accuracy_score(val_preds, val_y)
        print(f"Validation accuracy: {round(val_acc, 3)}")
        self.val_acc = val_acc
        return val_acc


if __name__ == '__main__':
    x = LogisticRegressionModel()
    self = x
    x.run()
