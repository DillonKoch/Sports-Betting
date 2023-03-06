# ==============================================================================
# File: random_forest.py
# Project: allison
# File Created: Tuesday, 28th February 2023 7:11:12 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 7:11:13 am
# Modified By: Dillon Koch
# -----
#
# -----
# random forest model
# ==============================================================================

import sys
from os.path import abspath, dirname

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.models.model_parent import Model_Parent


class RandomForestModel(Model_Parent):
    def __init__(self, league, bet_type, hyperparameters):
        super().__init__(league, bet_type)
        self.hyperparameters = hyperparameters
        self.n_games, self.n_estimators, self.max_features, self.max_depth = hyperparameters
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_features=self.max_features, max_depth=self.max_depth)

    def __str__(self):
        return f"Random Forest, n_games={self.n_games}, n_estimators={self.n_estimators}, max_features={self.max_features}, max_depth={self.max_depth}"

    def train(self):  # Run
        train, val, test = self.load_data()
        train_X, train_y = self.separate_X_y(train, balance_classes=True)

        val_X, val_y = self.separate_X_y(val)
        self.model.fit(train_X, train_y)

        val_preds = self.model.predict(val_X)
        val_acc = accuracy_score(val_preds, val_y)
        print(val_acc)
        self.val_acc = val_acc
        return val_acc


if __name__ == '__main__':
    league = "NBA"
    bet_type = "Spread"
    n_games = 10
    hyperparameters = [n_games]
    x = RandomForestModel(league, bet_type, hyperparameters)
    self = x
    x.train()
