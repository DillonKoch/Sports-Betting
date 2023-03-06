# ==============================================================================
# File: random_forest.py
# Project: allison
# File Created: Monday, 6th March 2023 9:38:11 am
# Author: Dillon Koch
# -----
# Last Modified: Monday, 6th March 2023 9:38:12 am
# Modified By: Dillon Koch
# -----
#
# -----
# modeling bet outcomes with random forest
# ==============================================================================

import sys
from os.path import abspath, dirname

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.models.model_parent import Model_Parent


class RandomForestModel(Model_Parent):
    def __init__(self, league, bet_type, hyperparameters):
        super().__init__(league, bet_type)
        self.n_games, self.n_estimators, self.max_features, self.max_depth = hyperparameters
        print("Random Forest")
        print(self.__str__())
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_features=self.max_features, max_depth=self.max_depth)

    def __str__(self):
        return f"Random Forest, n_games={self.n_games}, n_estimators={self.n_estimators}, max_features={self.max_features}, max_depth={self.max_depth}"

    def train(self):  # Run
        train, val, test = self.load_data()
        train_X, train_y = self.separate_X_y(train, balance_classes=True)

        val_X, val_y = self.separate_X_y(val)
        self.model.fit(train_X, train_y)

        val_preds = self.model.predict(val_X)
        print(pd.Series(val_preds).value_counts())
        val_acc = accuracy_score(val_preds, val_y)
        print(val_acc)
        self.val_acc = val_acc
        return val_acc


if __name__ == '__main__':
    x = RandomForestModel()
    self = x
    x.run()
