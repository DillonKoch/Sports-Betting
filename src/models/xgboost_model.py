# ==============================================================================
# File: xgboost.py
# Project: allison
# File Created: Monday, 6th March 2023 10:21:48 am
# Author: Dillon Koch
# -----
# Last Modified: Monday, 6th March 2023 10:21:49 am
# Modified By: Dillon Koch
# -----
#
# -----
# xgboost model for predicting bet outcomes
# ==============================================================================

import sys
from os.path import abspath, dirname
import pandas as pd
from sklearn.metrics import accuracy_score
import xgboost as xgb

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from src.models.model_parent import Model_Parent


class XGBoostModel(Model_Parent):
    def __init__(self, league, bet_type, hyperparameters):
        super().__init__(league, bet_type)
        self.n_games, self.lr, self.max_depth, self.n_estimators = hyperparameters
        print(self.__str__())
        self.model = xgb.XGBClassifier(learning_rate=self.lr, max_depth=self.max_depth, n_estimators=self.n_estimators, objective="binary:logistic")

    def __str__(self):  # Run
        return f"XGBoost, n_games={self.n_games}, lr={self.lr}, max_depth={self.max_depth}, n_estimators={self.n_estimators}"

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
    x = XGBoostModel()
    self = x
    x.run()
