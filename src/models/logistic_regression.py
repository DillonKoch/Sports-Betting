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

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.models.model_parent import Model_Parent


class LogisticRegressionModel(Model_Parent):
    def __init__(self, league, bet_type, hyperparameters):
        super().__init__(league, bet_type)
        self.n_games, self.feature_selection = hyperparameters
        print(self.__str__())
        self.model = LogisticRegression()

    def __str__(self):
        return f"Logistic Regression, n_games={self.n_games}, feature_selection={self.feature_selection}"

    def train(self):  # Run
        train, val, test = self.load_data()
        train_X, train_y = self.separate_X_y(train, balance_classes=True)
        val_X, val_y = self.separate_X_y(val)

        if self.feature_selection:
            self.selector = SelectKBest(mutual_info_classif, k=self.feature_selection)
            train_X = self.selector.fit_transform(train_X, train_y)
            val_X = self.selector.transform(val_X)

        # if self.dimensionality_reduction:
        #     self.pca = PCA(n_components=2)

        self.model.fit(train_X, train_y)

        val_preds = self.model.predict(val_X)
        print(pd.Series(val_preds).value_counts())
        val_acc = accuracy_score(val_preds, val_y)
        print(f"Validation accuracy: {round(val_acc, 3)}")
        self.val_acc = val_acc
        return val_acc

    def predict(self, X):  # Run
        if self.feature_selection:
            X = self.selector.transform(X)
        return self.model.predict(X)


if __name__ == '__main__':
    x = LogisticRegressionModel()
    self = x
    x.run()
