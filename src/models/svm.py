# ==============================================================================
# File: svm.py
# Project: allison
# File Created: Tuesday, 28th February 2023 7:11:36 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 7:11:37 am
# Modified By: Dillon Koch
# -----
#
# -----
# svm model
# ==============================================================================


import sys
from os.path import abspath, dirname

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.models.model_parent import Model_Parent


class SVMModel(Model_Parent):
    def __init__(self, league, bet_type, hyperparameters):
        super().__init__(league, bet_type)
        self.n_games, self.c, self.kernel, self.gamma, self.degree = hyperparameters
        print(self.__str__())
        self.model = SVC(C=self.c, kernel=self.kernel, gamma=self.gamma, degree=self.degree, verbose=1)

    def __str__(self):
        return f"SVM, n_games={self.n_games}, c={self.c}, kernel={self.kernel}, gamma={self.gamma}, degree={self.degree}"

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
    x = SVMModel()
    self = x
    x.run()
