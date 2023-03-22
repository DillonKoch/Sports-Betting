# ==============================================================================
# File: model_parent.py
# Project: allison
# File Created: Saturday, 4th March 2023 7:15:06 am
# Author: Dillon Koch
# -----
# Last Modified: Saturday, 4th March 2023 7:15:06 am
# Modified By: Dillon Koch
# -----
#
# -----
# parent class with common methods for models
# ==============================================================================

import datetime
import sys
from os.path import abspath, dirname

import pandas as pd
from imblearn.over_sampling import RandomOverSampler

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Model_Parent:
    def __init__(self, league, bet_type):
        self.league = league
        self.bet_type = bet_type
        self.non_feature_cols = set(['Home_Won', 'Home_Diff', 'Total', 'raw_Home_Line', 'raw_Home_Line_ML',
                                     'raw_Away_Line', 'raw_Away_Line_ML', 'raw_Over', 'raw_Over_ML', 'raw_Under',
                                     'raw_Under_ML', 'raw_Home_ML', 'raw_Away_ML', 'Home_Covered', 'Over_Hit'])

        # * information about the model
        self.train_ts = datetime.datetime.now()
        self.val_acc = None

    def _balance_classes(self, df, col):  # Specific Helper load_data
        ros = RandomOverSampler(random_state=18)
        df_resampled, y_resampled = ros.fit_resample(df, df[col])
        df_resampled = df_resampled.sample(frac=1)
        return df_resampled

    def load_data(self):  # Top Level
        train_path = ROOT_PATH + f"/data/processed/{self.league}/{self.n_games}games_player_stats_train.csv"
        val_path = ROOT_PATH + f"/data/processed/{self.league}/{self.n_games}games_player_stats_val.csv"
        test_path = ROOT_PATH + f"/data/processed/{self.league}/{self.n_games}games_player_stats_test.csv"
        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)
        test = pd.read_csv(test_path)
        return train, val, test

    def separate_X_y(self, df, balance_classes=False):  # Top Level
        if self.bet_type == "Spread":
            df['Home_Covered'] = ((df['Home_Diff'] + df['raw_Home_Line']) > 0).astype(int)
            if balance_classes:
                df = self._balance_classes(df, 'Home_Covered')
            y = df['Home_Covered']
        elif self.bet_type == "Total":
            df['Over_Hit'] = (df['Total'] > df['raw_Over']).astype(int)
            if balance_classes:
                df = self._balance_classes(df, 'Over_Hit')
            y = df['Over_Hit']

        X = df.loc[:, [col for col in list(df.columns) if col not in self.non_feature_cols]]
        return X, y

    def predict(self, X):  # Run
        return self.model.predict(X)


if __name__ == '__main__':
    x = Model_Parent()
    self = x
    x.run()
