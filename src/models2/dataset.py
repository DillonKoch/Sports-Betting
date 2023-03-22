# ==============================================================================
# File: dataset.py
# Project: allison
# File Created: Wednesday, 22nd March 2023 10:11:07 am
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 22nd March 2023 10:11:07 am
# Modified By: Dillon Koch
# -----
#
# -----
# dataset class for training models
# ==============================================================================

import datetime
import sys
from os.path import abspath, dirname

import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class BettingDataset(Dataset):
    def __init__(self, league, bet_type, n_games, split):
        super().__init__()
        self.league = league
        self.bet_type = bet_type
        self.n_games = n_games
        self.split = split
        # TODO add feature selection and dimensionality reduction as possible hyperparameter here

        self.non_feature_cols = set(['Home_Won', 'Home_Diff', 'Total', 'raw_Home_Line', 'raw_Home_Line_ML',
                                     'raw_Away_Line', 'raw_Away_Line_ML', 'raw_Over', 'raw_Over_ML', 'raw_Under',
                                     'raw_Under_ML', 'raw_Home_ML', 'raw_Away_ML', 'Home_Covered', 'Over_Hit'])

        # load data, separate x/y, balance classes
        self.path = ROOT_PATH + f"/data/processed/{self.league}/{self.n_games}games_player_stats_{split}.csv"
        self.df = pd.read_csv(self.path)
        self.X, self.y = self.total_X_y(self.df) if bet_type == 'total' else self.spread_X_y(self.df)

        self.train_ts = datetime.datetime.now()
        self.acc = None

    def _balance_classes(self, df, col):  # Specific Helper total_X_y, spread_X_y
        ros = RandomOverSampler(random_state=18)
        df_resampled, y_resampled = ros.fit_resample(df, df[col])
        df_resampled = df_resampled.sample(frac=1)
        return df_resampled

    def total_X_y(self, df):  # Top Level __init__
        df['Over_Hit'] = (df['Total'] > df['raw_Over']).astype(int)
        df = self._balance_classes(df, 'Over_Hit') if self.split == 'train' else df
        y = df['Over_Hit']
        X = df.loc[:, [col for col in list(df.columns) if col not in self.non_feature_cols]]
        return X, y

    def spread_X_y(self, df):  # Top Level __init__
        df['Home_Covered'] = ((df['Home_Diff'] + df['raw_Home_Line']) > 0).astype(int)
        df = self._balance_classes(df, 'Home_Covered') if self.split == 'train' else df
        y = df['Home_Covered']
        X = df.loc[:, [col for col in list(df.columns) if col not in self.non_feature_cols]]
        return X, y

    def __len__(self):  # Run
        return len(self.X)

    def __getitem__(self, idx):  # Run
        return torch.tensor(self.X.iloc[idx].values).to(torch.float32), torch.tensor(self.y.iloc[idx]).to(torch.float32)


if __name__ == '__main__':
    league = 'NBA'
    bet_type = 'total'
    n_games = 3
    split = 'train'
    x = BettingDataset(league, bet_type, n_games, split)
    a, b = x.__getitem__(100)
    print('here')
