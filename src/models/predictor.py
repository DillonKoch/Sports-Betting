# ==============================================================================
# File: predictor.py
# Project: allison
# File Created: Tuesday, 28th February 2023 7:20:06 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 7:20:07 am
# Modified By: Dillon Koch
# -----
#
# -----
# loading best saved models and making predictions on games
# predictions saved in /data/predictions
# ==============================================================================


import os
import pickle
import sys
from os.path import abspath, dirname

import pandas as pd

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Predictor:
    def __init__(self, league, bet_type):
        self.league = league
        self.bet_type = bet_type

    def load_pred_df(self):  # Top Level
        path = ROOT_PATH + f'/data/predictions/{self.league}_predictions.csv'
        cols = ['Date', 'Home', 'Away', 'Bet_Type', 'Bet_Value', 'Bet_ML', 'Prediction', 'Outcome',
                'Model_Info', 'Accuracy', 'Train_ts', 'Pred_ts']
        df = pd.DataFrame(columns=cols) if not os.path.exists(path) else pd.read_csv(path)
        return df

    def load_model(self):  # Top Level
        path = ROOT_PATH + f"/models/{self.league}/{self.bet_type}_best.pickle"
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model

    def load_future_data(self, model):  # Top Level
        n_games = int(model.__str__().split('n_games=')[1].split(',')[0])
        path = ROOT_PATH + f"/data/processed/{self.league}/{n_games}games_future.csv"
        future = pd.read_csv(path)
        return future

    def run(self):  # Run
        pred_df = self.load_pred_df()
        # load best model
        model = self.load_model()
        # load future data
        future = self.load_future_data(model)
        print('ere')


if __name__ == '__main__':
    league = "NBA"
    bet_type = "Spread"
    x = Predictor(league, bet_type)
    self = x
    x.run()
