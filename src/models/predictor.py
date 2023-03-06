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


import datetime
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
        self.current_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.pred_df_path = ROOT_PATH + f'/data/predictions/{self.league}_predictions.csv'

    def load_pred_df(self):  # Top Level
        cols = ['Date', 'Home', 'Away', 'Bet_Type', 'Bet_Value', 'Bet_ML', 'Prediction', 'Outcome',
                'Model_Info', 'Accuracy', 'Train_ts', 'Pred_ts']
        df = pd.DataFrame(columns=cols) if not os.path.exists(self.pred_df_path) else pd.read_csv(self.pred_df_path)
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

    def save_preds(self, pred_df, future, future_info, preds, model):  # Top Level
        for i in range(len(future)):
            new_row = [future_info.loc[i, 'Date'], future_info.loc[i, 'Home'], future_info.loc[i, 'Away'],
                       self.bet_type, future.loc[i, 'raw_Home_Line'], future.loc[i, 'raw_Home_Line_ML'], preds[i],
                       None, model.__str__(), model.val_acc, model.train_ts, self.current_ts]
            pred_df.loc[len(pred_df)] = new_row

        pred_df = pred_df.drop_duplicates(subset=['Home', 'Away', 'Date', 'Bet_Type'], keep='last')
        pred_df.to_csv(self.pred_df_path, index=False)

    def run(self):  # Run
        pred_df = self.load_pred_df()
        # load best model
        model = self.load_model()
        # load future data
        future = self.load_future_data(model)
        future_drop_cols = ['Date', 'Home', 'Away', 'Home_Won', 'Home_Diff', 'Total'] + [col for col in list(future.columns) if col.startswith('raw_')]
        future_info, future_X = future.loc[:, ['Date', 'Home', 'Away']], future.drop(future_drop_cols, axis=1)
        preds = model.predict(future_X)
        self.save_preds(pred_df, future, future_info, preds, model)


if __name__ == '__main__':
    league = "NBA"
    bet_type = "Spread"
    x = Predictor(league, bet_type)
    self = x
    x.run()
