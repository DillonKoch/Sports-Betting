# ==============================================================================
# File: label_predictions.py
# Project: allison
# File Created: Sunday, 5th March 2023 6:59:52 am
# Author: Dillon Koch
# -----
# Last Modified: Sunday, 5th March 2023 6:59:53 am
# Modified By: Dillon Koch
# -----
#
# -----
# labeling predictions made in /data/predictions/ using game outcomes
# ==============================================================================

import sys
from os.path import abspath, dirname

import numpy as np
import pandas as pd

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Label_Predictions:
    def __init__(self, league):
        self.league = league

    def espn_scores(self, espn_df, date, home, away):  # Top Level
        home_score = None
        away_score = None

        row = espn_df.loc[(espn_df['Date'] == date) & (((espn_df['Home'] == home) & (espn_df['Away'] == away)) | ((espn_df['Home'] == away) & (espn_df['Away'] == home)))]
        if len(row):
            home_score = int(row['Home_Final'])
            away_score = int(row['Away_Final'])

            if row['Home'] == away:
                home_score, away_score = away_score, home_score

        return home_score, away_score

    def run(self):  # Run
        pred_df = pd.read_csv(ROOT_PATH + f'/data/predictions/{self.league}_predictions.csv')
        espn_df = pd.read_csv(ROOT_PATH + f'/data/external/espn/{self.league}/Games.csv')

        for i in range(len(pred_df)):
            row = pred_df.iloc[i, :]

            if np.isnan(row['Outcome']):
                date = row['Date']
                home = row['Home']
                away = row['Away']

                home_score, away_score = self.espn_scores(espn_df, date, home, away)
                print('here')


if __name__ == '__main__':
    league = "NBA"
    x = Label_Predictions(league)
    self = x
    x.run()
