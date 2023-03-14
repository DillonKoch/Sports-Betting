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

import datetime
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
        self.pred_df_path = ROOT_PATH + f'/data/predictions/{self.league}_predictions.csv'

    def espn_scores(self, espn_df, date, home, away):  # Top Level
        home_score = None
        away_score = None

        row = espn_df.loc[(espn_df['Date'] == date) & (((espn_df['Home'] == home) & (espn_df['Away'] == away)) | ((espn_df['Home'] == away) & (espn_df['Away'] == home)))]
        if len(row) and datetime.datetime.today() - datetime.timedelta(days=1) > datetime.datetime.strptime(list(row['Date'])[0], '%Y-%m-%d'):
            home_score = int(list(row['Home_Final'])[0])
            away_score = int(list(row['Away_Final'])[0])

            if list(row['Home'])[0] == away:
                home_score, away_score = away_score, home_score

        return home_score, away_score

    def run(self):  # Run
        pred_df = pd.read_csv(self.pred_df_path)
        espn_df = pd.read_csv(ROOT_PATH + f'/data/external/espn/{self.league}/Games.csv')

        for i in range(len(pred_df)):
            row = pred_df.iloc[i, :]
            outcome = None

            if np.isnan(row['Outcome']):
                date = row['Date']
                home = row['Home']
                away = row['Away']

                home_score, away_score = self.espn_scores(espn_df, date, home, away)
                if (not home_score) or (not away_score):
                    continue

                if row['Bet_Type'] == "Spread":
                    outcome = None
                    home_bet_total = home_score + row['Bet_Value']
                    if home_bet_total > away_score:
                        home_won = 1
                    elif home_bet_total == away_score:
                        outcome = 0.5
                    else:
                        home_won = 0

                    if not outcome:
                        outcome = 1 if ((home_won and row['Prediction'] == 1) or ((not home_won) and row['Prediction'] == 0)) else 0

                elif row['Bet_Type'] == 'Total':
                    outcome = None
                    real_total = home_score + away_score
                    betting_total = row['Bet_Value']

                    if real_total > betting_total:
                        over_hit = 1
                    elif real_total == betting_total:
                        outcome = 0.5
                    else:
                        over_hit = 0

                    if not outcome:
                        outcome = 1 if ((over_hit and row['Prediction'] == 1) or ((not over_hit) and row['Prediction'] == 0)) else 0

                row['Outcome'] = outcome
                pred_df.iloc[i, :] = row

        pred_df.to_csv(self.pred_df_path, index=False)


if __name__ == '__main__':
    league = "NBA"
    x = Label_Predictions(league)
    self = x
    x.run()
