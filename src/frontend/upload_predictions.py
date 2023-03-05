# ==============================================================================
# File: upload_predictions.py
# Project: allison
# File Created: Tuesday, 28th February 2023 7:22:56 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 7:22:57 am
# Modified By: Dillon Koch
# -----
#
# -----
# uploading predictions from /data/predictions to MongoDB frontend
# ==============================================================================


import datetime
import sys
import time
from os.path import abspath, dirname

import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm
import numpy as np

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Upload_Predictions:
    def __init__(self, league):
        self.league = league

    def _load_collection(self):  # Specific Helper upload_preds
        cluster = MongoClient("mongodb+srv://DillonKoch:QBface14$@personal-website-cluste.zuunk.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
        db = cluster["SportsBetting"]
        collection = db["Predictions"]
        return collection

    def _clean_row_dict(self, row_dict):  # Specific Helper upload_preds
        row_dict['_id'] = f"{self.league} {row_dict['Date']} {row_dict['Home']} {row_dict['Away']} {row_dict['Bet_Type']} {row_dict['Bet_Value']}"
        row_dict['Date'] = row_dict['Date'].strftime("%Y-%m-%d")
        row_dict['Odds'] = row_dict['Bet_Value']
        row_dict['ML'] = row_dict['Bet_ML']
        row_dict['Outcome'] = "" if np.isnan(row_dict['Outcome']) else "Push" if row_dict['Outcome'] == 0.5 else "Win" if row_dict['Outcome'] > 0.5 else "Loss"

        if row_dict['Prediction'] > 0.5:
            row_dict['Bet'] = row_dict['Home'] if row_dict['Bet_Type'] == "Spread" else "Over"
        else:
            row_dict['Bet'] = row_dict['Away'] if row_dict['Bet_Type'] == "Spread" else "Under"

        row_dict['Confidence'] = row_dict['Prediction'] if row_dict['Prediction'] > 0.5 else (1 - row_dict['Prediction'])
        row_dict['Confidence'] *= 100
        row_dict['League'] = self.league

        keep = set(['_id', 'Date', 'Home', 'Away', 'Bet_Type', 'Bet_Value', 'Bet_ML', 'Prediction', 'Outcome',
                    'Pred_ts', 'Odds', 'ML', 'Bet', 'Confidence', 'League'])
        row_dict = {key: val for key, val in row_dict.items() if key in keep}
        return row_dict

    def upload_preds(self, preds):  # Top Level
        row_dicts = preds.to_dict('records')
        collection = self._load_collection()
        for row_dict in tqdm(row_dicts):
            row_dict = self._clean_row_dict(row_dict)
            try:
                collection.insert_one(row_dict)
                print('insert')
            except BaseException as e:
                print(e)
                collection.replace_one({"_id": row_dict['_id']}, row_dict)
                time.sleep(3)
                print('replace')

    def run(self):  # Run
        pred_df = pd.read_csv(ROOT_PATH + f"/data/predictions/{self.league}_predictions.csv")
        pred_df['Date'] = pd.to_datetime(pred_df['Date'])
        recent_preds = pred_df.loc[pred_df['Date'] > datetime.datetime.today() - datetime.timedelta(days=7)]
        self.upload_preds(recent_preds)


if __name__ == '__main__':
    league = "NBA"
    x = Upload_Predictions(league)
    self = x
    x.run()
