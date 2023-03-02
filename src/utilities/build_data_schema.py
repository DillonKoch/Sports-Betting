# ==============================================================================
# File: build_data_schema.py
# Project: allison
# File Created: Tuesday, 28th February 2023 7:24:13 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 7:24:14 am
# Modified By: Dillon Koch
# -----
#
# -----
# building a data schema file for a given dataset
# ==============================================================================


import json
import sys
from os.path import abspath, dirname

import pandas as pd

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Build_Data_Schema:
    def __init__(self):
        pass

    def add_features(self, schema, features):  # Top Level
        d = {"type": "",
             "min": 0,
             "max": 0,
             "ranges": [],
             "whitelist": "",
             "regex": "",
             "allow_null": True}
        for feature in features:
            schema[feature] = d

        return schema

    def save_schema(self, schema, path):  # Top Level
        with open(path, 'w') as f:
            json.dump(schema, f, indent=4)

    def run(self, path, features):  # Run
        # output: json file
        schema = {}
        schema = self.add_features(schema, features)
        self.save_schema(schema, path)


if __name__ == '__main__':
    sbro = ['']
    covers = ['scraped_ts', 'Team', 'Player_ID', 'Player', 'Position', 'Status', 'Status_Date', 'Description']
    esb = ['Title', 'Date', 'Game_Time', 'Home', 'Away', 'Over', 'Over_ML', 'Under', 'Under_ML',
           'Home_Line', 'Home_Line_ML', 'Away_Line', 'Away_Line_ML', 'Home_ML', 'Away_ML', 'scraped_ts']
    espn_nba = pd.read_csv(ROOT_PATH + "/data/external/espn/NBA/Games.csv")
    espn_nba = list(espn_nba.columns)
    espn_rosters = ['Team', 'Player', 'Player_ID', 'scrape_ts']
    espn_players = ['Player_ID', 'Player', 'Team', 'Number', 'Position', 'Height', 'Weight', 'Birth_Date',
                    'Birth_Place', 'College', 'Draft_Year', 'Draft_Round', 'Draft_Pick', 'Draft_Team',
                    'Experience', 'Status', 'Team_History', 'Career_Highlights', 'scrape_ts']
    path = ROOT_PATH + "/data/external/espn/players.json"
    x = Build_Data_Schema()
    self = x
    x.run(path, espn_players)
