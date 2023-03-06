# ==============================================================================
# File: update_espn_abbrevs.py
# Project: allison
# File Created: Sunday, 5th March 2023 7:20:36 am
# Author: Dillon Koch
# -----
# Last Modified: Sunday, 5th March 2023 7:20:36 am
# Modified By: Dillon Koch
# -----
#
# -----
# updating the abbreviations in team JSON files using ESPN's team abbreviations
# (found in ESPN's game lines - "LAL -5.5") - adding "LAL" to "Los Angeles Lakers"
# ==============================================================================


import json
import sys
from os.path import abspath, dirname

import pandas as pd

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from src.utilities.misc import load_json


class Update_ESPN_Abbrevs:
    def __init__(self, league):
        self.league = league
        self.json_teams_path = ROOT_PATH + f"/data/teams/{self.league}.json"

    def add_abbrev_key(self, json_dict):  # Top Level
        teams = list(json_dict['Teams'].keys())
        for team in teams:
            if "Abbreviation" not in json_dict['Teams'][team]:
                json_dict['Teams'][team]["Abbreviation"] = None
        return json_dict

    def run(self):  # Run
        # add abbrev array to json dict
        # load espn df
        # access all abbrevs, use user input to assign to either home/away team using 1/2
        # update teams json, save
        json_dict = load_json(self.json_teams_path)
        json_dict = self.add_abbrev_key(json_dict)
        existing_abbrevs = set([json_dict['Teams'][team]['Abbreviation'] for team in list(json_dict['Teams'].keys())])
        espn_df = pd.read_csv(ROOT_PATH + f"/data/external/espn/{self.league}/Games.csv")

        for i in range(len(espn_df)):
            row = espn_df.iloc[i, :]
            abbrev = row['Line'].split(' ')[0] if isinstance(row['Line'], str) else None
            if abbrev and abbrev not in existing_abbrevs:
                home = row['Home']
                away = row['Away']
                print("-" * 50)
                print(abbrev)
                print(f"1: {home}")
                print(f"2: {away}")
                label = input("1 or 2: ")

                if label == '1':
                    json_dict['Teams'][home]['Abbreviation'] = abbrev
                elif label == '2':
                    json_dict['Teams'][away]['Abbreviation'] = abbrev

                existing_abbrevs.add(abbrev)

        with open(self.json_teams_path, 'w') as f:
            json.dump(json_dict, f, indent=4)


if __name__ == '__main__':
    league = "NBA"
    x = Update_ESPN_Abbrevs(league)
    self = x
    x.run()
