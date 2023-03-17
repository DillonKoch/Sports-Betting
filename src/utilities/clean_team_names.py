# ==============================================================================
# File: clean_team_names.py
# Project: allison
# File Created: Wednesday, 1st March 2023 7:02:34 am
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 1st March 2023 7:02:34 am
# Modified By: Dillon Koch
# -----
#
# -----
# editing team names in a df to match the official names
# purpose is to do this to an older df one time, should clean new names as scraped
# ==============================================================================


import sys
from os.path import abspath, dirname

import pandas as pd

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.utilities.match_team import Match_Team
from src.utilities.misc import load_json


class Clean_Team_Names:
    def __init__(self, league):
        self.league = league
        self.match_team = Match_Team(self.league)

    def clean_vals(self, vals):  # Top Level
        new_vals = []
        for val in vals:
            if val in self.match_team.valid_teams:
                new_vals.append(val)
            else:
                valid_team = self.match_team.run(val)
                new_vals.append(valid_team)
        return new_vals

    def run(self, df_path, col_name="Team"):  # Run
        df = pd.read_csv(df_path)
        vals = list(df[col_name])
        vals = self.clean_vals(vals)
        df[col_name] = pd.Series(vals)
        df.to_csv(df_path, index=False)


if __name__ == '__main__':
    df_path = ROOT_PATH + "/data/external/espn/NBA/Player_Stats.csv"
    league = "NBA"

    x = Clean_Team_Names(league)
    self = x
    x.run(df_path, "Team")
