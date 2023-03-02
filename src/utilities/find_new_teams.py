# ==============================================================================
# File: find_new_teams.py
# Project: allison
# File Created: Wednesday, 1st March 2023 7:04:50 am
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 1st March 2023 7:04:51 am
# Modified By: Dillon Koch
# -----
#
# -----
# finding new team names not in the official /data/teams/ json files
# ==============================================================================


import sys
from os.path import abspath, dirname

import pandas as pd
from tqdm import tqdm

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from src.utilities.misc import leagues, load_json


class Find_New_Teams:
    def __init__(self, league):
        self.league = league

    def load_covers(self):  # Top Level
        df = pd.read_csv(ROOT_PATH + f"/data/external/covers/{self.league}/Injuries.csv")
        teams = df['Team']
        return list(set(teams))

    def load_esb(self):  # Top Level
        return []

    def load_espn(self):  # Top Level
        return []

    def run(self):  # Run
        team_dict = load_json(ROOT_PATH + f"/data/teams/{self.league}.json")
        teams = self.load_covers()
        teams += self.load_esb()
        teams += self.load_espn()

        team_dict_teams = list(team_dict['Teams'].keys())
        other_names = []
        for team in team_dict_teams:
            other_names += team_dict['Teams'][team]['Other Names']
        all_names = team_dict_teams + other_names

        new_teams = set([team for team in teams if team not in all_names])
        for new_team in tqdm(new_teams):
            # TODO add the new team (do this once I find one)
            pass


if __name__ == '__main__':
    for league in leagues():
        x = Find_New_Teams(league)
        self = x
        x.run()
