# ==============================================================================
# File: match_team.py
# Project: allison
# File Created: Tuesday, 28th February 2023 7:03:51 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 7:03:52 am
# Modified By: Dillon Koch
# -----
#
# -----
# matching a scraped or downloaded team name with the "official" one from ESPN
# ==============================================================================


from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.utilities.misc import load_json


class Match_Team:
    def __init__(self, league):
        self.league = league
        self.json = load_json(ROOT_PATH + f"/data/teams/{self.league}.json")
        self.valid_teams = set(self.json['Teams'].keys())

    @property
    def team_to_official(self):  # Property
        d = {}
        for key in self.json['Teams'].keys():
            d[key] = key
            other_names = self.json['Teams'][key]["Other Names"]
            for other_name in other_names:
                d[other_name] = key
        return d

    def run(self, team):  # Run
        if team in self.team_to_official:
            return self.team_to_official[team]

        raise ValueError(f"Team {team} is not in the teams json file!")


if __name__ == '__main__':
    x = Match_Team("NCAAF")
    self = x
    x.run()
