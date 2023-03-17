# ==============================================================================
# File: espn_player_stats.py
# Project: allison
# File Created: Tuesday, 28th February 2023 6:46:48 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 6:46:49 am
# Modified By: Dillon Koch
# -----
#
# -----
# scraping data about players' stats from espn.com and saving to /data/external/
# ==============================================================================


import sys
import time
import urllib.request
from os.path import abspath, dirname

import pandas as pd
from bs4 import BeautifulSoup as soup
from tqdm import tqdm

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from src.utilities.test_data_schema import Test_Data_Schema


class ESPN_Player_Stats:
    def __init__(self, league):
        self.league = league
        self.football_league = league in ['NFL', 'NCAAF']
        self.df_path = ROOT_PATH + f"/data/external/espn/{league}/Player_Stats.csv"
        self.test_data_schema = Test_Data_Schema(league)

        self.football_categories = ['Passing', 'Rushing', 'Receiving', 'Fumbles', 'Defensive',
                                    'Interceptions', 'Kick Returns', 'Punt Returns', 'Kicking', 'Punting']

        self.football_colname_idx_dict = {"Passing": {"C/ATT": [6, 7], "YDS": 8, "AVG": 9,
                                                      "TD": 10, "INT": 11, "SACKS": [12, 13], "QBR": 14, "RTG": 15},
                                          "Rushing": {"CAR": 16, "YDS": 17, "AVG": 18, "TD": 19, "LONG": 20},
                                          "Receiving": {"REC": 21, "YDS": 22, "AVG": 23, "TD": 24, "LONG": 25, "TGTS": 26},
                                          "Fumbles": {"FUM": 27, "LOST": 28, "REC": 29},
                                          "Defensive": {"TOT": 30, "SOLO": 31, "SACKS": 32, "TFL": 33, "HUR": 34, "PD": 35, "QB HTS": 36, "TD": 37},
                                          "Interceptions": {"INT": 38, "YDS": 39, "TD": 40},
                                          "Kick Returns": {"NO": 41, "YDS": 42, "AVG": 43, "LONG": 44, "TD": 45},
                                          "Punt Returns": {"NO": 46, "YDS": 47, "AVG": 48, "LONG": 49, "TD": 50},
                                          "Kicking": {"FG": [51, 52], "PCT": 53, "LONG": 54, "XP": [55, 56], "PTS": 57},
                                          "Punting": {"NO": 58, "YDS": 59, "TB": 60, "In 20": 61, "LONG": 62}}
        self.basketball_colname_idx_dict = {"MIN": 6, "FG": [7, 8], "3PT": [9, 10], "FT": [11, 12], "OREB": 13,
                                            "DREB": 14, "REB": 15, "AST": 16, "STL": 17, "BLK": 18, "TO": 19,
                                            "PF": 20, "+/-": 21, "PTS": 22}

    def query_new_game_ids(self, games_df, stats_df):  # Top Level
        """
        locating all ESPN Game ID's in the Games.csv file not in the player stats csv
        """
        all_game_ids = set(list(games_df['Game_ID']))

        existing_player_stat_game_ids = set(list(stats_df['Game_ID']))

        new_game_ids = all_game_ids - existing_player_stat_game_ids
        return list(new_game_ids)

    def query_game_infos(self, games_df, new_game_id):  # Top Level
        row = games_df.loc[games_df['Game_ID'] == new_game_id]
        date = list(row['Date'])[0]
        home = list(row['Home'])[0]
        away = list(row['Away'])[0]
        return date, home, away

    def get_stats_link(self, new_game_id):  # Top Level
        """
        creating the link to the box score for a new_game_id
        """
        ncaab_str = "mens-college-basketball"
        ncaaf_str = "college-football"
        nfl_str = "nfl"
        nba_str = "nba"
        str_dict = {"NFL": nfl_str, "NBA": nba_str, "NCAAF": ncaaf_str, "NCAAB": ncaab_str}
        stats_link = f"https://www.espn.com/{str_dict[self.league]}/boxscore/_/gameId/{new_game_id}"
        return stats_link

    def _get_sp(self, link):  # Specific Helper scrape_stats
        user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
        headers = {'User-Agent': user_agent, }
        request = urllib.request.Request(link, None, headers)  # The assembled request
        response = urllib.request.urlopen(request)
        a = response.read().decode('utf-8', 'ignore')
        sp = soup(a, 'html.parser')
        time.sleep(5)
        return sp

    def _cteam_player_ids_names(self, team):  # Helping Helper _scrape_football
        player_ids = []
        player_names = []
        player_sps = team.find_all('div', attrs={'class': 'Boxscore__Athlete'})
        for player_sp in player_sps:
            player_link = player_sp.find('a', href=True)
            if not player_link:
                continue
            player_link = player_link['href']
            player_id = int(player_link.split('id/')[1].split("/")[0])
            player_name = player_link.split('/')[-1]
            player_ids.append(player_id)
            player_names.append(player_name)
        return player_ids, player_names

    def _cteam_colnames_rows(self, cteam):  # Helping Helper scrape_football
        # colnames = cteam.find_all('thead')[1].find_all('th')
        colnames = cteam.find_all('thead')
        if len(colnames) < 2:
            return [], []
        colnames = colnames[1].find_all('th')

        colnames = [item.get_text() for item in colnames]
        body = cteam.find_all('tbody')[1]
        rows = body.find_all('tr')[:-1]  # excluding totals at bottom row
        colnames = [item for item in colnames if item not in ['TACKLES', 'MISC', '']]  # those 2 are meta-headers, some ncaaf are empty
        return colnames, rows

    def _cteam_category(self, cteam):  # Helping Helper scrape_football
        name = cteam.find('div', attrs={'class': 'TeamTitle__Name'}).get_text()
        for category in self.football_categories:
            if category in name:
                return category
        raise ValueError(f"Could not find valid category in {name}")

    def _add_stat(self, lis, colname, val, category):  # Helping Helper _update_stats
        # * didn't have avg punt yards in table - calculable with # punts and total yards anyway
        if category == 'Punting' and colname == 'AVG':
            return lis

        if val == '--':  # ! sometimes QBR is "--" for some reason
            val = "NULL"

        # * conditions for splitting a dash/slash value
        if (category == 'Passing' and colname in ['C/ATT', "SACKS"]) or (colname in ['FG', 'XP']):
            val1, val2 = val.split("-") if "-" in val else val.split("/")
            idx1, idx2 = self.football_colname_idx_dict[category][colname]
            lis[idx1] = val1
            lis[idx2] = val2
        else:
            idx = self.football_colname_idx_dict[category][colname]
            lis[idx] = val
        return lis

    def _update_stats(self, lis, row, category, table_colnames):  # Helping Helper _update_player_lists
        vals = row.find_all('td')
        vals = [item.get_text() for item in vals]
        vals = [item for item in vals if item != '']
        for colname, val in zip(table_colnames, vals):
            lis = self._add_stat(lis, colname, val, category)
        assert len(lis) == 63
        return lis

    def _update_player_lists(self, player_dicts, new_game_id, player_id, player_name, table_colnames, row, team, date, category):  # Helping Helper _scrape_football
        lis = player_dicts[player_id] if player_id in player_dicts else [None] * len(self.cols)
        lis[0] = new_game_id
        lis[1] = date
        lis[2] = team
        lis[3] = player_name
        lis[4] = player_id
        lis = self._update_stats(lis, row, category, table_colnames)
        return lis

    def _scrape_football(self, boxscore, new_game_id, home, away, date):  # Specific Helper scrape_stats
        player_lists = {}
        categories = boxscore.find_all('div', attrs={'class': 'Boxscore__Category'})
        for category in categories:
            c_teams = category.find_all('div', attrs={'class': 'Boxscore__Team'})
            for i, cteam in enumerate(c_teams):
                team = away if i == 0 else home
                player_ids, player_names = self._cteam_player_ids_names(cteam)
                table_colnames, table_rows = self._cteam_colnames_rows(cteam)
                category = self._cteam_category(cteam)
                for player_id, player_name, row in zip(player_ids, player_names, table_rows):
                    player_lists[player_id] = self._update_player_lists(player_lists, new_game_id, player_id, player_name, table_colnames, row, team, date, category)

        return player_lists

    def _add_stat_bball(self, lis, colname, val):  # Helping Helper _update_stats_bball
        idx = self.basketball_colname_idx_dict[colname]
        if len(val.replace('-', '')) == 0:
            return lis
        if isinstance(idx, list):
            idx1 = idx[0]
            idx2 = idx[1]
            val1, val2 = val.split("-")
            lis[idx1] = val1
            lis[idx2] = val2
        else:
            lis[idx] = val
        return lis

    def _update_stats_bball(self, lis, stat_row, colnames):  # Helping Helper _update_player_lists_bball
        vals = stat_row.find_all('td')
        vals = [item.get_text() for item in vals]
        if len(vals) < len(colnames):
            return lis
        for colname, val in zip(colnames, vals):
            lis = self._add_stat_bball(lis, colname, val)
        return lis

    def _update_player_lists_bball(self, player_lists, new_game_id, player_id, player_name, player_row, stat_row, team, date, colnames):  # Helping Helper _scrape_basketball
        lis = player_lists[player_id] if player_id in player_lists else [None] * len(self.cols)
        lis[0] = new_game_id
        lis[1] = date
        lis[2] = team
        lis[3] = player_name
        lis[4] = player_id
        lis = self._update_stats_bball(lis, stat_row, colnames)
        return lis

    def _scrape_basketball(self, boxscore, new_game_id, home, away, date):  # Specific Helper scrape_stats
        player_lists = {}
        team_sps = boxscore.find_all('div', attrs={'class': 'Boxscore flex flex-column'})
        for i, team_sp in enumerate(team_sps):
            team = away if i == 0 else home
            bodies = team_sp.find_all('tbody')
            # rows = body.find_all('tr')
            player_rows = bodies[0].find_all('tr')
            stat_rows = bodies[1].find_all('tr')
            for i, (player_row, stat_row) in enumerate(zip(player_rows, stat_rows)):
                if i == 0 and player_row.get_text() in ['starters', 'bench']:
                    colnames = [item.get_text() for item in stat_row.find_all('td')]
                if player_row.get_text() not in ['starters', 'bench', 'team'] and player_row.get_text():
                    player_link = player_row.find('a', href=True)['href']
                    player_id = player_link.split('/id/')[1].split("/")[0]
                    player_name = player_link.split('/')[-1]
                    player_lists[player_id] = self._update_player_lists_bball(player_lists, new_game_id, player_id, player_name, player_row, stat_row, team, date, colnames)
        return player_lists

    def scrape_stats(self, stats_link, new_game_id, date, home, away):  # Top Level
        """
        scraping stats from the boxscore to a list of dictionaries (one per player) to be inserted to DB
        """
        print(stats_link)
        sp = self._get_sp(stats_link)
        boxscore = sp.find('div', attrs={'class': 'Boxscore'})
        if boxscore is None or not boxscore.get_text():
            print("NO BOX SCORE, MOVING ON")
            return []

        elif self.football_league:
            stats = self._scrape_football(boxscore, new_game_id, home, away, date)
        else:
            stats = self._scrape_basketball(boxscore, new_game_id, home, away, date)

        return stats

    def run(self):  # Run
        stats_df = pd.read_csv(self.df_path)
        self.cols = list(stats_df.columns)
        games_df = pd.read_csv(ROOT_PATH + f"/data/external/espn/{self.league}/Games.csv")
        schema_path = ROOT_PATH + "/data/external/espn/player_stats.json"
        # self.test_data_schema.run(schema_path, df) # TODO uncomment

        new_game_ids = self.query_new_game_ids(games_df, stats_df)
        for new_game_id in tqdm(new_game_ids):
            try:
                date, home, away = self.query_game_infos(games_df, new_game_id)
                stats_link = self.get_stats_link(new_game_id)
                stat_dict = self.scrape_stats(stats_link, new_game_id, date, home, away)
                for key, val in stat_dict.items():
                    stats_df.loc[len(stats_df)] = val
                stats_df.to_csv(self.df_path, index=False)
                print("success")
                # if not stat_dict:
                #     # self.insert_blank(new_game_id, date)
                #     pass
                # else:
                #     # for stat_list in stat_dict.values():
                #     #     self.db_ops.insert_row(f"ESPN_Player_Stats_{self.league}", stat_list)
                #     pass
                # self.db.commit()
            except Exception as e:
                print(e)
                print("ERROR")


if __name__ == '__main__':
    league = 'NBA'
    x = ESPN_Player_Stats(league)
    self = x
    x.run()
