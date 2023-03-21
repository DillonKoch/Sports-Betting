# ==============================================================================
# File: player_features.py
# Project: allison
# File Created: Friday, 17th March 2023 6:39:13 am
# Author: Dillon Koch
# -----
# Last Modified: Friday, 17th March 2023 6:39:14 am
# Modified By: Dillon Koch
# -----
#
# -----
# building feature vectors for players (bio, stats, injury status)
# ==============================================================================

import datetime
from operator import itemgetter
import sys
from os.path import abspath, dirname

import pandas as pd
import numpy as np

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Player_Features:
    """
    given a player ID and a game date, this creates a feature vector for the player
    """

    def __init__(self, league):
        self.league = league
        self.football_league = league in ['NFL', 'NCAAF']
        self.injury_days = 7 if self.football_league else 3

        # ! POSITION NUMBERS
        self.football_pos_nums = [(('Quarterback',), 2), (('Running Back',), 2), (('Wide Receiver',), 4),
                                  (('Tight End',), 2), (('Fullback',), 1), (('Place Kicker',), 1), (('Punter',), 1),
                                  (('Defensive Tackle', 'Defensive End', 'Nose Tackle', 'Defensive Lineman'), 5),
                                  (('Linebacker',), 4),
                                  (('Defensive Back', 'Safety', 'Free Safety', 'Strong Safety', 'Cornerback'), 5)]

        self.basketball_pos_nums = [(('Point Guard',), 2), (('Shooting Guard', 'Guard'), 2),
                                    (('Small Forward', 'Forward'), 2), (('Power Forward',), 2),
                                    (('Center',), 2)]
        self.pos_nums = self.football_pos_nums if self.football_league else self.basketball_pos_nums

        # ! STAT LISTS
        # self.football_cols = self.db_info.get_cols("ESPN_Player_Stats_NFL")
        # qb_stats = self.football_cols[6:21]
        # rb_stats = self.football_cols[16:30]
        # wr_stats = self.football_cols[21:30]
        # def_stats = self.football_cols[29:41]
        # k_stats = self.football_cols[51:58]
        # p_stats = self.football_cols[58:]
        # self.football_stat_dict = {"Quarterback": qb_stats, "Running Back": rb_stats, "Wide Receiver": wr_stats,
        #                            "Tight End": wr_stats, "Fullback": rb_stats, "Place Kicker": k_stats,
        #                            "Punter": p_stats, "Defensive Tackle": def_stats, "Linebacker": def_stats,
        #                            "Defensive Back": def_stats, "Defensive End": def_stats, "Cornerback": def_stats,
        #                            "Safety": def_stats, "Free Safety": def_stats, "Strong Safety": def_stats,
        #                            "Nose Tackle": def_stats, "Defensive Lineman": def_stats}
        self.football_stat_dict = {}
        # TODO basketball
        self.basketball_cols = ['Minutes', 'FG_Made', 'FG_Att', '3PT_Made', '3PT_Att', 'FT_Made', 'FT_Att',
                                'Offensive_Rebounds', 'Defensive_Rebounds', 'Total_Rebounds', 'Assists',
                                'Steals', 'Blocks', 'Turnovers', 'Fouls', 'Plus_Minus', 'Points']
        self.basketball_stat_dict = {position: self.basketball_cols for position in [item[0][0] for item in self.basketball_pos_nums]}
        self.stat_dict = self.football_stat_dict if self.football_league else self.basketball_stat_dict

        # * loading df's
        self.stats_df = pd.read_csv(ROOT_PATH + f"/data/external/espn/{self.league}/Player_Stats.csv")
        self.stats_df['Date'] = pd.to_datetime(self.stats_df['Date'])
        self.covers_df = pd.read_csv(ROOT_PATH + f"/data/external/covers/{self.league}/Injuries.csv")
        self.covers_df['scraped_ts'] = pd.to_datetime(self.covers_df['scraped_ts'])
        self.players_df = pd.read_csv(ROOT_PATH + f"/data/external/espn/{self.league}/Players.csv")

    def _past_game_stats(self, stats_df, game_date, team, past_games):  # Specific Helper get_player_ids
        # remove dates after game_date
        # keep 'team' only
        # list game_ids
        # keep last n game_ids
        past_df = stats_df.loc[stats_df['Date'] < game_date]
        past_df = past_df.loc[past_df['Team'] == team]
        past_df = past_df.sort_values(by='Date')

        game_ids = []
        seen = set()
        for pdf_game_id in list(past_df['Game_ID']):
            if pdf_game_id not in seen:
                game_ids.append(pdf_game_id)
                seen.add(pdf_game_id)

        last_n_ids = game_ids[-past_games:]
        past_df = past_df.loc[past_df['Game_ID'].isin(last_n_ids)]
        return past_df

    def get_player_ids(self, team, game_date, past_games, stats_df):  # Top Level
        """
        finding player IDs we'll use for the game
        - if the game was played, use the IDs that appeared
        - if the game is in the future, locate IDs from past n games
        - (incorporating injuries later)
        """
        if game_date < datetime.datetime.today() - datetime.timedelta(days=2):
            # * game is in past
            rows = stats_df.loc[(stats_df['Date'] == game_date) & (stats_df['Team'] == team)]
            player_ids = list(rows['Player_ID'])

        else:
            # * game is in the future
            past_game_stats = self._past_game_stats(stats_df, game_date, team, past_games)
            player_ids = list(set(list(past_game_stats['Player_ID'])))

        return player_ids

    def pos_ids_dict(self, player_ids, players_df):  # Top Level
        player_rows = players_df.loc[players_df['Player_ID'].isin(player_ids)]
        player_ids = list(player_rows['Player_ID'])
        positions = list(player_rows['Position'])
        d = {}
        for position, player_id in zip(positions, player_ids):
            if position in d:
                d[position].append(player_id)
            else:
                d[position] = [player_id]
        return d

    def pos_ids_from_dict(self, pos_ids_dict, positions):  # Top Level
        output = []
        for position in positions:
            output += pos_ids_dict[position] if position in pos_ids_dict else []
        return output

    def _injury_status(self, players_df, player_id, game_date, covers_df):  # Specific Helper player_stats
        """
        0=out, 1=doubtful, 2=questionable, 3=probable, 4=healthy
        """
        # locate player dash name using espn_player_id
        # find player in covers_df using dash name
        # reduce covers_df to player only
        # take most recent status if scraped within last 3 days, else healthy

        injury_dict = {"out": 0, "elig": 0, "late": 0, "mid": 0, "early": 0,
                       "doub": 1, "doubt": 1, "ques": 2, "day-to-day": 2,
                       "prob": 3}

        player_row = players_df.loc[(players_df['Player_ID'] == player_id)]
        if len(player_row) == 0:
            return 4

        name = list(player_row['Player'])[0]
        dash_name = name.lower().replace(' ', '-')

        # covers_player_df = covers_df.loc[(covers_df['Player'] == dash_name) & (game_date + datetime.timedelta(days=1) >= covers_df['scraped_ts'] >= game_date - datetime.timedelta(days=3))]
        covers_player_df = covers_df.loc[(covers_df['Player'] == dash_name) & (game_date + datetime.timedelta(days=1) >= covers_df['scraped_ts'])
                                         & (covers_df['scraped_ts'] >= game_date - datetime.timedelta(days=3))]
        if len(covers_player_df) == 0:
            return 4

        status = list(covers_player_df['Status'])[0]
        first_word = status.lower().split()[0]
        return injury_dict[first_word]

    def player_stats(self, stats_df, player_id, team, game_date, position, past_games, avg_stats, covers_df, players_df):  # Top Level
        player_df = stats_df.loc[stats_df['Player_ID'] == player_id]
        player_df = player_df.loc[player_df['Date'] < game_date]
        player_df = player_df.iloc[-past_games:, :]
        stat_vals_d = {stat: [] for stat in self.stat_dict[position]}
        for i in range(len(player_df)):
            for stat in self.stat_dict[position]:
                val = list(player_df[stat])[i]
                stat_vals_d[stat].append(val)

        output = []
        for stat in self.stat_dict[position]:
            if avg_stats:
                avg = sum(stat_vals_d[stat]) / (len(stat_vals_d[stat]) + 0.0001)
                output.append(avg)
            else:
                output += stat_vals_d[stat]

        output = [item if not np.isnan(item) else 0 for item in output]
        output.append(self._injury_status(players_df, player_id, game_date, covers_df))
        return output

    def sort_pos_stats(self, pos_stats):  # Top Level
        if self.football_league:
            pass
        else:
            pos_stats = sorted(pos_stats, key=itemgetter(0), reverse=True)
        return pos_stats

    def pad_blank_pos_stats(self, pos_stats, position, num):  # Top Level
        n_vals = len(self.stat_dict[position]) + 1  # adding 1 for injury value
        while len(pos_stats) < num:
            pos_stats.append([None] * n_vals)
        return pos_stats

    def run(self, team, game_date, past_games=5, avg_stats=True):  # Run

        output = []
        col_names = []
        player_ids = self.get_player_ids(team, game_date, past_games, self.stats_df)
        pos_ids_dict = self.pos_ids_dict(player_ids, self.players_df)

        for positions, num in self.pos_nums:
            pos_player_ids = self.pos_ids_from_dict(pos_ids_dict, positions)
            pos_stats = [self.player_stats(self.stats_df, player_id, team, game_date, positions[0], past_games, avg_stats, self.covers_df, self.players_df)
                         for player_id in pos_player_ids]
            pos_stats = self.sort_pos_stats(pos_stats)
            pos_stats = self.pad_blank_pos_stats(pos_stats, positions[0], num)
            col_names += [f'{positions[0]}{i+1}_{stat}'.replace(' ', '_') for i in range(num) for stat in list(self.stat_dict[positions[0]]) + ['Injury_Status']]
            output += [subitem for item in pos_stats[:num] for subitem in item]

        # ! nan in the output means there was a player who didn't play. None means there was no player
        return output, col_names


if __name__ == '__main__':
    league = 'NBA'
    player_id = 4395725  # herro
    game_date = '2013-03-15'
    team = 'Miami Heat'

    x = Player_Features(league)
    self = x
    x.run(team, game_date)
