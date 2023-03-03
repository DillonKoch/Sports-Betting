# ==============================================================================
# File: build_features.py
# Project: allison
# File Created: Tuesday, 28th February 2023 6:51:33 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 6:51:37 am
# Modified By: Dillon Koch
# -----
#
# -----
# using raw and interim data to build dataset in /data/processed ready for modeling
# ==============================================================================


import datetime
import sys
from os.path import abspath, dirname

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Build_Features:
    def __init__(self, league):
        self.league = league

    def _avg(self, lis):  # Specific Helper espn_game_avgs
        lis = [item for item in lis if isinstance(item, (int, float)) and not np.isnan(item)]
        if not lis:
            return None

        return round(sum(lis) / len(lis), 2)

    def espn_game_avgs(self, n_games):  # Top Level
        # load espn games df
        # create new df
        # go through all games in espn games df, storing values for each stat for each team
        # at each game, add the new value and pop the oldest (if reached n_games)
        # compute averages, add to new df
        espn_games = pd.read_csv(ROOT_PATH + f"/data/external/espn/{self.league}/Games.csv")
        espn_games = espn_games.loc[(espn_games['Date'].notnull()) & (espn_games['Home'].notnull()) & (espn_games['Away'].notnull())]
        espn_games['HOT'] = espn_games['HOT'].fillna(0)
        espn_games['AOT'] = espn_games['AOT'].fillna(0)

        teams = set([team for team in espn_games['Home'] if isinstance(team, str)])
        stats = list(espn_games.columns)[12:]
        d = {}
        for team in teams:
            d[team] = {}
            for stat in stats:
                for item in ['Home_', 'Away_']:
                    d[team][item + stat] = []

        target_cols = ['Home_Won', 'Home_Diff', 'Total']
        avgs_cols = list(espn_games.columns)[:12] + [item + stat for item in ['Home_Home_', 'Home_Away_', 'Away_Home_', 'Away_Away_'] for stat in stats] + target_cols
        avgs_df = pd.DataFrame(columns=avgs_cols)

        for game in tqdm(espn_games.to_dict('records')):
            if datetime.datetime.strptime(game['Date'], "%Y-%m-%d") > datetime.datetime.now() + datetime.timedelta(days=30):
                continue
            new_row = pd.DataFrame([[None] * len(avgs_df.columns)], columns=avgs_df.columns)
            new_row.iloc[0, :12] = list(game.values())[:12]
            home = game['Home']
            away = game['Away']
            for stat in stats:
                # * heat home stat, heat away stat
                new_row['Home_Home_' + stat] = self._avg(d[home]['Home_' + stat])
                new_row['Home_Away_' + stat] = self._avg(d[home]['Away_' + stat])

                # * bucks home stat, bucks away stat
                new_row['Away_Home_' + stat] = self._avg(d[away]['Home_' + stat])
                new_row['Away_Away_' + stat] = self._avg(d[away]['Away_' + stat])

                d[home]['Home_' + stat].append(game[stat])
                d[home]['Home_' + stat] = d[home]['Home_' + stat][-n_games:]

                d[away]['Away_' + stat].append(game[stat])
                d[away]['Away_' + stat] = d[away]['Away_' + stat][-n_games:]

            # * adding targets
            new_row['Home_Won'] = 1 if game['Home_Final'] > game['Away_Final'] else 0
            new_row['Home_Diff'] = game['Home_Final'] - game['Away_Final']
            new_row['Total'] = game['Home_Final'] + game['Away_Final']

            avgs_df = pd.concat([avgs_df, new_row], ignore_index=True)

        return avgs_df

    def add_player_stats(self, df, n_games):  # Top Level
        pass

    def add_betting_odds(self, df):  # Top Level
        betting_cols = ['Home_Line', 'Home_Line_ML', 'Away_Line', 'Away_Line_ML',
                        'Over', 'Over_ML', 'Under', 'Under_ML',
                        'Home_ML', 'Away_ML']
        for col in betting_cols:
            df[col] = None

        sbro = pd.read_csv(ROOT_PATH + f"/data/interim/{self.league}/odds.csv")

        for i, game in tqdm(enumerate(df.to_dict('records'))):
            home = game['Home']
            away = game['Away']
            date = game['Date']

            # * SBRO odds
            sbro_row = sbro.loc[(sbro['Date'] == date) & (sbro['Home'] == home) & (sbro['Away'] == away)]
            if len(sbro_row) > 0:
                df.at[i, 'Home_Line'] = list(sbro_row['Home_Line_Close'])[0]
                df.at[i, 'Away_Line'] = list(sbro_row['Away_Line_Close'])[0]
                df.at[i, 'Home_Line_ML'] = list(sbro_row['Home_Line_Close_ML'])[0]
                df.at[i, 'Away_Line_ML'] = list(sbro_row['Away_Line_Close_ML'])[0]
                df.at[i, 'Over'] = list(sbro_row['OU_Close'])[0]
                df.at[i, 'Over_ML'] = list(sbro_row['OU_Close_ML'])[0]
                df.at[i, 'Under'] = list(sbro_row['OU_Close'])[0]
                df.at[i, 'Under_ML'] = list(sbro_row['OU_Close_ML'])[0]
                df.at[i, 'Home_ML'] = list(sbro_row['Home_ML'])[0]
                df.at[i, 'Away_ML'] = list(sbro_row['Away_ML'])[0]

        return df

    def partition_data(self, df):  # Top Level
        # TODO partition in different ways - by year, customizable, etc
        train_df, holdout_df = train_test_split(df, test_size=0.2, random_state=18)
        val_df, test_df = train_test_split(holdout_df, test_size=0.5, random_state=18)
        return train_df, val_df, test_df

    def fill_missing_values(self, train, val, test):  # Top Level

        # * categorical
        for col in ['Network']:
            train[col] = train[col].fillna("Other")
            val[col] = val[col].fillna("Other")
            test[col] = test[col].fillna("Other")

        # * numeric
        train_means = train.mean()
        train = train.fillna(train_means)
        val = val.fillna(train_means)
        test = test.fillna(train_means)

        return train, val, test

    def one_hot_encoding(self, df):  # Top Level
        pass

    def scale_features(self, train, val, test):  # Top Level
        # everything is numeric at this point
        pass

    def feature_selection(self, df):  # Top Level
        pass

    def dimensionality_reduction(self, df):  # Top Level
        pass

    def run(self, n_games=3):  # Run
        # ! building the actual dataset
        # TODO data leakage with home_wins/etc
        df = self.espn_game_avgs(n_games)
        # TODO espn player stats (avgs) with player info, injury info
        # df = self.add_player_stats(df, n_games)
        df = self.add_betting_odds(df)

        # TODO make feature out of date (day 1-365)? and year
        # TODO remove irrelevant cols
        remove_cols = ['Game_ID', 'Season', 'Week', 'Home', 'Away', 'Final_Status']
        df = df.drop(columns=remove_cols)

        df.to_csv("temp.csv", index=False)
        return
        # ! Dataset is fully created, time to partition and clean/build features
        # partition train/val/test
        train, val, test = self.partition_data(df)
        # fill missing values
        train, val, test = self.fill_missing_values(train, val, test)
        # one hot encoding (network, overtime, team)
        train, val, test = self.one_hot_encoding(train, val, test)
        # scale features
        train, val, test = self.scale_features(train, val, test)
        # feature selection
        train, val, test = self.feature_selection(train, val, test)
        # dimensionality reduction
        train, val, test = self.dimensionality_reduction(train, val, test)

        # * saving
        train.to_csv(ROOT_PATH + f"/data/processed/{n_games}games_train.csv", index=False)
        val.to_csv(ROOT_PATH + f"/data/processed/{n_games}games_val.csv", index=False)
        test.to_csv(ROOT_PATH + f"/data/processed/{n_games}games_test.csv", index=False)

        # TODO balance classes while right about to train model (won't here bc multiple targets)
        # TODO keep raw betting data in and remove in data loader, but manually make scaled features of them here


if __name__ == '__main__':
    for league in ['NBA']:
        x = Build_Features(league)
        self = x
        x.run()
