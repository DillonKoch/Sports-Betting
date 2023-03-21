# ==============================================================================
# File: build_features2.py
# Project: allison
# File Created: Sunday, 19th March 2023 3:56:53 pm
# Author: Dillon Koch
# -----
# Last Modified: Sunday, 19th March 2023 3:56:54 pm
# Modified By: Dillon Koch
# -----
#
# -----
# building features for ML with all data sources
# ==============================================================================


import datetime
import os
import pickle
import sys
from os.path import abspath, dirname

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from src.features.player_features import Player_Features
from src.utilities.match_team import Match_Team


class Build_Features:
    def __init__(self, league):
        self.league = league
        self.football_league = league in ['NFL', 'NCAAF']
        self.days_per_game = 7 if self.football_league else 4

        self.player_features = Player_Features(league)
        self.match_team = Match_Team(league)

    def start_date(self, n_games, restart):  # Top Level
        """
        locating the date to start calculations
        - if restarting, we compute everything since 2007
        - else, the start date is n_games * self.days_per_game
        """
        if restart:
            start_date = datetime.datetime(2000, 1, 1)
        else:
            start_date = datetime.datetime.today() - datetime.timedelta(days=n_games * self.days_per_game)
        return start_date

    def _load_espn_games(self, start_date):  # Specific Helper espn_game_avgs
        espn_games = pd.read_csv(ROOT_PATH + f"/data/external/espn/{self.league}/Games.csv")
        espn_games['Date'] = pd.to_datetime(espn_games['Date'])
        espn_games = espn_games.loc[espn_games['Date'] >= start_date]
        espn_games = espn_games.drop(['Line', 'Over_Under'], axis=1)
        espn_games = espn_games.loc[(espn_games['Date'].notnull()) & (espn_games['Home'].notnull()) & (espn_games['Away'].notnull())]
        espn_games['HOT'] = espn_games['HOT'].fillna(0)
        espn_games['AOT'] = espn_games['AOT'].fillna(0)
        # espn_games = espn_games.iloc[:201, :]  # ! REMOVE REMOVE REMOVE REMOVE REMOVE
        return espn_games

    def _empty_stat_dict(self, espn_games):  # Specific Helper espn_game_avgs
        teams = set([team for team in espn_games['Home'] if isinstance(team, str)])
        stats = list(espn_games.columns)[12:]
        d = {}
        for team in teams:
            d[team] = {}
            for stat in stats:
                for item in ['Home_', 'Away_']:
                    d[team][item + stat] = []
        return d

    def _empty_avgs_df(self, espn_games):  # Specific Helper espn_game_avgs
        target_cols = ['Home_Won', 'Home_Diff', 'Total']
        stats = list(espn_games.columns)[12:]
        avgs_cols = list(espn_games.columns)[:12] + [item + stat for item in ['Home_Home_', 'Home_Away_', 'Away_Home_', 'Away_Away_'] for stat in stats] + target_cols
        avgs_df = pd.DataFrame(columns=avgs_cols)
        return avgs_df

    def _avg(self, lis):  # Helping Helper _game_to_row
        lis = [item for item in lis if isinstance(item, (int, float)) and not np.isnan(item)]
        if not lis:
            return None

        return round(sum(lis) / len(lis), 2)

    def _game_to_row(self, game, avgs_df, stats, stat_dict, n_games):  # Helping Helper _populate_avgs_df
        new_row = pd.DataFrame([[None] * len(avgs_df.columns)], columns=avgs_df.columns)
        new_row.iloc[0, :12] = list(game.values())[:12]
        home = game['Home']
        away = game['Away']
        for stat in stats:
            # * heat home stat, heat away stat
            new_row['Home_Home_' + stat] = self._avg(stat_dict[home]['Home_' + stat])
            new_row['Home_Away_' + stat] = self._avg(stat_dict[home]['Away_' + stat])

            # * bucks home stat, bucks away stat
            new_row['Away_Home_' + stat] = self._avg(stat_dict[away]['Home_' + stat])
            new_row['Away_Away_' + stat] = self._avg(stat_dict[away]['Away_' + stat])

            stat_dict[home]['Home_' + stat].append(game[stat])
            stat_dict[away]['Away_' + stat].append(game[stat])

            if game['Date'] <= datetime.datetime.now():
                stat_dict[home]['Home_' + stat] = stat_dict[home]['Home_' + stat][-n_games:]
                stat_dict[away]['Away_' + stat] = stat_dict[away]['Away_' + stat][-n_games:]
        return new_row, stat_dict

    def _targets_to_new_row(self, game, new_row):  # Helping Helper _populate_avgs_df
        # * adding targets
        if not np.isnan(game['Home_Final']):
            new_row['Home_Won'] = 1 if game['Home_Final'] > game['Away_Final'] else 0
        new_row['Home_Diff'] = game['Home_Final'] - game['Away_Final']
        new_row['Total'] = game['Home_Final'] + game['Away_Final']
        return new_row

    def _populate_avgs_df(self, avgs_df, espn_games, stat_dict, n_games):  # Specific Helper espn_game_avgs
        stats = list(espn_games.columns)[12:]
        for game in tqdm(espn_games.to_dict('records')):
            if game['Date'] > datetime.datetime.now() + datetime.timedelta(days=7):
                continue
            new_row, stat_dict = self._game_to_row(game, avgs_df, stats, stat_dict, n_games)
            new_row = self._targets_to_new_row(game, new_row)
            avgs_df = pd.concat([avgs_df, new_row], ignore_index=True)

        return avgs_df

    def espn_game_avgs(self, n_games, start_date):  # Top Level
        espn_games = self._load_espn_games(start_date)
        stat_dict = self._empty_stat_dict(espn_games)
        avgs_df = self._empty_avgs_df(espn_games)
        avgs_df = self._populate_avgs_df(avgs_df, espn_games, stat_dict, n_games)
        return avgs_df

    def add_player_stats(self, df, n_games, last_date=None):  # Top Level

        for i, game in tqdm(enumerate(df.to_dict('records'))):
            home = game['Home']
            away = game['Away']
            date = game['Date']
            if last_date and (date < last_date):
                continue

            home_player_stats, player_cols = self.player_features.run(home, date, n_games)
            away_player_stats, player_cols = self.player_features.run(away, date, n_games)

            # * adding cols to df if needed
            if i == 0 and last_date is None:
                for home_away in ['Home', 'Away']:
                    for col_name in player_cols:
                        df[f'{home_away}_{col_name}'] = None

            all_player_stats = home_player_stats + away_player_stats
            row_vals = list(df.iloc[i, :])
            row_vals[-len(all_player_stats):] = all_player_stats
            df.iloc[i, :] = row_vals

            if i % 100 == 0:
                pstats_str = "_player_stats" if player_stats else ""
                path = ROOT_PATH + f"/data/processed/{self.league}/{n_games}games{pstats_str}_checkpoint.csv"
                df.to_csv(path, index=False)

        return df

    def _add_betting_cols(self, df):  # Specific Helper add_betting_odds
        betting_cols = ['Home_Line', 'Home_Line_ML', 'Away_Line', 'Away_Line_ML',
                        'Over', 'Over_ML', 'Under', 'Under_ML',
                        'Home_ML', 'Away_ML']
        for col in betting_cols:
            df[col] = None
        return df

    def _add_sbro_odds(self, df, sbro_row, i):  # Specific Helper add_betting_odds
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

    def _add_esb_odds(self, df, esb_row, i):  # Specific Helper add_betting_odds
        df.at[i, 'Home_Line'] = list(esb_row['Home_Line'])[-1]  # * using -1 because we could have multiple rows for a single game
        df.at[i, 'Away_Line'] = list(esb_row['Away_Line'])[-1]
        df.at[i, 'Home_Line_ML'] = list(esb_row['Home_Line_ML'])[-1]
        df.at[i, 'Away_Line_ML'] = list(esb_row['Away_Line_ML'])[-1]
        df.at[i, 'Over'] = list(esb_row['Over'])[-1]
        df.at[i, 'Over_ML'] = list(esb_row['Over_ML'])[-1]
        df.at[i, 'Under'] = list(esb_row['Under'])[-1]
        df.at[i, 'Under_ML'] = list(esb_row['Under_ML'])[-1]
        df.at[i, 'Home_ML'] = list(esb_row['Home_ML'])[-1]
        df.at[i, 'Away_ML'] = list(esb_row['Away_ML'])[-1]
        return df

    def _add_espn_odds(self, df, espn_row, i, home):  # Specific Helper add_betting_odds
        over_under = list(espn_row['Over_Under'])[0]
        line_str = list(espn_row['Line'])[0]
        if not (isinstance(over_under, (int, float)) and isinstance(line_str, str)):
            return df

        if line_str == 'EVEN':
            home_line = 0
            away_line = 0
        else:
            abbrev, line = line_str.split(' ')
            line = float(line)
            team = self.match_team.abbreviation_to_team[abbrev]
            if team == home:
                home_line = line
                away_line = line * -1
            else:
                away_line = line
                home_line = line * -1
        print(home_line, away_line, over_under)
        assert isinstance(home_line, (int, float)), f"home line {home_line} error"
        assert isinstance(away_line, (int, float)), f"away line {away_line} error"
        assert isinstance(over_under, (int, float)), f"over under {over_under} error"

        df.at[i, 'Home_Line'] = home_line
        df.at[i, 'Away_Line'] = away_line
        df.at[i, 'Home_Line_ML'] = -110
        df.at[i, 'Away_Line_ML'] = -110
        df.at[i, 'Over'] = over_under
        df.at[i, 'Over_ML'] = -110
        df.at[i, 'Under'] = over_under
        df.at[i, 'Under_ML'] = -110
        df.at[i, 'Home_ML'] = None
        df.at[i, 'Away_ML'] = None

        return df

    def add_betting_odds(self, df):  # Top Level
        df = self._add_betting_cols(df)

        sbro = pd.read_csv(ROOT_PATH + f"/data/interim/{self.league}/odds.csv")
        sbro['Date'] = pd.to_datetime(sbro['Date'])
        esb = pd.read_csv(ROOT_PATH + f"/data/external/esb/{self.league}/Game_Lines.csv")
        esb['Date'] = pd.to_datetime(esb['Date'])
        espn = pd.read_csv(ROOT_PATH + f"/data/external/espn/{self.league}/Games.csv")
        espn['Date'] = pd.to_datetime(espn['Date'])

        for i, game in tqdm(enumerate(df.to_dict('records'))):
            home = game['Home']
            away = game['Away']
            date = game['Date']

            sbro_row = sbro.loc[(sbro['Date'] == date) & (((sbro['Home'] == home) & (sbro['Away'] == away)) | ((sbro['Home'] == away) & (sbro['Away'] == home)))]
            esb_row = esb.loc[(esb['Date'] == date) & (((esb['Home'] == home) & (esb['Away'] == away)) | ((esb['Home'] == away) & (esb['Away'] == home)))]
            espn_row = espn.loc[(espn['Date'] == date) & (((espn['Home'] == home) & (espn['Away'] == away)) | ((espn['Home'] == away) & (espn['Away'] == home)))]

            if len(sbro_row) > 0:
                df = self._add_sbro_odds(df, sbro_row, i)
            elif len(esb_row) > 0:
                df = self._add_esb_odds(df, esb_row, i)
            elif len(espn_row) > 0:
                df = self._add_espn_odds(df, espn_row, i, home)

        return df

    def _encode_network(self, df):  # Specific Helper one_hot_encoding
        network_vals = ['NBA TV', 'ESPN', 'TNT', 'ABC', 'CSN', 'YES', 'TSN']
        col_vals = []
        for observed_network in list(df['Network']):
            new_vals = [0] * len(network_vals)
            for i, network_val in enumerate(network_vals):
                if isinstance(observed_network, str) and network_val in observed_network:
                    new_vals[i] = 1
            col_vals.append(new_vals)

        for i in range(len(network_vals)):
            df['Network_' + network_vals[i].replace(' ', '_')] = pd.Series([item[i] for item in col_vals])

        df = df.drop('Network', axis=1)
        return df

    def one_hot_encoding(self, df):  # Top Level
        df = self._encode_network(df)

        # * month
        month_vals = [item.month for item in list(pd.to_datetime(df['Date']))]
        df['Month'] = pd.Series(month_vals)
        df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

        # * day
        day_vals = [item.day for item in list(pd.to_datetime(df['Date']))]
        df['Day'] = pd.Series(day_vals)
        df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 12)
        df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 12)

        # * weekday
        weekday_vals = [item.weekday() for item in list(pd.to_datetime(df['Date']))]
        df['Weekday'] = pd.Series(weekday_vals)
        df['Weekday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 12)
        df['Weekday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 12)

        # * drop
        df = df.drop('Month', axis=1)
        df = df.drop('Day', axis=1)
        df = df.drop('Weekday', axis=1)

        # * year
        df['Year'] = pd.Series([item.year for item in list(pd.to_datetime(df['Date']))])

        return df

    def past_future_dfs(self, df):  # Top Level
        remove_cols = ['Game_ID', 'Season', 'Week', 'Final_Status']

        past = df[df['Date'] < datetime.datetime.today() - datetime.timedelta(days=1)]
        past = past.reset_index(drop=True)
        future = df[df['Date'] >= datetime.datetime.today() - datetime.timedelta(days=1)]
        future = future.loc[(future['Home_Line'].notnull()) | (future['Over'].notnull())]
        future = future.reset_index(drop=True)

        past = past.drop(columns=remove_cols + ['Home', "Away"])
        future = future.drop(columns=remove_cols)
        return past, future

    def partition_data(self, df, method):  # Top Level
        if method == 'raw':
            df = df.drop('Date', axis=1)
            train_df, holdout_df = train_test_split(df, test_size=0.2, random_state=18)
            val_df, test_df = train_test_split(holdout_df, test_size=0.5, random_state=18)

        # TODO partition in different ways - by year, customizable, etc
        elif method == 'val_test_recent':
            df = df.drop('Date', axis=1)
            n = int(len(df) * 0.6)
            train_old = df.iloc[:n]
            new = df.iloc[n:]
            train_new, holdout_new = train_test_split(new, test_size=0.5, random_state=18)

            train_df = pd.concat([train_old, train_new])
            val_df, test_df = train_test_split(holdout_new, test_size=0.5, random_state=18)

        return train_df, val_df, test_df

    def fill_missing_values(self, train, val, test, future):  # Top Level
        # * numeric
        train_means = train.mean()
        train = train.fillna(train_means)
        val = val.fillna(train_means)
        test = test.fillna(train_means)
        future = future.fillna(train_means)

        return train, val, test, future

    def scale_features(self, train, val, test, future):  # Top Level
        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)
        test = test.reset_index(drop=True)
        future = future.reset_index(drop=True)
        future_date_home_away, future = future.loc[:, ['Date', 'Home', 'Away']], future.drop(['Date', 'Home', 'Away'], axis=1)

        odds_cols = ['Home_Line', 'Home_Line_ML', 'Away_Line', 'Away_Line_ML',
                     'Over', 'Over_ML', 'Under', 'Under_ML', 'Home_ML', 'Away_ML']

        target_cols = ['Home_Won', 'Home_Diff', 'Total']
        X_train, y_train, odds_train = train.drop(target_cols, axis=1), train[target_cols], train[odds_cols]
        X_val, y_val, odds_val = val.drop(target_cols, axis=1), val[target_cols], val[odds_cols]
        X_test, y_test, odds_test = test.drop(target_cols, axis=1), test[target_cols], test[odds_cols]
        X_future, y_future, odds_future = future.drop(target_cols, axis=1), future[target_cols], future[odds_cols]

        # * renaming the columns of "odds_x" to include "raw_x" so there aren't duplicates
        odds_train = odds_train.rename(columns={col: f"raw_{col}" for col in odds_train.columns})
        odds_val = odds_val.rename(columns={col: f"raw_{col}" for col in odds_val.columns})
        odds_test = odds_test.rename(columns={col: f"raw_{col}" for col in odds_test.columns})
        odds_future = odds_future.rename(columns={col: f"raw_{col}" for col in odds_future.columns})

        # * scaling (checking lengths helps aviod errors when scaling empty df)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train) if len(X_train) else X_train
        X_val_scaled = scaler.transform(X_val) if len(X_val) else X_val
        X_test_scaled = scaler.transform(X_test) if len(X_test) else X_test
        X_future_scaled = scaler.transform(X_future) if len(X_future) else X_future

        train_final = pd.concat([pd.DataFrame(X_train_scaled, columns=X_train.columns), y_train, odds_train], axis=1)
        val_final = pd.concat([pd.DataFrame(X_val_scaled, columns=X_val.columns), y_val, odds_val], axis=1)
        test_final = pd.concat([pd.DataFrame(X_test_scaled, columns=X_test.columns), y_test, odds_test], axis=1)
        future_final = pd.concat([pd.DataFrame(X_future_scaled, columns=X_future.columns), y_future, odds_future], axis=1)

        future_final = pd.concat([future_date_home_away, future_final], axis=1)

        return train_final, val_final, test_final, future_final, scaler

    def merge_with_existing_df(self):  # Top Level
        pass

    def save_data(self, train, val, test, future, scaler, n_games, player_stats):  # Top Level
        pstats_str = "_player_stats" if player_stats else ""
        train.to_csv(ROOT_PATH + f"/data/processed/{self.league}/{n_games}games{pstats_str}_train.csv", index=False)
        val.to_csv(ROOT_PATH + f"/data/processed/{self.league}/{n_games}games{pstats_str}_val.csv", index=False)
        test.to_csv(ROOT_PATH + f"/data/processed/{self.league}/{n_games}games{pstats_str}_test.csv", index=False)
        future.to_csv(ROOT_PATH + f"/data/processed/{self.league}/{n_games}games{pstats_str}_future.csv", index=False)
        with open(ROOT_PATH + f"/data/processed/{self.league}/{n_games}games{pstats_str}_scaler.pickle", "wb") as f:
            pickle.dump(scaler, f)
        print("SAVED")

    # def save_data(self, train, val, test, future, scaler, n_games, player_stats):  # Top Level
    #     pstats_str = "_player_stats" if player_stats else ""
    #     train_path = ROOT_PATH + f"/data/processed/{self.league}/{n_games}games{pstats_str}_train.csv"
    #     val_path = ROOT_PATH + f"/data/processed/{self.league}/{n_games}games{pstats_str}_val.csv"
    #     test_path = ROOT_PATH + f"/data/processed/{self.league}/{n_games}games{pstats_str}_test.csv"
    #     future_path = ROOT_PATH + f"/data/processed/{self.league}/{n_games}games{pstats_str}_future.csv"

    #     dfs = [train, val, test, future]
    #     paths = [train_path, val_path, test_path, future_path]

    #     for df, path in zip(dfs, paths):
    #         if os.path.exists(path):
    #             old_df = pd.read_csv(path)
    #             df = pd.concat([old_df, df], axis=1)
    #             df.drop_duplicates(subset=['Game_ID'], keep='last')

    def load_checkpoint(self, n_games, player_stats):  # Top Level
        pstats_str = "_player_stats" if player_stats else ""
        path = ROOT_PATH + f"/data/processed/{self.league}/{n_games}games{pstats_str}_checkpoint.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'])
            last_date = list(df['Date'])[-1]
            return df, last_date
        return None, None

    def run(self, n_games, player_stats=True, partition='val_test_recent', restart=False):  # Run
        # start_date = self.start_date(n_games, restart)
        if restart:
            df = self.espn_game_avgs(n_games, datetime.datetime.date(2007, 1, 1))
            df = self.add_player_stats(df, n_games) if player_stats else df
        else:
            # df = pd.read_csv(ROOT_PATH + f"/data/processed/{self.league}/{n_games}games{pstats_str}_checkpoint.csv")
            checkpoint_df, last_date = self.load_checkpoint(n_games, player_stats)
            df = self.espn_game_avgs(n_games, last_date) if last_date is not None else self.espn_game_avgs(n_games, datetime.datetime(2007, 1, 1))
            df = pd.concat([checkpoint_df, df]) if checkpoint_df is not None else df
            df = self.add_player_stats(df, n_games, last_date) if player_stats else df
            df.drop_duplicates(subset=['Home', 'Away', 'Date'], keep='first')

        df = self.add_betting_odds(df)
        df = self.one_hot_encoding(df)

        # df = df.dropna(thresh=df.shape[1] - 50)  # * removing rows with 50+ missing vals (no stats or 2007 start games)
        df = df.loc[df['Home_Home_H1Q'].notnull()]
        df = df.reset_index(drop=True)
        # not removing cols without home_line/over # TODO do this in dataset class

        # TODO data augmentation, save that in another "augmented" file and merge in dataset

        past, future = self.past_future_dfs(df)
        train, val, test = self.partition_data(past, partition)
        train, val, test, future = self.fill_missing_values(train, val, test, future)
        train, val, test, future, scaler = self.scale_features(train, val, test, future)
        self.save_data(train, val, test, future, scaler, n_games, player_stats)


if __name__ == '__main__':
    league = 'NBA'
    n_games = 3
    player_stats = True

    x = Build_Features(league)
    self = x
    x.run(n_games, player_stats)
