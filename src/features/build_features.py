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


import concurrent.futures
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


def multithread(func, func_args):  # Multithreading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = list(tqdm(executor.map(func, func_args), total=len(func_args)))
    return result


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
        checkpoint_path = ROOT_PATH + f"/data/processed/{self.league}/{n_games}games_espn_avgs_checkpoint.csv"
        checkpoint_df = None
        if os.path.exists(checkpoint_path):
            checkpoint_df = pd.read_csv(checkpoint_path)

        espn_games = pd.read_csv(ROOT_PATH + f"/data/external/espn/{self.league}/Games.csv")
        espn_games = espn_games.drop(['Line', 'Over_Under'], axis=1)
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
        if checkpoint_df is not None:
            avgs_df = checkpoint_df
            espn_games = espn_games.iloc[len(avgs_df):, :]

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
                d[away]['Away_' + stat].append(game[stat])

                if datetime.datetime.strptime(game['Date'], "%Y-%m-%d") <= datetime.datetime.now():
                    d[home]['Home_' + stat] = d[home]['Home_' + stat][-n_games:]
                    d[away]['Away_' + stat] = d[away]['Away_' + stat][-n_games:]

            # * adding targets
            if not np.isnan(game['Home_Final']):
                new_row['Home_Won'] = 1 if game['Home_Final'] > game['Away_Final'] else 0
            new_row['Home_Diff'] = game['Home_Final'] - game['Away_Final']
            new_row['Total'] = game['Home_Final'] + game['Away_Final']

            avgs_df = pd.concat([avgs_df, new_row], ignore_index=True)

        avgs_df.to_csv(checkpoint_path, index=False)
        return avgs_df

    def add_player_stats(self, df, n_games):  # Top Level
        pass

    def add_betting_odds(self, df):  # Top Level
        # TODO add espn (caesar) odds, predict ML from spread
        betting_cols = ['Home_Line', 'Home_Line_ML', 'Away_Line', 'Away_Line_ML',
                        'Over', 'Over_ML', 'Under', 'Under_ML',
                        'Home_ML', 'Away_ML']
        for col in betting_cols:
            df[col] = None

        sbro = pd.read_csv(ROOT_PATH + f"/data/interim/{self.league}/odds.csv")
        esb = pd.read_csv(ROOT_PATH + f"/data/external/esb/{self.league}/Game_Lines.csv")

        for i, game in tqdm(enumerate(df.to_dict('records'))):
            home = game['Home']
            away = game['Away']
            date = game['Date']

            sbro_row = sbro.loc[(sbro['Date'] == date) & (((sbro['Home'] == home) & (sbro['Away'] == away)) | ((sbro['Home'] == away) & (sbro['Away'] == home)))]
            esb_row = esb.loc[(esb['Date'] == date) & (((esb['Home'] == home) & (esb['Away'] == away)) | ((esb['Home'] == away) & (esb['Away'] == home)))]

            # * SBRO odds
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
            elif len(esb_row) > 0:
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

    def feature_selection(self, df):  # Top Level
        pass

    def dimensionality_reduction(self, df):  # Top Level
        pass

    def edit_wins_losses(self, df):  # Top Level
        """
        the home/away wins/losses columns are calculated from the espn record
        (includes outcome of the game's row, so setting this back for past records)
        """
        for i in range(len(df)):
            val = df['Home_Won'][i]

            # * records from today and into the future are accurate
            if datetime.datetime.strptime(df['Date'][i], '%Y-%m-%d') >= datetime.datetime.today():
                continue

            if val:
                df.at[i, 'Home_Wins'] = df['Home_Wins'][i] - 1
                df.at[i, 'Away_Losses'] = df['Away_Losses'][i] - 1
            else:
                df.at[i, 'Home_Losses'] = df['Home_Losses'][i] - 1
                df.at[i, 'Away_Wins'] = df['Away_Wins'][i] - 1

        return df

    def run(self, n_games=3, partition='val_test_recent'):  # Run
        # ! building the actual dataset
        df = self.espn_game_avgs(n_games)
        df = self.edit_wins_losses(df)
        # TODO espn player stats (avgs) with player info, injury info
        # df = self.add_player_stats(df, n_games)
        df = self.add_betting_odds(df)

        df = df.dropna(thresh=df.shape[1] - 50)  # * removing rows with 50+ missing vals (no stats or 2007 start games)
        for col in ['Home_Line', 'Over']:
            df = df.loc[df[col].notnull()]
        df = df.reset_index(drop=True)
        df = self.one_hot_encoding(df)
        df['Date'] = pd.to_datetime(df["Date"])
        df.to_csv("temp.csv", index=False)
        future = df[df['Date'] >= datetime.datetime.today() - datetime.timedelta(days=1)]
        df = df[df['Date'] < datetime.datetime.today() - datetime.timedelta(days=1)]

        # * removing cols
        remove_cols = ['Game_ID', 'Season', 'Week', 'Final_Status']
        if self.league != "NCAAB":
            remove_cols += [col for col in list(df.columns) if '1H' in col or '2H' in col]
        df = df.drop(columns=remove_cols + ['Home', 'Away'])
        future = future.drop(columns=remove_cols)

        future.to_csv("future.csv", index=False)
        # ! Dataset is fully created, time to partition and clean/build features
        # partition train/val/test
        train, val, test = self.partition_data(df, partition)
        # fill missing values
        train, val, test, future = self.fill_missing_values(train, val, test, future)
        # scale features
        train, val, test, future, scaler = self.scale_features(train, val, test, future)
        # feature selection
        # train, val, test = self.feature_selection(train, val, test)
        # dimensionality reduction
        # train, val, test = self.dimensionality_reduction(train, val, test)

        # * saving
        train.to_csv(ROOT_PATH + f"/data/processed/{self.league}/{n_games}games_train.csv", index=False)
        val.to_csv(ROOT_PATH + f"/data/processed/{self.league}/{n_games}games_val.csv", index=False)
        test.to_csv(ROOT_PATH + f"/data/processed/{self.league}/{n_games}games_test.csv", index=False)
        future.to_csv(ROOT_PATH + f"/data/processed/{self.league}/{n_games}games_future.csv", index=False)
        with open(ROOT_PATH + f"/data/processed/{self.league}/{n_games}games_scaler.pickle", "wb") as f:
            pickle.dump(scaler, f)

        # TODO balance classes while right about to train model (won't here bc multiple targets)


if __name__ == '__main__':
    for league in ['NBA']:
        x = Build_Features(league)
        self = x
        # for n in [3]:  # , 5, 10, 15, 25]:
        #     x.run(n_games=n)
        multithread(x.run, [3, 5, 10, 15, 25])
        # x.run(n_games=3)
