# ==============================================================================
# File: sbro.py
# Project: allison
# File Created: Tuesday, 22nd November 2022 1:28:58 pm
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 22nd November 2022 1:28:59 pm
# Modified By: Dillon Koch
# -----
#
# -----
#
# ==============================================================================


import datetime
import os
import re
import string
import sys
import warnings
from os.path import abspath, dirname

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.utilities.misc import leagues
from src.utilities.match_team import Match_Team
from src.utilities.dataset_validator import Dataset_Validator


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class Odds:
    def __init__(self, league):
        self.league = league
        self.match_team = Match_Team(league)
        self.dataset_validator = Dataset_Validator()

        self.df_cols = ['Season', 'Date', 'Home', 'Away', 'Is_Neutral',
                        'OU_Close', 'OU_Close_ML',
                        'Home_Line_Close', 'Home_Line_Close_ML',
                        'Away_Line_Close', 'Away_Line_Close_ML',
                        'Home_ML', 'Away_ML']

    def load_xlsx_paths(self):  # Top Level
        """
        loading full paths to .xlsx files for a given league
        """
        folder = ROOT_PATH + f"/data/raw/{self.league}/"
        paths = sorted(listdir_fullpath(folder))
        paths = [path for path in paths if "~lock" not in path and path.endswith('.xlsx')]
        return paths

    def _clean_dates(self, df):  # Specific Helper clean_odds_df
        """
        manually cleaning dates that are not accurate in raw data and returning df
        """
        for i, date in enumerate(list(df['Date'])):
            # * manual correction - date was "7" for 2011-12 NCAAB Kentucky game row 7501
            if date == 7 and df['Rot'][i] == 874 and df['Team'][i] == 'Kentucky':
                df['Date'][i] = 323

            # * manual correctino - date was "1192" for 2012-13 NCAAF WashingtonU game row 1112
            if date == 1192 and df['Rot'][i] == 309 and df['Team'][i] == 'WashingtonU':
                df['Date'][i] = 1102

            # * manual correction - date was "130" for 2010-11 NBA Clippers game row 1397
            if date == 130 and df['Rot'][i] == 516 and df['Team'][i] == 'LAClippers':
                df['Date'][i] = 129

            # * manual correction - date was "1125" for 2018-19 NCAAB Depaul game row 1397
            if date == 1125 and df['Rot'][i] == 715 and df['Team'][i] == 'Depaul':
                df['Date'][i] = 1124

        return df

    def _clean_quarter_half_final(self, df):  # Specific Helper clean_odds_df
        """
        cleaning quarter/half/final values to int's and 0-200 or nan
        """
        for col in ['1st', '2nd', '3rd', '4th', 'Final']:
            if col in list(df.columns):
                df[col] = self.dataset_validator.set_to_nan(df[col], valid_types=(int, float, np.number), min_val=0, max_val=200)
        return df

    def _clean_bet_vals(self, df):  # Specific Helper clean_odds_df
        """
        cleaning bet values by running replacements, splitting on values, and converting to floats
        """
        replace_val_combos = [(',', '.'), ('½', '.5'), ('..', '.'), ('ev', '')]
        split_val_combos = [('u', True), ('o', True), ('+', True)]
        replace_equals = [('p', 'pk', True), ("PK", "pk", False)]
        for col in ['Open', 'Close', 'ML', '2H']:
            df[col] = self.dataset_validator.float_if_possible(df[col])

            for (replace_val, new_val) in replace_val_combos:
                df[col] = self.dataset_validator.replace_vals(df[col], replace_val, new_val)

            df[col] = self.dataset_validator.split_vals(df[col], '-', True, start=1)  # 6-209
            for (split_str, keep_before) in split_val_combos:
                df[col] = self.dataset_validator.split_vals(df[col], split_str, keep_before)

            for (replace_val, new_val, set_lower) in replace_equals:
                df[col] = self.dataset_validator.replace_vals_equal(df[col], replace_val, new_val, set_lower=set_lower)

            vals = df[col]
            for i, val in enumerate(vals):
                if len(str(val)) > 0 and str(val)[0] == '+':  # fixing "+145\xao"
                    val = float(''.join(char for char in val if char in string.digits))

            df[col] = self.dataset_validator.float_if_possible(df[col])

        return df

    def _clean_moneyline(self, df):  # Specific Helper clean_odds_df
        """
        setting NULL/invlaid values to np.nan
        """
        for i, ml in enumerate(list(df['ML'])):
            if ml == "NL":
                df['ML'][i] = np.nan

            elif not isinstance(ml, (int, float)):
                df['ML'][i] = np.nan

            elif not ((float(ml) <= -100) or (float(ml) >= 100)) and 'nan' not in str(ml):
                df['ML'][i] = np.nan
                print(f"ML {ml} -> np.nan")

        return df

    def _clean_no_lines(self, df):  # Specific Helper clean_odds_df
        """
        NL's to np.nan
        """
        df["ML"] = self.dataset_validator.set_to_nan(df["ML"], invalid_vals=['NL'])
        df["Open"] = self.dataset_validator.set_to_nan(df["Open"], invalid_vals=['NL'])
        df["Close"] = self.dataset_validator.set_to_nan(df["Close"], invalid_vals=['NL'])
        df["2H"] = self.dataset_validator.set_to_nan(df["2H"], invalid_vals=['NL'])
        return df

    def clean_odds_df(self, df):  # Top Level
        """
        cleaning the raw odds df from SBRO as best as possible
        """
        df = self._clean_dates(df)
        df = self._clean_quarter_half_final(df)
        df = self._clean_bet_vals(df)
        df = self._clean_moneyline(df)
        df = self._clean_no_lines(df)

        # * moneyline "pk" to 100, others are edited later looking at both rows together
        df["ML"] = df["ML"].replace("pk", 100)

        return df

    def _dates_qa(self, df):  # Specific Helper odds_df_quality_assurance
        """
        checking dates are ints, 3-4 characters, chars are valid months/days
        """
        for date in list(df['Date']):
            assert isinstance(date, int), f"{date} is not an int"
            assert len(str(date)) in [3, 4], f"length of date {date} is not valid"
            if len(str(date)) == 4:
                assert str(date)[:2] in ['10', '11', '12'], f"{date} is not valid date"
            else:
                assert str(date)[0] in '123456789', f"{date} is not valid date"

            assert 1 <= int(str(date)[-2:]) <= 31, f"{date} day of month is not valid"

    def _vhn_qa(self, df):  # Specific Helper odds_df_quality_assurance
        """
        'VH' column must have "V", "H", "N" (neutral) or be np.nan
        """
        for vhn in list(df['VH']):
            assert (isinstance(vhn, str) and vhn in 'VHN') or np.isnan(vhn), f"{vhn} is not a string for VHN"

    def _team_names_qa(self, df):  # Specific Helper odds_df_quality_quality_assurance
        """
        asserting that every team name is one of the valid ones in /Data/Teams or is np.nan
        """
        for team in list(df['Team']):
            assert team in self.match_team.valid_teams or np.isnan(team), f"{team} not in valid teams!"

    def _quarter_half_final_qa(self, df):  # Specific Helper odds_df_quality_assurance
        """
        asserting quarter/half/final values are int/float, and in the range 0-200
        """
        for col in ['1st', '2nd', '3rd', '4th', 'Final']:
            if col in df.columns:
                vals = list(df[col])
                for val in vals:
                    assert isinstance(val, (int, float)), f"{val} is not an integer for col {col}"
                    assert np.isnan(val) or 0 <= val <= 200

    def _odds_qa(self, df):  # Specific Helper odds_df_quality_assurance
        """
        asserting values are int/float/"pk" and that moneylines are int/float/np.nan and abs(ml) >= 100
        """
        for col in ['Open', 'Close', '2H']:
            vals = df[col]
            for val in vals:
                assert isinstance(val, (int, float)) or val == "pk", f"{val} is not a number for col {col}"

        for ml in df['ML']:
            assert isinstance(ml, (int, float)), f"{ml} is not a number for ML"
            assert abs(ml) >= 100 or np.isnan(ml), f"{ml} is not a valid moneyline"

    def odds_df_quality_assurance(self, df):  # QA Testing
        """
        I'm supposed to check every df, and of course the one year I blow it off this happens!
        """
        self._dates_qa(df)
        self._vhn_qa(df)
        self._team_names_qa(df)
        self._quarter_half_final_qa(df)
        self._odds_qa(df)

    def _odds_path_to_season(self, odds_path):  # Specific Helper  _odds_df_to_one_row
        """
        finds the season string inside the odds df path (like 2020-21)
        """
        filename = odds_path.split('/')[-1]
        season = re.findall(re.compile(r"\d{4}-\d{2}"), filename)[0]
        return season

    def _odds_df_to_row_pairs(self, odds_df):  # Specific Helper  _odds_df_to_one_row
        """
        converts the odds dataframe into a list of row pairs, since each game has 2 rows in the df
        """
        row_lists = odds_df.values.tolist()
        row_pairs = [(row_lists[i], row_lists[i + 1]) for i in range(0, len(row_lists), 2)]
        return row_pairs

    def _check_game_in_january(self, row_pair):  # Specific Helper _odds_df_to_one_row
        """
        returns True if the game is in January, else False
        - used to change the year over in the date (change as soon as we see January)
        """
        row1, _ = row_pair
        date1 = str(row1[0])
        return True if (len(date1) == 3) and (date1[0] == '1') else False

    def _home_away_row(self, row_pair):  # Global Helper
        """
        given a row pair, this method identifies the home and away row specifically
        """
        row1, row2 = row_pair
        row1_vh = row1[2]
        row2_vh = row2[2]
        # assert [row1_vh, row2_vh] in [['V', 'H'], ['H', 'V'], ['N', 'N']]
        row1_home = not row2_vh == 'H'
        home_row = row1 if row1_home else row2
        away_row = row2 if row1_home else row1
        return home_row, away_row

    def _date(self, row_pair, season, seen_january):  # Helping Helper row_pair_to_df
        """
        uses the row_pair to create a datetime object of the game date
        """
        row1, row2 = row_pair
        date1 = str(row1[0])
        date2 = str(row2[0])
        assert date1 == date2, f"Dates {date1} and {date2} are not equal"

        year = int(season[:4]) + 1 if seen_january else int(season[:4])
        month = date1[:2] if len(date1) == 4 else date1[0]
        day = date1[-2:]
        dt_ob = datetime.datetime(year, int(month), int(day))
        return dt_ob.strftime("%Y-%m-%d")

    def _home_away_neutral(self, row_pair):  # Helping Helper _row_pair_to_df
        """
        returns the home/away official name from the row pair, and whether the game is at a neutral location
        """
        home_row, away_row = self._home_away_row(row_pair)
        home = self.match_team.run(home_row[3])
        away = self.match_team.run(away_row[3])
        is_neutral = 1 if home_row[2] == 'N' else 0
        return home, away, is_neutral

    def _bet_val_to_val_ml(self, bet_val):  # Helping Helper _open_close_2h_bets
        """
        converts the bet_val from the odds df into either the val, None or the val, ML
        - ML only shows up when the bet_val is in format 7-105
        """
        bet_val = bet_val.replace('½', '.5') if isinstance(bet_val, str) else bet_val
        bet_val = abs(bet_val) if isinstance(bet_val, (int, float)) else bet_val
        try:
            bet_val = str(bet_val).lower().replace('pk', '0')
            if '-' in bet_val:
                val, ml_val = bet_val.split('-')
                ml_val = -float(ml_val)
            else:
                val = bet_val
                ml_val = -110
            return float(val), float(ml_val)
        except Exception as e:
            print(e)
            print(bet_val)

    def _open_close_2h_bets(self, row_pair, col_index):  # Helping Helper _row_pair_to_df
        """
        given a row pair and col index, this method returns the O/U, lines for open/close/2h
        """
        # * returning None's if the line is "NL"
        home_row, away_row = self._home_away_row(row_pair)
        if 'nl' in [str(home_row[col_index]).lower(), str(away_row[col_index]).lower()]:
            return None, None, None, None, None, None

        # * extracting home/away values and ML values, determining if home is over/under or not
        home_val, home_ml_val = self._bet_val_to_val_ml(home_row[col_index])
        away_val, away_ml_val = self._bet_val_to_val_ml(away_row[col_index])
        home_is_ou = home_val > away_val

        # * setting O/U, home/away lines based on data from rows
        ou = home_val if home_is_ou else away_val
        ou_ml = home_ml_val if home_is_ou else away_ml_val
        hline = away_val if home_is_ou else (-1 * home_val)
        hline_ml = (-220 - away_ml_val) if home_is_ou else home_ml_val
        aline = (-1 * away_val) if home_is_ou else home_val
        aline_ml = away_ml_val if home_is_ou else (-220 - home_ml_val)
        return ou, ou_ml, hline, hline_ml, aline, aline_ml

    def _ml_bets(self, row_pair):  # Helping Helper _row_pair_to_df
        """
        finding the ML bets in the row_pair
        """
        home_row, away_row = self._home_away_row(row_pair)
        if 'nl' in [str(home_row[-2]).lower(), str(away_row[-2]).lower()]:
            return None, None
        home_ml = float(str(home_row[-2]).lower().replace('pk', '0').replace('½', '.5'))
        away_ml = float(str(away_row[-2]).lower().replace('pk', '0').replace('½', '.5'))
        return home_ml, away_ml

    def _row_pair_to_df(self, row_pair, df, season, seen_january):  # Specific Helper odds_df_to_one_row
        """
        adding data from the row pair (one game) to the df
        """
        # date, rot, vh, team, 1st, 2nd, 3rd, 4th, final, open, close, ml, 2h
        date = self._date(row_pair, season, seen_january)
        home, away, is_neutral = self._home_away_neutral(row_pair)
        # ou_open, ou_open_ml, hline_open, hline_open_ml, aline_open, aline_open_ml = self._open_close_2h_bets(row_pair, -4)
        ou_close, ou_close_ml, hline_close, hline_close_ml, aline_close, aline_close_ml = self._open_close_2h_bets(row_pair, -3)
        # ou_2h, ou_2h_ml, hline_2h, hline_2h_ml, aline_2h, aline_2h_ml = self._open_close_2h_bets(row_pair, -1)
        home_ml, away_ml = self._ml_bets(row_pair)
        new_row = [season, date, home, away, is_neutral,
                   ou_close, ou_close_ml,
                   hline_close, hline_close_ml, aline_close, aline_close_ml,
                   home_ml, away_ml]
        new_row = [round(item, 1) if isinstance(item, float) else item for item in new_row]  # 1 decimal for SQL
        df.loc[len(df)] = new_row
        return df

    def odds_df_to_one_row(self, path, df):  # Top Level
        """
        converting the odds_df from two cols per game to just one
        """
        clean_df = pd.DataFrame(columns=self.df_cols)
        season = self._odds_path_to_season(path)
        row_pairs = self._odds_df_to_row_pairs(df)

        seen_january = False
        for row_pair in tqdm(row_pairs):
            seen_january = True if seen_january else self._check_game_in_january(row_pair)
            clean_df = self._row_pair_to_df(row_pair, clean_df, season, seen_january)

        return clean_df

    def df_quality_assurance(self, df):  # QA Testing
        pass

    def df_to_db(self, df):  # Top Level
        """
        inserting cleaned values from the df to the MySQL database
        """
        self.cursor.execute("USE sports_betting;")
        for i, row in df.iterrows():
            col_names = "(" + ', '.join(row.keys()) + ")"
            val_list = [f'"{i}"' if isinstance(i, str) or (i is not None and not np.isnan(i))
                        else "NULL" for i in row.values]
            vals = "(" + ', '.join(val_list) + ")"
            sql = f"INSERT IGNORE INTO SBRO_{self.league} {col_names} VALUES {vals};"
            self.cursor.execute(sql)
        self.db.commit()

    def run(self):  # Run
        final_df = pd.DataFrame(columns=self.df_cols)
        xlsx_paths = self.load_xlsx_paths()
        odds_dfs = [pd.read_excel(xlsx_path) for xlsx_path in tqdm(xlsx_paths)]
        for i, odds_df in enumerate(odds_dfs):
            print(xlsx_paths[i])
            odds_df = self.clean_odds_df(odds_df)
            cur_odds_df = self.odds_df_to_one_row(xlsx_paths[i], odds_df)
            final_df = pd.concat([final_df, cur_odds_df])

        # * fill in gaps with ESPN/ESB data
        # insert games not found in SBRO, including future games
        # insert odds from ESPN (if past) or ESB (if future)
        # TODO

        final_df.to_csv(ROOT_PATH + f"/data/interim/{self.league}/odds.csv", index=False)


if __name__ == '__main__':
    leagues = leagues()
    for league in leagues:
        x = Odds(league)
        self = x
        x.run()
