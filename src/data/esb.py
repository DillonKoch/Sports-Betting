# ==============================================================================
# File: esb.py
# Project: allison
# File Created: Tuesday, 28th February 2023 6:44:28 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 6:44:29 am
# Modified By: Dillon Koch
# -----
#
# -----
# scraping data from Elite Sportsbook and saving to /data/external/
# ==============================================================================


import datetime
import re
import sys
import time
from os.path import abspath, dirname

import pandas as pd
from bs4 import BeautifulSoup as soup
from selenium import webdriver
from selenium.webdriver import FirefoxOptions

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.utilities.misc import leagues
from src.utilities.test_data_schema import Test_Data_Schema


def mfloat(val):
    if val is None:
        return None
    else:
        return float(val)


class ESB:
    def __init__(self, league):
        self.league = league
        link_dict = {"NFL": "https://www.elitesportsbook.com/sports/nfl-betting/game-lines-full-game.sbk",
                     "NBA": "https://www.elitesportsbook.com/sports/nba-betting/game-lines-full-game.sbk",
                     "NCAAF": "https://www.elitesportsbook.com/sports/ncaa-football-betting/game-lines-full-game.sbk",
                     "NCAAB": "https://www.elitesportsbook.com/sports/ncaa-men's-basketball-betting/game-lines-full-game.sbk"}
        self.link = link_dict[league]
        self.scrape_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.df_path = ROOT_PATH + f"/data/external/esb/{self.league}/Game_Lines.csv"
        self.test_data_schema = Test_Data_Schema(league)

    def get_sp1(self):  # Top Level
        """
        using selenium to pull up the ESB site and grab sp (sp won't show with normal get_sp1 method)
        """
        opts = FirefoxOptions()
        opts.add_argument('--headless')
        self.driver = webdriver.Firefox(executable_path=ROOT_PATH + "/geckodriver", options=opts)
        time.sleep(1)
        self.driver.get(self.link)
        html = self.driver.page_source
        sp = soup(html, 'html.parser')
        return sp

    def date_event_is_date(self, date_event):  # Top Level
        """
        returning a bool to indicate whether a date_event is a date or an event
        """
        return "col-xs-12 date" in str(date_event)

    def date(self, date_event):  # Top Level
        """
        extracting the date from a 'date' date_event
        """
        txt = date_event.get_text()
        date = datetime.datetime.strptime(txt, "%B %d, %Y")
        return date.strftime("%Y-%m-%d")

    def _game_time(self, event):  # Specific Helper event_to_db
        """
        finds the game time of an event
        """
        time = event.find_all('div', attrs={'id': 'time'})
        time = time[0].get_text()
        time_comp = re.compile(r"\d{2}:\d{2} C(S|D)T")
        match = re.search(time_comp, time)
        return match.group(0) if match is not None else None

    def _teams(self, event):  # Specific Helper event_to_db
        """
        finds the home and away teams in an event
        """
        away = event.find_all('span', attrs={'id': ['firstTeamName', 'awayTeamName']})
        away = away[0].get_text()
        home = event.find_all('span', attrs={'id': ['secondTeamName', 'homeTeamName']})
        home = home[0].get_text()
        return home, away

    def _moneylines_match(self, text):  # Helping Helper _moneylines
        """
        returns the moneyline if it matches the correct format, else None
        """
        ml_comp = re.compile(r"(((\+|-)\d+)|(even))")
        match = re.match(ml_comp, text)

        if match is None:
            print("No match for ", text)
            return None
        else:
            ml = match.group(1)
            return ml

    def _moneylines(self, event):  # Specific Helper event_to_db
        """
        finds the home/away moneylines of an event
        - the html of ESB labels the totals as moneylines and moneylines as totals
        """
        moneylines = event.find_all('div', attrs={'class': 'column total pull-right'})
        ml_texts = [item.get_text().strip() for item in moneylines]
        away_text = ml_texts[0]
        home_text = ml_texts[1]

        away_ml = self._moneylines_match(away_text)
        home_ml = self._moneylines_match(home_text)

        home_ml = '100' if home_ml == 'even' else home_ml
        away_ml = '100' if away_ml == 'even' else away_ml
        return mfloat(home_ml), mfloat(away_ml)

    def _spreads_match(self, text):  # Helping Helper _spreads
        """
        returns the spread and its moneyline if it matches the correct format, else None
        """
        spread_comp = re.compile(r"^((\+|-)?\d+\.?\d?)\((((\+|-)\d+)|(even))\)$")
        match = re.match(spread_comp, text)
        if match is None:
            print("No match for ", text)
            return None, None
        else:
            spread = match.group(1)
            spread_ml = match.group(3)
            return spread, spread_ml

    def _spreads(self, event):  # Specific Helper event_to_db
        """
        finds the home/away spread/spread_ml of an event
        """
        spreads = event.find_all('div', attrs={'class': 'column spread pull-right'})
        if len(spreads) == 0:
            return tuple([None] * 4)
        spreads_texts = [item.get_text().strip() for item in spreads]
        away_text = spreads_texts[0]
        home_text = spreads_texts[1]

        away_spread, away_spread_ml = self._spreads_match(away_text)
        home_spread, home_spread_ml = self._spreads_match(home_text)

        home_spread_ml = '100' if home_spread_ml == 'even' else home_spread_ml
        away_spread_ml = '100' if away_spread_ml == 'even' else away_spread_ml

        return mfloat(home_spread), mfloat(home_spread_ml), mfloat(away_spread), mfloat(away_spread_ml)

    def _totals_match(self, text):  # Helping Helper _totals
        """
        returns the total and its moneyline if it matches the correct format, else None
        """
        total_comp = re.compile(r"(O|U) (\d+\.?\d?)\((((\+|-)\d+)|(even))\)")
        match = re.search(total_comp, text)
        if match is None:
            if text.replace('-', '').strip() != '':
                print("No match for ", text)
            return (None, None)
        else:
            total = match.group(2)
            ml = match.group(3)
            return total, ml

    def _totals(self, event):  # Helping Helper _date_event_to_row
        """
        finds the over/under totals for an event
        the html of ESB labels the totals as moneylines and moneylines as totals
        """
        totals = event.find_all('div', attrs={'class': 'column money pull-right'})
        totals_texts = [item.get_text().strip() for item in totals]
        over_text = totals_texts[0]
        under_text = totals_texts[1]

        over, over_ml = self._totals_match(over_text)
        under, under_ml = self._totals_match(under_text)

        over_ml = '100' if over_ml == 'even' else over_ml
        under_ml = '100' if under_ml == 'even' else under_ml
        return mfloat(over), mfloat(over_ml), mfloat(under), mfloat(under_ml)

    def date_event_to_df(self, df, event, date):  # Top Level
        game_time = self._game_time(event)
        home, away = self._teams(event)
        home_ml, away_ml = self._moneylines(event)
        home_spread, home_spread_ml, away_spread, away_spread_ml = self._spreads(event)
        over, over_ml, under, under_ml = self._totals(event)
        row = [date, game_time, home, away, over, over_ml, under, under_ml, home_spread,
               home_spread_ml, away_spread, away_spread_ml, home_ml, away_ml, self.scrape_ts]
        df.loc[len(df)] = row
        return df

    def run(self):  # Run
        df = pd.read_csv(self.df_path)
        schema_path = ROOT_PATH + f"/data/external/esb/esb.json"
        self.test_data_schema.run(schema_path, df)

        sp = self.get_sp1()
        main_content = sp.find_all('div', attrs={'id': 'main-content'})[0]
        date_events = main_content.find_all('div', attrs={'class': ['col-xs-12 date', 'col-sm-12 eventbox']})
        date = None
        for date_event in date_events:
            if self.date_event_is_date(date_event):
                date = self.date(date_event)
                print(date)
            else:
                df = self.date_event_to_df(df, date_event, date)
        self.test_data_schema.run(schema_path, df)
        df.to_csv(self.df_path, index=False)


if __name__ == '__main__':
    for league in ['NBA']:
        x = ESB(league)
        self = x
        x.run()
