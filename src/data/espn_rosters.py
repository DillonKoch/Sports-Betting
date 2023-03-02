# ==============================================================================
# File: espn_rosters.py
# Project: allison
# File Created: Tuesday, 28th February 2023 6:47:28 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 6:47:28 am
# Modified By: Dillon Koch
# -----
#
# -----
# scraping data about rosters from espn.com and saving to /data/external/
# ==============================================================================


import datetime
import json
import sys
import time
import urllib.request
from os.path import abspath, dirname

import pandas as pd
from bs4 import BeautifulSoup as soup

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.utilities.test_data_schema import Test_Data_Schema


class ESPN_Rosters:
    def __init__(self, league):
        self.league = league
        self.df_path = ROOT_PATH + f"/data/external/espn/{self.league}/Rosters.csv"
        self.test_data_schema = Test_Data_Schema(league)
        self.scraped_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    def load_roster_links(self):  # Top Level
        """
        loading links to a team's roster page on ESPN and returning team name
        """
        path = ROOT_PATH + f"/data/teams/{self.league}.json"
        with open(path, 'r') as f:
            teams_dict = json.load(f)

        # * extracting all the team roster links from the json
        teams = list(teams_dict['Teams'].keys())
        roster_links = [teams_dict['Teams'][team]['Roster'] for team in teams]
        return roster_links, teams

    def _get_sp1(self, link):  # Specific Helper  scrape_roster
        """
        scraping the HTML from a website
        """
        user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
        headers = {'User-Agent': user_agent, }
        request = urllib.request.Request(link, None, headers)  # The assembled request
        response = urllib.request.urlopen(request)
        a = response.read().decode('utf-8', 'ignore')
        sp = soup(a, 'html.parser')
        time.sleep(5)
        return sp

    def scrape_roster(self, df, roster_link, team):  # Top Level
        """
        scrapes the roster from each roster_link to a dataframe
        """
        sp = self._get_sp1(roster_link)
        table_sp = sp.find('div', attrs={'class': 'Wrapper Card__Content'})
        player_sps = table_sp.find_all('tr', attrs={'class': 'Table__TR Table__TR--lg Table__even'})
        for player_sp in player_sps:
            name_section_sp = player_sp.find_all('td', attrs={'class': 'Table__TD'})[1]
            name_sp = name_section_sp.find('a', attrs={'class': 'AnchorLink'})
            player = name_sp.get_text()
            player_link = name_sp['href']
            player_id = player_link.split('/')[-2]
            new_row = [team, player, int(player_id), self.scraped_ts]
            df.loc[len(df)] = new_row
        return df

    def run(self):  # Run
        df = pd.read_csv(self.df_path)
        schema_path = ROOT_PATH + f"/data/external/espn/rosters.json"
        self.test_data_schema.run(schema_path, df)

        roster_links, teams = self.load_roster_links()
        for i, (roster_link, team) in enumerate(zip(roster_links, teams)):
            print(f"{i} - {team} - {roster_link}")
            # try:
            df = self.scrape_roster(df, roster_link, team)
            self.test_data_schema.run(schema_path, df)
            df.to_csv(self.df_path, index=False)
            # except Exception as e:
            #     print(f"{roster_link} failed with {e}")
            #     time.sleep(10)


if __name__ == '__main__':
    for league in ['NBA']:
        x = ESPN_Rosters(league)
        self = x
        x.run()
