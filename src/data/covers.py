# ==============================================================================
# File: covers.py
# Project: allison
# File Created: Tuesday, 28th February 2023 6:43:31 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 6:43:31 am
# Modified By: Dillon Koch
# -----
#
# -----
# scraping injury data from covers.com and sending to /data/external/
# ==============================================================================

import datetime
import sys
import time
import urllib.request
from os.path import abspath, dirname

import pandas as pd
from bs4 import BeautifulSoup as soup

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.utilities.misc import leagues
from src.utilities.test_data_schema import Test_Data_Schema
from src.utilities.match_team import Match_Team


class Covers:
    def __init__(self, league):
        self.league = league
        self.sport = "football" if self.league in ['NFL', 'NCAAF'] else 'basketball'
        self.link = f"https://www.covers.com/sport/{self.sport}/{self.league.lower()}/injuries"
        self.scrape_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.test_data_schema = Test_Data_Schema(league)
        self.df_path = ROOT_PATH + f"/data/external/covers/{self.league}/Injuries.csv"
        self.match_team = Match_Team(self.league)

    def get_sp1(self, link):  # Top Level
        user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
        headers = {'User-Agent': user_agent, }
        request = urllib.request.Request(link, None, headers)  # The assembled request
        response = urllib.request.urlopen(request)
        a = response.read().decode('utf-8', 'ignore')
        sp = soup(a, 'html.parser')
        time.sleep(5)
        return sp

    def get_section_sps(self, sp):  # Top Level
        """
        extracts the "section" sps from the site, each section represents a team
        """
        content_sp = sp.find('div', {'id': 'content'})
        section_sps = content_sp.find_all('section')
        return section_sps

    def section_sp_team(self, section_sp):  # Top Level
        """
        scraping the team name from the section_sp
        """
        team_sp = section_sp.find('div', attrs={'class': 'covers-CoversMatchups-teamName'})
        full_name = team_sp.get_text().strip()
        team_name = team_sp.find('span').get_text()
        full_name = full_name.replace(team_name, ' ' + team_name)
        return self.match_team.run(full_name)

    def section_sp_player_sps(self, section_sp):  # Top Level
        """
        extracts lists of the player_sps from the section_sp
        - each player has two rows, one for the player name/info and one for the injury description
        """
        table_body_sp = section_sp.find('tbody')
        row_sps = table_body_sp.find_all('tr')
        player_sps = [row_sps[i:i + 2] for i in range(0, len(row_sps), 2)]
        return player_sps

    def _split_status_date_str(self, status_date_str):  # Specific Helper
        """
        splitting the status_date_str into the status, and status_datetime in separate variables
        - the status_date didn't include the year, so I'm using the current year if the date already happened,
          otherwise the last year (so in Jan 2021, if we see a Dec injury, we know it's from Dec 2020)
        """
        # * splitting the string into status and datetime (will have year 1900 at first)
        status = status_date_str.split('(')[0]
        status_date = status_date_str.split('(')[1].split(')')[0].strip()
        status_datetime = datetime.datetime.strptime(status_date, '%a, %b %d')

        # * using current date to determine which year the injury is from
        now = datetime.datetime.now()
        status_dt_this_year = status_datetime.replace(year=now.year)
        status_dt_last_year = status_datetime.replace(year=now.year - 1)
        status_datetime = status_dt_this_year if status_dt_this_year < datetime.datetime.now() else status_dt_last_year
        return status, status_datetime

    def player_sp_info(self, player_sp):  # Top Level
        """
        extracting player info from the player_sp
        """
        row1, row2 = player_sp
        row1_tds = row1.find_all('td')
        player_link = row1_tds[0].find('a', href=True)['href']
        name = player_link.split('/')[-1]
        player_id = player_link.split('/')[-2]
        pos = row1_tds[1].get_text()
        status_date_str = row1_tds[2].get_text()
        status, status_datetime = self._split_status_date_str(status_date_str)
        description = row2.get_text().strip()
        return player_id, name, pos, status, status_datetime, description

    def run(self):  # Run
        df = pd.read_csv(self.df_path)
        schema_path = ROOT_PATH + f"/data/external/covers/covers.json"
        self.test_data_schema.run(schema_path, df)

        sp = self.get_sp1(self.link)
        section_sps = self.get_section_sps(sp)

        for section_sp in section_sps:
            team = self.section_sp_team(section_sp)
            player_sps = self.section_sp_player_sps(section_sp)

            # * going through each player on the team
            for player_sp in player_sps:
                player_id, name, pos, status, status_date, description = self.player_sp_info(player_sp)
                # row = [self.scrape_ts, team, player_id, status_date, name, pos, status, description]
                row = [self.scrape_ts, team, int(player_id), name, pos, status, status_date.strftime("%Y-%m-%d"), description]
                self.test_data_schema.run(schema_path, pd.DataFrame([row], columns=df.columns))
                df.loc[len(df)] = row
        df.to_csv(self.df_path, index=False)


if __name__ == '__main__':
    for league in ['NBA']:
        x = Covers(league)
        self = x
        x.run()
