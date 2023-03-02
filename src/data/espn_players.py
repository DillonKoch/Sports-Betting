# ==============================================================================
# File: espn_players.py
# Project: allison
# File Created: Tuesday, 28th February 2023 6:45:43 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 6:45:44 am
# Modified By: Dillon Koch
# -----
#
# -----
# scraping data about players from espn.com and saving to /data/external/
# ==============================================================================


import datetime
import re
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


class ESPN_Players:
    def __init__(self, league):
        self.league = league
        self.df_path = ROOT_PATH + f"/data/external/espn/{self.league}/Players.csv"
        self.test_data_schema = Test_Data_Schema(league)
        self.scrape_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.player_dict_keys = ['Team', 'Position', 'HT/WT', 'DOB', 'College', 'Draft Info', 'Status',
                                 'Experience', 'Birthplace', 'Number', 'Name']
        self.cols = ['Player_ID', 'Player', 'Team', 'Number', 'Position', 'Height', 'Weight',
                     'Birthdate', 'Birthplace', 'College', 'Draft_Year', 'Draft_Round',
                     'Draft_Pick', 'Draft_Team', 'Experience', 'Status', 'Team_History',
                     'Career_Highlights', 'scrape_ts']

    def unscraped_player_ids(self, df):  # Top Level
        """
        locating unscraped player ids that are shown in the roster data, but not in players.csv
        """
        scraped_ids = set(list(df['Player_ID']))
        roster_df = pd.read_csv(ROOT_PATH + f"/data/external/espn/{self.league}/Rosters.csv")
        all_ids = list(roster_df['Player_ID'])

        new_ids = [item for item in all_ids if item not in scraped_ids]
        return new_ids

    def _new_player_dict(self, player_id):  # Specific Helper scrape_player_data
        """
        blank dict to store the player data before putting it in the df
        """
        player_dict = {key: None for key in self.cols}
        player_dict['Player_ID'] = player_id
        return player_dict

    def _get_sp1(self, link):  # Specific Helper scrape_player_data
        """
        scraping HTML from a link
        """
        time.sleep(5)
        user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
        headers = {'User-Agent': user_agent, }
        request = urllib.request.Request(link, None, headers)  # The assembled request
        response = urllib.request.urlopen(request)
        a = response.read().decode('utf-8', 'ignore')
        sp = soup(a, 'html.parser')
        return sp

    def _bio_item_to_player_dict(self, player_dict, bio_item):  # Specific Helper scrape_player_data
        """
        updating the player_dict with info from the player's bio
        """
        bio_label = bio_item.find('span', attrs={'class': 'Bio__Label ttu mr2 dib clr-gray-04'})
        bio_value = bio_item.find('span', attrs={'class': 'dib flex-uniform mr3 clr-gray-01'})
        if bio_label is not None:
            player_dict[bio_label.get_text().strip()] = bio_value.get_text().strip()
        return player_dict

    def _number(self, header_sp):  # Specific Helper scrape_player_data
        """
        scraping a player's number from the header_sp
        """
        item_sps = header_sp.find_all('li')
        number_comp = re.compile(r"^#\d{1,2}$")
        number = None
        for item_sp in item_sps:
            item_text = item_sp.get_text().strip()
            if re.match(number_comp, item_text):
                number = int(item_text.replace('#', ''))
        return number

    def _name(self, header_sp):  # Specific Helper scrape_player_data
        """
        scraping a player's full name
        """
        first_name = header_sp.find('span', attrs={'class': 'truncate min-w-0 fw-light'}).get_text().strip()
        last_name = header_sp.find('span', attrs={'class': 'truncate min-w-0'}).get_text().strip()
        return first_name + ' ' + last_name

    def _team_history(self, sp):  # Specific Helper scrape_player_data
        """
        scraping all the teams a player has played for
        """
        history_sp = sp.find('div', attrs={'class': 'Wrapper Card__Content Career__History__Content'})
        if history_sp is None:
            return None
        team_items = history_sp.find_all('a', attrs={'class': 'AnchorLink Career__History__Item n8 mb4 flex align-center'})
        history_str = ""
        for team_item in team_items:
            team = team_item.find('span', attrs={'class': 'db n8 clr-black'}).get_text().strip()
            seasons = team_item.find('span', attrs={'class': 'db n10 clr-gray-05'}).get_text().strip()
            history_str += team + ' ' + seasons + ', '
        return history_str[:-2]

    def _career_highlights(self, sp):  # Specific Helper scrape_player_data
        """
        scraping a player's career highlights (stuff like MVP, ROY, etc)
        """
        highlights_sp = sp.find('div', attrs={'class': 'Wrapper Card__Content Career__Highlights__Content'})
        if highlights_sp is None:
            return None

        highlight_items = highlights_sp.find_all('div', attrs={'class': 'Career__Highlights__Item n8 mb4 flex align-center'})
        highlights_str = ""
        for highlight_item in highlight_items:
            count_name_sp = highlight_item.find('div', attrs={'class': 'clr-black'})
            count_name_spans = count_name_sp.find_all('span')
            count_name = ' '.join([item.get_text() for item in count_name_spans])
            seasons = highlight_item.find('div', attrs={'class': 'n10 clr-gray-05'}).get_text().strip()
            highlights_str += count_name + ' ' + seasons + ', '
        return highlights_str[:-2]

    def _height_weight(self, player_dict, bio_items):  # Specific Helper scrape_player_data
        height = None
        weight = None
        for bio_item in bio_items:
            bio_label = bio_item.find('span', attrs={'class': 'Bio__Label ttu mr2 dib clr-gray-04'})
            bio_value = bio_item.find('span', attrs={'class': 'dib flex-uniform mr3 clr-gray-01'})
            if bio_label.get_text().strip() == 'HT/WT':
                bio_val = bio_value.get_text().strip()
                if isinstance(bio_val, str) and ',' in bio_val:
                    height, weight = bio_val.split(',')
                    player_dict['Height'] = height
                    player_dict['Weight'] = weight
        return player_dict

    def _draft_info(self, player_dict, bio_items):  # Specific Helper scrape_player_data
        draft_year = None
        draft_round = None
        draft_pick = None
        draft_team = None

        for bio_item in bio_items:
            bio_label = bio_item.find('span', attrs={'class': 'Bio__Label ttu mr2 dib clr-gray-04'})
            bio_value = bio_item.find('span', attrs={'class': 'dib flex-uniform mr3 clr-gray-01'})
            if bio_label.get_text().strip() == 'Draft Info':
                print('here')
                s = bio_value.get_text().strip()
                draft_year = int(s[:4])
                draft_round = int(s.split(",")[0].split(" ")[-1])
                draft_pick = int(s.split("Pk ")[1].split(" ")[0])
                draft_team = s.split("(")[1].split(")")[0]
        player_dict['Draft_Year'] = draft_year
        player_dict['Draft_Round'] = draft_round
        player_dict['Draft_Pick'] = draft_pick
        player_dict['Draft_Team'] = draft_team

        return player_dict

    def scrape_player_data(self, player_id):  # Top Level
        """
        scraping data from the player's ESPN bio
        """
        link = f"https://www.espn.com/{self.league.lower()}/player/bio/_/id/{int(player_id)}/"
        # link = "https://www.espn.com/nba/player/bio/_/id/4395725/tyler-herro" # ! HERRROOOOO
        link = link.replace('ncaaf', 'college-football').replace('ncaab', 'mens-college-basketball')
        print(link)
        sp = self._get_sp1(link)
        header_sp = sp.find('div', attrs={'class': 'ResponsiveWrapper'})
        bio_sp = sp.find('section', attrs={'class': 'Card Bio'}) or sp.find('div', attrs={'class': 'PlayerHeader__Bio pv5'})
        bio_items = bio_sp.find_all('div', attrs={'class': ['Bio__Item n8 mb4', 'fw-medium clr-black']})
        bio_items = [bio_item for bio_item in bio_items if bio_item not in ['HT/WT', 'Draft Info', 'Name']]

        player_dict = self._new_player_dict(player_id)
        for bio_item in bio_items:
            player_dict = self._bio_item_to_player_dict(player_dict, bio_item)

        player_dict = self._height_weight(player_dict, bio_items)
        player_dict = self._draft_info(player_dict, bio_items)
        player_dict['Number'] = self._number(header_sp)
        player_dict['Player'] = self._name(header_sp)
        player_dict['Team_History'] = self._team_history(sp)
        player_dict['Career_Highlights'] = self._career_highlights(sp)
        player_dict['Birthdate'] = datetime.datetime.strptime(player_dict['Birthdate'].split(" ")[0], "%m/%d/%Y").strftime("%Y-%m-%d") if player_dict['Birthdate'] else None
        player_dict['scrape_ts'] = self.scrape_ts
        return player_dict

    def run(self):  # Run
        df = pd.read_csv(self.df_path)
        schema_path = ROOT_PATH + "/data/external/espn/players.json"
        self.test_data_schema.run(schema_path, df)

        print('here')
        unscraped_player_ids = self.unscraped_player_ids(df)
        for i, unscraped_player_id in enumerate(unscraped_player_ids):
            try:
                print(f"{i}/{len(unscraped_player_ids)}")
                player_dict = self.scrape_player_data(unscraped_player_id)
                df.loc[len(df)] = list(player_dict.values())[:19]
                self.test_data_schema.run(schema_path, df)
                print(player_dict['Player'])
                df.to_csv(self.df_path, index=False)
            except Exception as e:
                print(e)
                print(f"{unscraped_player_id} failed")


if __name__ == '__main__':
    for league in ['NBA']:
        x = ESPN_Players(league)
        self = x
        x.run()
