# ==============================================================================
# File: espn_game.py
# Project: allison
# File Created: Tuesday, 28th February 2023 6:45:10 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 6:45:11 am
# Modified By: Dillon Koch
# -----
#
# -----
# scraping data from games on espn.com and saving to /data/external/
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


class ESPN_Game:
    def __init__(self, league):
        self.league = league
        self.df_path = ROOT_PATH + f"/data/external/espn/{self.league}/Games.csv"
        self.test_data_schema = Test_Data_Schema(league)

        self.link_dict = {"NFL": "nfl", "NBA": "nba", "NCAAF": "college-football", "NCAAB": "mens-college-basketball"}
        self.football_league = league in ['NFL', 'NCAAF']

        # ! STATS
        self.football_stats = ['1st_Downs', 'Passing_1st_downs', 'Rushing_1st_downs', '1st_downs_from_penalties',
                               '3rd_down_efficiency', '4th_down_efficiency', 'Total_Plays', 'Total_Yards', 'Total_Drives',
                               'Yards_per_Play', 'Passing', 'Comp_Att', 'Yards_per_pass', 'Interceptions_thrown',
                               'Sacks_Yards_Lost', 'Rushing', 'Rushing_Attempts', 'Yards_per_rush', 'Red_Zone_Made_Att',
                               'Penalties', 'Turnovers', 'Fumbles_lost', 'Defensive_Special_Teams_TDs',
                               'Possession']
        self.ncaaf_empties = ['Passing 1st downs', 'Rushing 1st downs', '1st downs from penalties', 'Total Plays', 'Total Drives',
                              'Yards per Play', 'Sacks-Yards Lost', 'Red Zone (Made-Att)', 'Defensive / Special Teams TDs']

        self.basketball_stats = ['FG', 'Field_Goal_pct', '3PT', 'Three_Point_pct', 'FT', 'Free_Throw_pct', 'Rebounds',
                                 'Offensive_Rebounds', 'Defensive_Rebounds', 'Assists', 'Steals', 'Blocks',
                                 'Total_Turnovers', 'Points_Off_Turnovers', 'Fast_Break_Points', 'Points_in_Paint',
                                 'Fouls', 'Technical_Fouls', 'Flagrant_Fouls', 'Largest_Lead']

    def _get_sp1(self, link):  # Global Helper
        user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
        headers = {'User-Agent': user_agent, }
        request = urllib.request.Request(link, None, headers)  # The assembled request
        response = urllib.request.urlopen(request)
        a = response.read().decode('utf-8', 'ignore')
        sp = soup(a, 'html.parser')
        time.sleep(5)
        return sp

    def query_unscraped_games(self):  # Top Level
        """
        querying data from ESPN_Games_{league} for unscraped games
        - we know it's unscraped if home team is Kennedy
        - also filtering out games that occur in the future
        """
        self.cursor.execute("USE sports_betting;")
        sql = f"""SELECT * FROM ESPN_Games_{self.league} WHERE Home = 'Kennedy Cougars'
                  OR Home = 'nan' OR Away = 'nan'
                  OR Home_Final IS NULL;"""
        self.cursor.execute(sql)
        games = self.cursor.fetchall()
        games = [list(game) for game in games if game[3] < datetime.date.today()]  # list so I can reassign values
        return games

    def scrape_summary_sp(self, game_id):  # Top Level
        """
        scrapes the sp from the game summary page
        """
        league_link_str = self.link_dict[self.league]
        link = f"https://www.espn.com/{league_link_str}/game/_/gameId/{game_id}"
        print(link)
        sp = self._get_sp1(link)
        return sp

    def final_status(self, sp):  # Top Level
        """
        Scraping the "Final Status" of the game (could be "Final" or include OT like "Final/OT")
        """
        final_text = sp.find('span', attrs={'class': 'game-time status-detail'}) or sp.find('div', attrs={'class': 'ScoreCell__Time Gamestrip__Time h9 clr-gray-01'})
        if final_text is None:
            return None
        else:
            return final_text.get_text() if 'Final' in final_text.get_text() else None

    def scrape_date(self, game, sp):  # Top Level
        """
        scrapes the date of the game
        """
        str_sp = str(sp)
        reg_comp = re.compile(
            r"Game Summary - ((January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4})")
        match = re.search(reg_comp, str_sp)
        datetime_ob = datetime.datetime.strptime(match.group(1), "%B %d, %Y")
        game[3] = datetime_ob.strftime("%Y-%m-%d")
        return game

    def scrape_teams(self, game, sp):  # Top Level
        """
        scrapes the team names and adds to new_row
        """
        if self.football_league:
            teams = sp.find_all('a', attrs={'class': 'team-name'})
            if not teams:
                teams = sp.find_all('h2', attrs={'class': 'ScoreCell__TeamName ScoreCell__TeamName--displayName truncate db'})
                away = teams[0].get_text()
                home = teams[1].get_text()
            else:
                away_long = teams[0].find('span', attrs={'class': 'long-name'}).get_text()
                away_short = teams[0].find('span', attrs={'class': 'short-name'}).get_text()
                away = away_long + ' ' + away_short
                home_long = teams[1].find('span', attrs={'class': 'long-name'}).get_text()
                home_short = teams[1].find('span', attrs={'class': 'short-name'}).get_text()
                home = home_long + ' ' + home_short
        else:
            home = sp.find_all('h2', attrs={'class': 'ScoreCell__TeamName ScoreCell__TeamName--displayName truncate db'})[1].get_text()
            away = sp.find_all('h2', attrs={'class': 'ScoreCell__TeamName ScoreCell__TeamName--displayName truncate db'})[0].get_text()
        game[4] = home.replace("'", "")
        game[5] = away.replace("'", "")
        return game

    def scrape_team_records(self, game, sp):  # Top Level
        """
        scrapes the home and away team records, adds to game
        """
        records = sp.find_all('div', attrs={'class': 'record'}) or sp.find_all('div', attrs={'class': 'Gamestrip__Record db n10 clr-gray-03'})
        away_record = records[0].get_text()
        home_record = records[1].get_text()
        home_wins = int(home_record.split("-")[0]) if home_record else None
        home_losses = int(home_record.split(",")[0].split("-")[1]) if home_record else None
        away_wins = int(away_record.split("-")[0]) if away_record else None
        away_losses = int(away_record.split(",")[0].split("-")[1]) if away_record else None
        game[6] = home_wins
        game[7] = home_losses
        game[8] = away_wins
        game[9] = away_losses
        return game

    def scrape_network(self, game, sp):  # Top Level
        """
        scrapes the TV network of the game, adds to game
        """
        try:
            network = sp.find_all('div', attrs={'class': 'game-network'}) or sp.find_all('div', attrs={'class': 'n8 GameInfo__Meta'})[0].find_all('span')[1:]
            network = network[0].get_text()
            network = network.replace("\n", '').replace("\t", "")
            network = network.replace("Coverage: ", "")
        except IndexError:
            network = None
        game[10] = network
        return game

    def scrape_stats_sp(self, game_id):  # Top Level
        """
        Scrapes the HTML from ESPN for the given game_id
        """
        league_link_str = self.link_dict[self.league]
        link = f"https://www.espn.com/{league_link_str}/matchup?gameId={game_id}"
        print('stats', link)
        sp = self._get_sp1(link)
        return sp

    def scrape_halves(self, game, sp, home):  # Top Level
        """
        scrapes the first and second half of the game if it's NCAAB, else returns None
        """
        first_half = None
        second_half = None

        # * scraping halves if the league is NCAAB
        if self.league == 'NCAAB':
            table_sp = sp.find('div', attrs={'class': 'Table__Scroller'})
            table_body = table_sp.find('tbody')
            away_row, home_row = table_body.find_all('tr')
            td_vals = home_row.find_all('td') if home else away_row.find_all('td')

            first_half = None
            second_half = None
            if len(td_vals) in [4, 5]:
                first_half = td_vals[1].get_text()
                second_half = td_vals[2].get_text()

        # * updating game based on home/away team
        if home:
            game[12] = int(first_half) if first_half else first_half
            game[13] = int(second_half) if second_half else second_half
        else:
            game[19] = int(first_half) if first_half else first_half
            game[20] = int(second_half) if second_half else second_half

        return game

    def scrape_quarters_ot(self, game, sp, home):  # Top Level
        """
        scrapes the quarter values and OT
        - quarters only if it's not NCAAB, but OT either way
        """
        # scores_sp = sp.find_all('table', attrs={'id': 'linescore'})[0] if self.football_league else sp.find('div', attrs={'class': 'Table__Scroller'})
        scores_sp = sp.find('div', attrs={'class': 'Table__Scroller'})
        body = scores_sp.find_all('tbody')[0]
        rows = body.find_all('tr')
        away_row, home_row = rows
        td_vals = home_row.find_all('td') if home else away_row.find_all('td')

        q1, q2, q3, q4, ot = None, None, None, None, None
        if len(td_vals) == 5:
            ot = td_vals[3].get_text()

        if len(td_vals) in [6, 7]:
            q1 = td_vals[1].get_text()
            q2 = td_vals[2].get_text()
            q3 = td_vals[3].get_text()
            q4 = td_vals[4].get_text()
            q1 = int(q1) if q1 else q1
            q2 = int(q2) if q2 else q2
            q3 = int(q3) if q3 else q3
            q4 = int(q4) if q4 else q4

        if len(td_vals) == 7:
            ot = td_vals[5].get_text()
            ot = int(ot) if ot else ot

        if home:
            game[14:19] = [q1, q2, q3, q4, ot]
        else:
            game[21:26] = [q1, q2, q3, q4, ot]
        return game

    def scrape_final_scores(self, game, sp):  # Top Level
        """
        scrapes the game's final scores, adds to new_row
        """
        # scores_sp = sp.find_all('table', attrs={'id': 'linescore'})[0] if self.football_league else sp.find('div', attrs={'class': 'Table__Scroller'})
        scores_sp = sp.find('div', attrs={'class': 'Table__Scroller'})
        body = scores_sp.find_all('tbody')[0]
        rows = body.find_all('tr')
        away_row, home_row = rows
        # away_score = away_row.find_all('td', attrs={'class': 'final-score'})[0].get_text() if self.football_league else away_row.find_all('td')[-1].get_text()
        # home_score = home_row.find_all('td', attrs={'class': 'final-score'})[0].get_text() if self.football_league else home_row.find_all('td')[-1].get_text()
        away_score = away_row.find_all('td')[-1].get_text()
        home_score = home_row.find_all('td')[-1].get_text()
        game[26] = home_score
        game[27] = away_score
        return game

    def _body_to_stats(self, body, stat):
        try:
            trs = body.find_all('tr')
            for tr in trs:
                data = tr.find_all('td')
                data = [item.get_text() for item in data]
                if data and stat == data[0]:
                    return data[1], data[2]

            raise ValueError(f"didnt find stat {stat}")

        except Exception as e:
            print(e)
            print(f"Invalid values for stat {stat}")
            return None, None

    def _body_to_dash_stats(self, body, stat):  # Helping Helper _scrape_football
        try:
            trs = body.find_all('tr')
            for tr in trs:
                data = tr.find_all('td')
                data = [item.get_text() for item in data]
                if data and stat == data[0]:

                    if '-' in data[1]:
                        r1, r2 = data[1].split("-")
                        r3, r4 = data[2].split('-')
                        return r1, r2, r3, r4
                    elif '/' in data[1]:
                        r1, r2 = data[1].split("/")
                        r3, r4 = data[2].split('/')
                        return r1, r2, r3, r4
                    else:
                        print('ehre')
                        print(data)
            raise ValueError(f"didnt find stat {stat}")

        except Exception as e:
            print(e)
            print(f"Invalid values for stat {stat}")
            return None, None

    def _scrape_football(self, game, body):  # Specific Helper scrape_stats
        game[28:30] = self._body_to_stats(body, "1st Downs")
        game[30:32] = self._body_to_stats(body, "Passing 1st downs")
        game[32:34] = self._body_to_stats(body, "Rushing 1st downs")
        game[34:36] = self._body_to_stats(body, "1st downs from penalties")
        game[36:40] = self._body_to_dash_stats(body, "3rd down efficiency")
        game[40:44] = self._body_to_dash_stats(body, "4th down efficiency")
        game[44:46] = self._body_to_stats(body, "Total Plays")
        game[46:48] = self._body_to_stats(body, "Total Yards")
        game[48:50] = self._body_to_stats(body, "Total Drives")
        game[50:52] = self._body_to_stats(body, "Yards per Play")
        game[52:54] = self._body_to_stats(body, "Passing")
        game[54:58] = self._body_to_dash_stats(body, "Comp-Att")
        game[58:60] = self._body_to_stats(body, "Yards per pass")
        game[60:62] = self._body_to_stats(body, "Interceptions thrown")
        game[62:66] = self._body_to_dash_stats(body, "Sacks-Yards Lost")
        game[66:68] = self._body_to_stats(body, "Rushing")
        game[68:70] = self._body_to_stats(body, "Rushing Attempts")
        game[70:72] = self._body_to_stats(body, "Yards per rush")
        game[72:76] = self._body_to_dash_stats(body, "Red Zone (Made-Att)")
        game[76:80] = self._body_to_dash_stats(body, "Penalties")
        game[80:82] = self._body_to_stats(body, "Turnovers")
        game[82:84] = self._body_to_stats(body, "Fumbles lost")
        game[84:86] = self._body_to_stats(body, "Defensive / Special Teams TDs")
        game[86:88] = self._body_to_stats(body, "Possession")
        return game

    def _scrape_basketball(self, game, body):  # Specific Helper scrape_stats
        # TODO accomodate stats split by dash
        split_stats = ['FG', '3PT', 'FT']
        rows = body.find_all('tr')
        start = 28
        for row in rows:
            stat, away, home = row.find_all('td')
            stat = stat.get_text()
            away = away.get_text()
            home = home.get_text()
            if stat in split_stats:
                assert ('-' in home) and ('-' in away)
                home_1, home_2 = home.split('-')
                away_1, away_2 = away.split('-')
                game[start:start + 4] = [home_1, home_2, away_1, away_2]
                start += 4

            else:
                game[start:start + 2] = [home, away]
                start += 2
        return game

    def scrape_stats(self, game, sp):  # Top Level
        try:
            # idx = 0 if self.football_league else 1
            # table = sp.find_all('table', attrs={'class': ['mod-data', 'Table Table--align-right']})[idx]
            table = sp.find('section', attrs={'class': 'Card TeamStatsTable'})
            body = table.find('tbody')
            if self.league in ['NFL', 'NCAAF']:
                game = self._scrape_football(game, body)
            else:
                game = self._scrape_basketball(game, body)
            return game
        except Exception as e:
            print(e)
            print('error scraping stats')
            return game

    def scrape_odds(self, game, summary_sp):  # Top Level
        # * over/under
        over_under = None
        over_under_sp = summary_sp.find_all('div', attrs={'class': 'n8 GameInfo__BettingItem flex-expand ou'})
        if over_under_sp:
            over_under = float(over_under_sp[0].get_text().split(' ')[1])

        # * spread
        spread = None
        spread_sp = summary_sp.find_all('div', attrs={'class': 'n8 GameInfo__BettingItem flex-expand line'})
        if spread_sp:
            spread = spread_sp[0].get_text().replace("Line: ", "")

        game[-2:] = [spread, over_under]
        return game

    def run(self):  # Run
        df = pd.read_csv(self.df_path)
        df.drop_duplicates(subset='Game_ID', keep='first', inplace=True)
        df.sort_values(by='Date', inplace=True)
        df_unscraped = df.loc[df['H1Q'].isnull()]  # TODO fix for NCAAB
        unscraped_games = df_unscraped.values.tolist()
        df_scraped = df.loc[df['H1Q'].notnull()]

        schema_path = ROOT_PATH + f"/data/external/espn/{self.league}/espn_{self.league.lower()}_games.json"
        self.test_data_schema.run(schema_path, df_scraped)

        for i, game in enumerate(unscraped_games):
            print(f"{i}/{len(unscraped_games)}")
            if isinstance(game[3], str) and datetime.datetime.strptime(game[3], "%Y-%m-%d") > datetime.datetime.today() + datetime.timedelta(days=1):
                continue

            try:
                game_id = game[0]

                # ! SUMMARY INFORMATION
                summary_sp = self.scrape_summary_sp(game_id)
                game = self.scrape_date(game, summary_sp)
                game = self.scrape_teams(game, summary_sp)
                game = self.scrape_team_records(game, summary_sp)
                game = self.scrape_network(game, summary_sp)
                game = self.scrape_odds(game, summary_sp)

                if datetime.datetime.strptime(game[3], "%Y-%m-%d") < datetime.datetime.today():  # TODO scrape games coming soon to update records/etc
                    # * game has been played, scrape everything
                    # ! STATS INFORMATION
                    stats_sp = self.scrape_stats_sp(game_id)
                    game = self.scrape_halves(game, stats_sp, home=True)
                    game = self.scrape_quarters_ot(game, stats_sp, home=True)
                    game = self.scrape_halves(game, stats_sp, home=False)
                    game = self.scrape_quarters_ot(game, stats_sp, home=False)
                    game = self.scrape_final_scores(game, stats_sp)
                    game = self.scrape_stats(game, stats_sp)
                    game[11] = "Final" if game[18] is None else "Final/OT"
                    game[12:-2] = [float(item) if item else item for item in game[12:-2]]
                print(f"{game[5]} at {game[4]}, {game[3]}")

                self.test_data_schema.run(schema_path, pd.DataFrame([game], columns=df.columns))
                df.loc[df['Game_ID'] == game[0]] = game
                df.to_csv(self.df_path, index=False)
            except BaseException as e:
                print(e)
                print(self.league, game_id)


if __name__ == '__main__':
    for league in ['NBA']:
        x = ESPN_Game(league)
        self = x
        x.run()
