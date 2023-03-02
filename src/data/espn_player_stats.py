# ==============================================================================
# File: espn_player_stats.py
# Project: allison
# File Created: Tuesday, 28th February 2023 6:46:48 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 6:46:49 am
# Modified By: Dillon Koch
# -----
#
# -----
# scraping data about players' stats from espn.com and saving to /data/external/
# ==============================================================================


from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class ESPN_Player_Stats:
    def __init__(self):
        pass

    def run(self):  # Run
        pass


if __name__ == '__main__':
    x = ESPN_Player_Stats()
    self = x
    x.run()
