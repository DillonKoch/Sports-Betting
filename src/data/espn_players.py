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


from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class ESPN_Players:
    def __init__(self):
        pass

    def run(self):  # Run
        pass


if __name__ == '__main__':
    x = ESPN_Players()
    self = x
    x.run()
