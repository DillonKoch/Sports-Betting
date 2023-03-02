# ==============================================================================
# File: match_player.py
# Project: allison
# File Created: Tuesday, 28th February 2023 7:04:39 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 7:04:40 am
# Modified By: Dillon Koch
# -----
#
# -----
# matching a scraped or downloaded player name with the official one from ESPN
# ==============================================================================

from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Match_Player:
    def __init__(self):
        pass

    def run(self):  # Run
        pass


if __name__ == '__main__':
    x = Match_Player()
    self = x
    x.run()
