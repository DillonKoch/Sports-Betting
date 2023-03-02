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


from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Build_Features:
    def __init__(self):
        pass

    def run(self):  # Run
        # espn game (avgs)
        # espn player stats (avgs) with player info, injury info
        # betting info
        pass


if __name__ == '__main__':
    x = Build_Features()
    self = x
    x.run()
