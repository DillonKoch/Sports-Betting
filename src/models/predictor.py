# ==============================================================================
# File: predictor.py
# Project: allison
# File Created: Tuesday, 28th February 2023 7:20:06 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 7:20:07 am
# Modified By: Dillon Koch
# -----
#
# -----
# loading best saved models and making predictions on games
# predictions saved in /data/predictions
# ==============================================================================


from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Predictor:
    def __init__(self):
        pass

    def run(self):  # Run
        pass


if __name__ == '__main__':
    x = Predictor()
    self = x
    x.run()
