# ==============================================================================
# File: baseline.py
# Project: allison
# File Created: Monday, 6th March 2023 12:25:40 pm
# Author: Dillon Koch
# -----
# Last Modified: Monday, 6th March 2023 12:25:41 pm
# Modified By: Dillon Koch
# -----
#
# -----
# baseline model for making predictions (predicts at random)
# ==============================================================================

import random
import sys
from os.path import abspath, dirname

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Baseline:
    def __init__(self):
        pass

    def predict(self, X):  # Run
        return 1 if random.random() > 0.5 else 0


if __name__ == '__main__':
    x = Baseline()
    self = x
    x.run()
