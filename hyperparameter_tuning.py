# ==============================================================================
# File: hyperparameter_tuning.py
# Project: allison
# File Created: Sunday, 5th March 2023 6:26:02 pm
# Author: Dillon Koch
# -----
# Last Modified: Sunday, 5th March 2023 6:26:02 pm
# Modified By: Dillon Koch
# -----
#
# -----
# class for performing different flavors of hyperparameter tuning
# grid search, random search, coarse-to-fine search
# ==============================================================================


from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Hyperparameter_Tuning:
    def __init__(self):
        pass

    def grid_search(self, search_space):
        # TODO return all combinations
        pass

    def random_search(self, search_space, n):
        # TODO return n random combinations
        pass

    def fine_search(self, search_space):
        pass


if __name__ == '__main__':
    x = Hyperparameter_Tuning()
    self = x
    x.run()
