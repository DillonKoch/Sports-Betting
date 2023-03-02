# ==============================================================================
# File: trainer.py
# Project: allison
# File Created: Tuesday, 28th February 2023 7:05:29 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 7:05:29 am
# Modified By: Dillon Koch
# -----
#
# -----
# using model classes to train models with hyperparameters and save the best one
# ==============================================================================


from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Trainer:
    def __init__(self):
        pass

    def run(self):  # Run
        pass


if __name__ == '__main__':
    x = Trainer()
    self = x
    x.run()
