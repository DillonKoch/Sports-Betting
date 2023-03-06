# ==============================================================================
# File: dataset.py
# Project: allison
# File Created: Saturday, 4th March 2023 7:15:36 pm
# Author: Dillon Koch
# -----
# Last Modified: Saturday, 4th March 2023 7:15:37 pm
# Modified By: Dillon Koch
# -----
#
# -----
# dataset class for pytorch models
# ==============================================================================

from os.path import abspath, dirname
import sys
import torch
from torch.utils.data import Dataset

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class SB_Dataset(Dataset):
    def __init__(self, league, bet_type):
        self.league = league
        self.bet_type = bet_type

    def __len__(self):  # Run
        pass

    def __getitem__(self, idx):  # Run
        pass
