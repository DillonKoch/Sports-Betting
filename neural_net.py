# ==============================================================================
# File: neural_net.py
# Project: allison
# File Created: Tuesday, 28th February 2023 7:11:58 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 28th February 2023 7:11:58 am
# Modified By: Dillon Koch
# -----
#
# -----
# neural net models
# ==============================================================================


from os.path import abspath, dirname
import sys
import torch
import torch.nn as nn

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class NeuralNet1(nn.Module):
    def __init__(self):
        super(NeuralNet1, self).__init__()
        self.fc1 = nn.Linear(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # Run
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


if __name__ == '__main__':
    x = NeuralNet1()
    self = x
    x.run()
