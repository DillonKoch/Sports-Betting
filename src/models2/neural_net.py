# ==============================================================================
# File: neural_net.py
# Project: allison
# File Created: Wednesday, 22nd March 2023 10:11:59 am
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 22nd March 2023 10:12:00 am
# Modified By: Dillon Koch
# -----
#
# -----
# neural net class for predicting spread/total/ML
# ==============================================================================


from os.path import abspath, dirname
import sys
import torch.nn as nn

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


# MODEL
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.sigmoid(self.fc6(x))
        return x
