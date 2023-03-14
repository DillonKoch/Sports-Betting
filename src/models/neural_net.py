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


import sys
from os.path import abspath, dirname

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from src.models.model_parent import Model_Parent


class NeuralNet1(nn.Module):
    def __init__(self, layer_sizes):
        super(NeuralNet1, self).__init__()
        self.fcs = [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 2)]
        self.final_fc = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for fc in self.fcs:
            x = fc(x)
            x = self.relu(x)
        out = self.final_fc(x)
        out = self.sigmoid(out)
        return out


class BettingDataset(Dataset):
    def __init__(self, league, bet_type, n_games):
        super().__init__()
        self.league = league
        self.bet_type = bet_type
        self.n_games = n_games
        self.parent = Model_Parent(self.league, self.bet_type)
        self.parent.n_games = self.n_games

        self.train, _, _ = self.parent.load_data()
        self.X, self.y = self.parent.separate_X_y(self.train, balance_classes=True)
        print('here')

    def __len__(self):  # Run
        return len(self.X)

    def __getitem__(self, idx):  # Run
        # self.X[idx], self.y[idx]
        return torch.tensor(self.X.iloc[idx].values).to(torch.float32), torch.tensor(self.y.iloc[idx]).to(torch.float32)


class NeuralNetModel(Model_Parent):
    def __init__(self, league, bet_type, hyperparameters):
        super().__init__(league, bet_type)
        self.n_games, self.feature_selection, self.architecture, self.batch_size, self.lr = hyperparameters
        print(self.__str__())
        self.model = None  # * have to define in train after finding out input size

    def __str__(self):  # Run
        print(f"Neural Net, n_games={self.n_games}, feature selection={self.feature_selection}, architecture={self.architecture}, batch_size={self.batch_size}, lr={self.lr}")

    def fit_model(self):  # Top Level
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        betting_dataset = BettingDataset(self.league, self.bet_type, self.n_games)
        betting_dataloader = DataLoader(betting_dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(10):
            for batch_X, batch_y in betting_dataloader:
                y_pred = self.model(batch_X)

                loss = criterion(y_pred, batch_y.unsqueeze(1))
                # acc = accuracy_score(y_pred.detach().numpy().astype(int), batch_y.detach().numpy())
                # print(round(acc, 4))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch} loss: {round(loss.item(), 4)}")

    def train(self):  # Run
        train, val, test = self.load_data()
        train_X, train_y = self.separate_X_y(train, balance_classes=True)
        self.model = NeuralNet1([train_X.shape[1]] + self.architecture + [1])
        val_X, val_y = self.separate_X_y(val)

        if self.feature_selection:
            self.selector = SelectKBest(mutual_info_classif, k=self.feature_selection)
            train_X = self.selector.fit_transform(train_X, train_y)
            val_X = self.selector.transform(val_X)

        self.fit_model()

        with torch.no_grad():
            self.model.eval()

            val_preds = self.model(torch.tensor(val_X.values).to(torch.float32))
            print(pd.Series([1 if item.item() > 0.5 else 0 for item in list(val_preds)]).value_counts())
            val_acc = accuracy_score(val_preds.detach().numpy().astype(int), val_y)
            print(f"Validation accuracy: {round(val_acc, 3)}")
            self.val_acc = val_acc
            return val_acc


if __name__ == '__main__':
    x = NeuralNetModel()
    self = x
    x.run()
