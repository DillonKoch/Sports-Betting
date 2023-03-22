# ==============================================================================
# File: trainer.py
# Project: allison
# File Created: Wednesday, 22nd March 2023 10:12:43 am
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 22nd March 2023 10:12:44 am
# Modified By: Dillon Koch
# -----
#
# -----
# class for training neural net model using dataset/model files
# ==============================================================================

import sys
from os.path import abspath, dirname

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import clear_output
from torch.utils.data import DataLoader

ROOT_PATH = dirname(dirname(dirname(abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from src.models2.dataset import BettingDataset
from src.models2.neural_net import NeuralNet

epochs = 1000
batch_size = 32
learning_rate = 0.0003

league = 'NBA'
bet_type = 'Spread'
n_games = 5


def train(epochs, batch_size, learning_rate, league, bet_type, n_games):

    train_dataset = BettingDataset(league, bet_type, n_games, 'train')
    val_dataset = BettingDataset(league, bet_type, n_games, 'val')
    test_dataset = BettingDataset(league, bet_type, n_games, 'test')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(train_dataset.X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels)
            if epoch == 0 and i == 0:
                print(f'initial loss: {loss.item()}')
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # preds = torch.argmax(outputs, dim=1)
            preds = (outputs > 0.5).int().view(-1)
            correct += sum(preds == labels)
            total += len(inputs)

        # evaluate on validation set
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            model.eval()
            for i, (val_inputs, val_labels) in enumerate(val_dataloader):
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs.view(-1), val_labels).item()
                # val_preds = torch.argmax(val_outputs, dim=1)
                val_preds = (val_outputs > 0.5).int().view(-1)
                val_correct += sum(val_preds == val_labels)
                val_total += len(val_inputs)
        val_loss /= len(val_dataloader)
        val_acc = val_correct / val_total
        train_acc = correct / total

        print(f'Train: {correct}/{total} correct')
        print(f'Val: {val_correct}/{val_total} correct')
        print(f'Epoch {epoch+1}/{epochs}, Loss: {round(running_loss/len(train_dataloader), 4)}')

        # * update plots
        train_losses.append(running_loss / len(train_dataloader))
        val_losses.append(val_loss)
        train_accs.append(train_acc.item())
        val_accs.append(val_acc.item())
        clear_output(wait=True)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train')
        plt.plot(val_accs, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{league}_{bet_type}_{n_games}_{batch_size}_{learning_rate}.png')
        plt.show()


if __name__ == '__main__':
    train(3, 32, 0.001, 'NBA', 'spread', 5)
