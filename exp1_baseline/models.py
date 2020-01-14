import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_NN(nn.Module):

    def __init__(self):
        super(MLP_NN, self).__init__()

        self.fc1 = nn.Linear(144, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        x = x[:, -1, :]

        # Feed to the first hidden layer
        x = F.relu(self.fc1(x))
        # Get the final output
        x = self.fc2(x)
        return x


class LSTM_NN(nn.Module):

    def __init__(self):
        super(LSTM_NN, self).__init__()

        self.lstm = nn.LSTM(input_size=144, hidden_size=32)
        self.fc1 = nn.Linear(32, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        # PyTorch needs the channels first (seq_len, batch_size,  dim)
        x = x.transpose(0, 1)
        # Get the last hidden state
        x = self.lstm(x)[0][-1, :, :]

        # Feed to the first hidden layer
        x = F.relu(self.fc1(x))
        # Get the final output
        x = self.fc2(x)
        return x
