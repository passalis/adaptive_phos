import torch
import torch.nn as nn
import torch.nn.functional as F


def photonic_sigmoid(x, cutoff=2):
    A1 = 0.060
    A2 = 1.005
    x0 = 0.145
    d = 0.033

    x = x - x0
    x[x > cutoff] = cutoff

    y = A2 + (A1 - A2) / (1 + torch.exp(x / d))

    return y


class PhotonicRecurrentNeuron(torch.nn.Module):
    def __init__(self, input_size=144, hidden_size=32, activation='sigmoid'):
        super(PhotonicRecurrentNeuron, self).__init__()

        self.fc1 = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden_size = hidden_size
        self.activation = activation

    def forward(self, x, scaler=None):
        # Create the hidden state
        h = torch.zeros((x.size(1), self.hidden_size)).cuda()

        # For each time_step
        for i in range(x.size(0)):
            x_concat = torch.cat([h, x[i, :, :]], dim=1)

            h = self.fc1(x_concat)
            if scaler is not None:
                h = h * torch.abs(scaler)

            if self.activation == 'sigmoid':
                h = F.sigmoid(h)
            elif self.activation == 'tanh':
                h = F.tanh(h)
            elif self.activation == 'photonic':
                h = photonic_sigmoid(h)

        return h


class Photonic_RNN(nn.Module):

    def __init__(self, activation='sigmoid'):
        super(Photonic_RNN, self).__init__()

        self.recurrent_neuron = PhotonicRecurrentNeuron(input_size=144, hidden_size=32, activation=activation)
        self.fc1 = nn.Linear(32, 512)
        self.fc2 = nn.Linear(512, 3)
        self.weights = [self.recurrent_neuron.fc1.weight, self.fc1.weight, self.fc2.weight]
        self.n_layers = 3
        self.layer_std = [0.2, 0.2, 0.8]
        self.activation = activation

        self.projector_sizes = [(32, 3), (512, 3), (3, 3)]

    def forward(self, x, layer=-1, scaler=None):
        # PyTorch needs the channels first (seq_len, batch_size,  dim)
        x = x.transpose(0, 1)

        # Get the last hidden state
        if layer == 0:
            x = self.recurrent_neuron(x, scaler=scaler)
        else:
            x = self.recurrent_neuron(x)

        # Exit point
        if layer == 0:
            return x

        # Feed to the first hidden layer
        if layer == 1:
            x = self.fc1(x)
            x = x * torch.abs(scaler)
        else:
            x = self.fc1(x)

        if self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.activation == 'tanh':
            x = torch.tanh(x)
        elif self.activation == 'photonic':
            x = photonic_sigmoid(x)

        # Exit point
        if layer == 1:
            return x

        if layer == 2:
            x = self.fc2(x) * torch.abs(scaler)
        else:
            x = self.fc2(x)

        return x
