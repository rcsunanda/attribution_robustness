import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedFullyConnectedNetwork(nn.Module):
    """ A fully connected network with hard-coded architecture. Used as a testing base
    3 hidden layers: 64, 32, 16 nodes, ReLU activated, logits output (not softmaxed)
    """
    def __init__(self, input_size, output_size):
        super(FixedFullyConnectedNetwork, self).__init__()
        h1, h2, h3 = 64, 32, 16
        self.fc1 = nn.Linear(input_size, h1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(h1, h2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(h2, h3)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(h3, output_size)
        # self.out_act = nn.Softmax(dim=1)

        # self.layers = [self.fc1, self.relu1, self.fc2, self.relu2, self.fc3, self.relu3, self.out]

    def forward(self, x):
        x = x.view(x.size(0), -1)   # Flatten image into a vector
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        logits = self.out(x)
        return logits
