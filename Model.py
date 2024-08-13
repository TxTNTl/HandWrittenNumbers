import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 128)
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        y = self.layer1(x)
        z = self.layer2(torch.relu(y))
        return z





