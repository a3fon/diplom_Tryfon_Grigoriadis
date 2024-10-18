import torch
import torch.nn as nn
import torch.nn.functional as F
class ReceptorNet(nn.Module):
    def __init__(self):
        super(ReceptorNet, self).__init__()
        self.fc_1 = nn.Linear(36, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.output = nn.Linear(256, 3)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x_out = self.output(x)  # No sigmoid is used here
        return x_out
