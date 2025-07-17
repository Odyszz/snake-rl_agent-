import torch
import torch.nn as nn

class SnakeDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(19, 196),
            nn.ReLU(),
            nn.Linear(196, 324),
            nn.ReLU(),
            nn.Linear(324, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        return self.net(x)