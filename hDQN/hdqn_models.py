import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class MetaControllerModel(nn.Module):
    def __init__(self, n_options, device):
        super(MetaControllerModel, self).__init__()

        self.device = device

        self.visual_pipeline = nn.Sequential(
            nn.Conv2d(1, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            Flatten()
        )

        self.value_function = nn.Sequential(
            nn.Linear(in_features=4, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_options)
        )

        self.to(device)

    def forward(self, state):
        x = self.visual_pipeline(state)
        x = self.value_function(x)
        return x


class ControllerModel(nn.Module):
    def __init__(self, n_actions, device):
        super(ControllerModel, self).__init__()

        self.device = device

        self.visual_pipeline = nn.Sequential(
            nn.Conv2d(1, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            Flatten()
        )

        self.value_function = nn.Sequential(
            nn.Linear(in_features=7616, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_actions)
        )

        self.to(device)

    def forward(self, state):
        x = self.visual_pipeline(state)
        x = self.value_function(x)
        return x
