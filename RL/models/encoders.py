from __future__ import annotations

from typing import List
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dims: List[int], dropout: float = 0.0):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NodeEncoder(MLP):
    pass


class EdgeEncoder(MLP):
    pass


class HaloEncoder(MLP):
    pass

