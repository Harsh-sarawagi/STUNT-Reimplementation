"""
MLP encoder for tabular rows.

Architecture adapted from the official STUNT MLPProto model.
"""

import torch
from torch import nn


class TabularEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 1024,
        embedding_dim: int = 1024,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
                bias=True,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
                bias=True,
            ),
        )

    def forward(self, x: torch.Tensor):

        return self.network(x)