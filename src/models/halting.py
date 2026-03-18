from __future__ import annotations

import torch
from torch import nn


class HaltingHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.logit = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logit(hidden_state))
