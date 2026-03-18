from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SymbolicAdapterConfig:
    input_dim: int
    workspace_dim: int


class SymbolicAdapter(nn.Module):
    def __init__(self, config: SymbolicAdapterConfig) -> None:
        super().__init__()
        self.projection = nn.Linear(config.input_dim, config.workspace_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs)
