from __future__ import annotations

import torch
from torch import nn


class FastWorker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.cell = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, worker_input: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.cell(worker_input, hidden_state)
