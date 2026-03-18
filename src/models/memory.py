from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MemoryConfig:
    slot_count: int
    slot_dim: int


class MemoryController(nn.Module):
    def __init__(self, config: MemoryConfig) -> None:
        super().__init__()
        self.config = config
        self.read_query = nn.Linear(config.slot_dim, config.slot_dim)
        self.write_gate = nn.Linear(config.slot_dim, config.slot_count)

    def empty_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.config.slot_count, self.config.slot_dim, device=device)

    def read(self, query_state: torch.Tensor, memory_state: torch.Tensor) -> torch.Tensor:
        query = self.read_query(query_state).unsqueeze(1)
        scores = torch.matmul(query, memory_state.transpose(1, 2)).squeeze(1)
        weights = scores.softmax(dim=-1)
        return torch.einsum("bs,bsd->bd", weights, memory_state)

    def write(self, write_state: torch.Tensor, memory_state: torch.Tensor) -> torch.Tensor:
        slot_scores = self.write_gate(write_state).softmax(dim=-1).unsqueeze(-1)
        updated = slot_scores * write_state.unsqueeze(1)
        return memory_state + updated
