from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class WorkspaceConfig:
    slot_count: int
    slot_dim: int


@dataclass
class WorkspaceState:
    slots: torch.Tensor
    mask: torch.Tensor
    budget: torch.Tensor
    confidence: torch.Tensor


class SharedWorkspace(nn.Module):
    def __init__(self, config: WorkspaceConfig) -> None:
        super().__init__()
        self.config = config
        self.initial_slots = nn.Parameter(torch.zeros(config.slot_count, config.slot_dim))

    def initial_state(self, batch_size: int, device: torch.device) -> WorkspaceState:
        slots = self.initial_slots.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device)
        mask = torch.ones(batch_size, self.config.slot_count, dtype=torch.bool, device=device)
        budget = torch.ones(batch_size, 1, device=device)
        confidence = torch.zeros(batch_size, 1, device=device)
        return WorkspaceState(slots=slots, mask=mask, budget=budget, confidence=confidence)

    def inject(self, workspace: WorkspaceState, encoded_input: torch.Tensor) -> WorkspaceState:
        workspace.slots[:, 0, :] = workspace.slots[:, 0, :] + encoded_input
        return workspace
