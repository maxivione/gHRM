from __future__ import annotations

import torch
from torch import nn


class OutputHeads(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.answer = nn.Linear(hidden_dim, output_dim)
        self.confidence = nn.Linear(hidden_dim, 1)
        self.action = nn.Linear(hidden_dim, 4)

    def forward(self, hidden_state: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "answer_logits": self.answer(hidden_state),
            "confidence": torch.sigmoid(self.confidence(hidden_state)),
            "action_logits": self.action(hidden_state),
        }
