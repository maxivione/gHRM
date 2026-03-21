from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.heads import ClassifierHead, ClassifierHeadConfig


@dataclass(frozen=True)
class HierarchicalGRUConfig:
    vocab_size: int
    embedding_dim: int
    worker_hidden_dim: int
    planner_hidden_dim: int
    fusion_dim: int
    planner_interval: int
    dropout: float
    num_classes: int
    pad_token_id: int = 0


class HierarchicalGRUBaseline(nn.Module):
    def __init__(self, config: HierarchicalGRUConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_token_id,
        )
        self.worker = nn.GRUCell(config.embedding_dim + config.planner_hidden_dim, config.worker_hidden_dim)
        self.planner_input = nn.Linear(config.worker_hidden_dim, config.planner_hidden_dim)
        self.planner = nn.GRUCell(config.planner_hidden_dim, config.planner_hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.fusion = nn.Linear(config.worker_hidden_dim + config.planner_hidden_dim, config.fusion_dim)
        self.head = ClassifierHead(
            ClassifierHeadConfig(
                input_dim=config.fusion_dim,
                hidden_dim=config.fusion_dim,
                output_dim=config.num_classes,
                dropout=config.dropout,
            )
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        batch_size, sequence_length, _ = embedded.shape
        device = embedded.device

        worker_state = torch.zeros(batch_size, self.config.worker_hidden_dim, device=device)
        planner_state = torch.zeros(batch_size, self.config.planner_hidden_dim, device=device)

        for step in range(sequence_length):
            active_mask = (step < lengths).unsqueeze(-1)
            worker_input = torch.cat([embedded[:, step, :], planner_state], dim=-1)
            next_worker_state = self.worker(worker_input, worker_state)
            worker_state = torch.where(active_mask, next_worker_state, worker_state)

            if (step + 1) % self.config.planner_interval == 0:
                planner_input = self.planner_input(worker_state)
                next_planner_state = self.planner(planner_input, planner_state)
                planner_state = torch.where(active_mask, next_planner_state, planner_state)

        fused_state = torch.cat([worker_state, planner_state], dim=-1)
        fused_state = self.dropout(torch.tanh(self.fusion(fused_state)))
        return self.head(fused_state)
