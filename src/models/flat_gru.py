from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.heads import ClassifierHead, ClassifierHeadConfig


@dataclass(frozen=True)
class FlatGRUConfig:
    vocab_size: int
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    num_classes: int
    pad_token_id: int = 0


class FlatGRUBaseline(nn.Module):
    def __init__(self, config: FlatGRUConfig) -> None:
        super().__init__()
        recurrent_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_token_id,
        )
        self.recurrent = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.head = ClassifierHead(
            ClassifierHeadConfig(
                input_dim=config.hidden_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.num_classes,
                dropout=config.dropout,
            )
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden_state = self.recurrent(packed)
        return self.head(hidden_state[-1])
