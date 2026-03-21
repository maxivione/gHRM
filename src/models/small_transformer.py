from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from src.models.heads import ClassifierHead, ClassifierHeadConfig


@dataclass(frozen=True)
class SmallTransformerConfig:
    vocab_size: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    max_seq_len: int
    dropout: float
    head_hidden_dim: int
    num_classes: int
    pad_token_id: int = 0


class SmallTransformerBaseline(nn.Module):
    def __init__(self, config: SmallTransformerConfig) -> None:
        super().__init__()
        self.pad_token_id = config.pad_token_id
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.head = ClassifierHead(
            ClassifierHeadConfig(
                input_dim=config.d_model,
                hidden_dim=config.head_hidden_dim,
                output_dim=config.num_classes,
                dropout=config.dropout,
            )
        )

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        embedded = self.embedding(input_ids) + self.pos_embedding(positions)

        # Causal mask not needed — this is an encoder-only classification model.
        # Padding mask: True where padded so the transformer ignores those positions.
        pad_mask = input_ids == self.pad_token_id  # (bsz, seq_len)
        encoded = self.encoder(embedded, src_key_padding_mask=pad_mask)

        # Pool by taking the hidden state at the last real token per sequence,
        # matching the GRU baseline's "use final hidden state" strategy.
        gather_idx = (lengths - 1).clamp(min=0).long()  # (bsz,)
        last_hidden = encoded[torch.arange(bsz, device=encoded.device), gather_idx]
        return self.head(last_hidden)
