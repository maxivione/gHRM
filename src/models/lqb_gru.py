from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from src.models.heads import ClassifierHead, ClassifierHeadConfig


@dataclass(frozen=True)
class LQBConfig:
    vocab_size: int
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    d_attn: int
    num_queries: int
    gate_init_bias: float
    dropout: float
    num_classes: int
    pad_token_id: int = 0
    freeze_queries: bool = False


class LQBModel(nn.Module):
    """Learned Query Bottleneck: fast packed GRU + cross-attention summary + gated residual."""

    def __init__(self, config: LQBConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_token_id,
        )

        recurrent_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.fast_gru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )

        # Cross-attention bottleneck
        self.proj_kv = nn.Linear(config.hidden_dim, config.d_attn)
        self.queries = nn.Parameter(torch.randn(config.num_queries, config.d_attn))
        nn.init.normal_(self.queries, std=1.0 / math.sqrt(config.d_attn))
        self.proj_out = nn.Linear(config.d_attn, config.hidden_dim)

        # Gate
        self.gate_linear = nn.Linear(config.hidden_dim + config.d_attn, 1)
        nn.init.constant_(self.gate_linear.bias, config.gate_init_bias)

        self.dropout = nn.Dropout(config.dropout)

        self.head = ClassifierHead(
            ClassifierHeadConfig(
                input_dim=config.hidden_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.num_classes,
                dropout=config.dropout,
            )
        )

        if config.freeze_queries:
            self.queries.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        embedded = self.embedding(input_ids)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        gru_out_packed, h_n = self.fast_gru(packed)
        h_T = h_n[-1]  # (B, hidden_dim)

        # Unpack for cross-attention
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            gru_out_packed, batch_first=True
        )  # (B, T, hidden_dim)
        batch_size, seq_len, _ = gru_out.shape

        # Project to attention dim: keys and values
        kv = self.proj_kv(gru_out)  # (B, T, d_attn)

        # Queries: (K, d_attn) → (1, K, d_attn) → broadcast to (B, K, d_attn)
        Q = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Attention scores: (B, K, d_attn) @ (B, d_attn, T) → (B, K, T)
        scale = math.sqrt(self.config.d_attn)
        attn_scores = torch.bmm(Q, kv.transpose(1, 2)) / scale

        # Mask padding positions to -inf
        timesteps = torch.arange(seq_len, device=lengths.device).unsqueeze(0)
        pad_mask = timesteps >= lengths.unsqueeze(1)  # (B, T) True where padded
        attn_scores.masked_fill_(pad_mask.unsqueeze(1), float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, K, T)

        # Weighted sum of values: (B, K, T) @ (B, T, d_attn) → (B, K, d_attn)
        attn_out = torch.bmm(attn_weights, kv)

        # Mean-pool across queries → (B, d_attn)
        summary = attn_out.mean(dim=1)

        # Gated residual
        gate = torch.sigmoid(
            self.gate_linear(torch.cat([h_T, summary], dim=-1))
        )  # (B, 1)
        fused = h_T + gate * self.proj_out(summary)
        fused = self.dropout(fused)

        logits = self.head(fused)

        if return_diagnostics:
            # Attention entropy per query (bits)
            # Clamp to avoid log(0) on fully-masked positions
            log_weights = torch.log(attn_weights.clamp(min=1e-12))
            entropy = -(attn_weights * log_weights).sum(dim=-1)  # (B, K)
            return logits, {
                "gate": gate,
                "attn_weights": attn_weights,
                "attn_entropy": entropy,
            }
        return logits
