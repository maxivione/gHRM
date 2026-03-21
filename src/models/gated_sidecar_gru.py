from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.heads import ClassifierHead, ClassifierHeadConfig


@dataclass(frozen=True)
class GatedSidecarGRUConfig:
    vocab_size: int
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    ctrl_dim: int
    ctrl_layers: int
    chunk_size: int
    gate_init_bias: float
    dropout: float
    num_classes: int
    pad_token_id: int = 0
    freeze_gate: bool = False
    freeze_controller: bool = False


def _chunk_and_mean(
    hidden_states: torch.Tensor,
    lengths: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Average-pool hidden states into fixed-size chunks, masking padding.

    Args:
        hidden_states: (B, T, H) padded output from the fast GRU.
        lengths: (B,) actual sequence lengths.
        chunk_size: number of timesteps per chunk.

    Returns:
        (B, num_chunks, H) where num_chunks = ceil(T / chunk_size).
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    # Pad to exact multiple of chunk_size so reshape works cleanly
    pad_len = num_chunks * chunk_size - seq_len
    if pad_len > 0:
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad_len))

    # (B, num_chunks, chunk_size, H)
    chunked = hidden_states.reshape(batch_size, num_chunks, chunk_size, hidden_dim)

    # Build per-timestep mask: (B, T_padded) -> (B, num_chunks, chunk_size)
    timestep_indices = torch.arange(
        num_chunks * chunk_size, device=lengths.device
    ).unsqueeze(0)
    mask = (timestep_indices < lengths.unsqueeze(1)).reshape(
        batch_size, num_chunks, chunk_size
    )

    # Masked mean per chunk; clamp denominator to avoid div-by-zero on
    # fully-padded trailing chunks (those will be zero vectors, which is fine
    # because the controller GRU processes a shorter packed sequence anyway,
    # but we keep it simple with a plain tensor here).
    counts = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, num_chunks, 1)
    mask_expanded = mask.unsqueeze(-1)  # (B, num_chunks, chunk_size, 1)
    chunk_means = (chunked * mask_expanded).sum(dim=2) / counts

    return chunk_means


class GatedSidecarGRU(nn.Module):
    def __init__(self, config: GatedSidecarGRUConfig) -> None:
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

        # --- sidecar controller ---
        self.ctrl_proj = nn.Linear(config.hidden_dim, config.ctrl_dim)
        self.ctrl_gru = nn.GRU(
            input_size=config.ctrl_dim,
            hidden_size=config.ctrl_dim,
            num_layers=config.ctrl_layers,
            batch_first=True,
        )
        self.ctrl_out = nn.Linear(config.ctrl_dim, config.hidden_dim)

        # Gate: scalar per sample, init biased toward closed
        self.gate_linear = nn.Linear(config.hidden_dim + config.ctrl_dim, 1)
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

        if config.freeze_gate:
            self.gate_linear.weight.requires_grad_(False)
            self.gate_linear.bias.requires_grad_(False)
            # Force gate to ~0 by setting bias very negative
            nn.init.constant_(self.gate_linear.bias, -30.0)
            nn.init.zeros_(self.gate_linear.weight)

        if config.freeze_controller:
            for param in self.ctrl_proj.parameters():
                param.requires_grad_(False)
            for param in self.ctrl_gru.parameters():
                param.requires_grad_(False)
            for param in self.ctrl_out.parameters():
                param.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(input_ids)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        gru_out_packed, h_n = self.fast_gru(packed)
        h_T = h_n[-1]  # (B, hidden_dim)

        # Sidecar: chunk-mean summaries of fast GRU output
        gru_out_padded, _ = nn.utils.rnn.pad_packed_sequence(
            gru_out_packed, batch_first=True
        )
        chunk_means = _chunk_and_mean(
            gru_out_padded, lengths, self.config.chunk_size
        )  # (B, num_chunks, hidden_dim)

        ctrl_input = self.ctrl_proj(chunk_means)  # (B, num_chunks, ctrl_dim)
        _, c_n = self.ctrl_gru(ctrl_input)
        c_T = c_n[-1]  # (B, ctrl_dim)

        gate = torch.sigmoid(
            self.gate_linear(torch.cat([h_T, c_T], dim=-1))
        )  # (B, 1)
        fused = h_T + gate * self.ctrl_out(c_T)
        fused = self.dropout(fused)

        logits = self.head(fused)
        if return_gate:
            return logits, gate
        return logits
