from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.heads import ClassifierHead, ClassifierHeadConfig


@dataclass(frozen=True)
class BoundaryHRMConfig:
    vocab_size: int
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    ctrl_dim: int
    halt_bias: float
    ponder_coeff: float
    dropout: float
    num_classes: int
    pad_token_id: int = 0


class BoundaryHRM(nn.Module):
    """Fused GRU worker + post-hoc learned boundary detection + controller GRU.

    Phase 1: standard packed nn.GRU (cuDNN fused, identical to flat GRU).
    Phase 2: boundary scorer runs on hidden states, ACT-style accumulation
    emits variable-length segment representations, controller GRUCell
    processes segments, gated residual fuses controller output into h_T.
    """

    def __init__(self, config: BoundaryHRMConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_token_id,
        )

        recurrent_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.worker = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )

        self.boundary_scorer = nn.Linear(config.hidden_dim, 1)
        nn.init.constant_(self.boundary_scorer.bias, config.halt_bias)

        self.seg_proj = nn.Linear(config.hidden_dim, config.ctrl_dim)
        self.controller = nn.GRUCell(config.ctrl_dim, config.ctrl_dim)
        self.ctrl_out = nn.Linear(config.ctrl_dim, config.hidden_dim)
        self.gate_linear = nn.Linear(config.hidden_dim + config.ctrl_dim, 1)

        self.dropout = nn.Dropout(config.dropout)
        self.head = ClassifierHead(
            ClassifierHeadConfig(
                input_dim=config.hidden_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.num_classes,
                dropout=config.dropout,
            )
        )

        # Stored after each forward for the training loop to read.
        self._last_ponder_cost: torch.Tensor = torch.tensor(0.0)
        # Boundary telemetry: populated during training forward passes.
        self._last_boundary_telemetry: dict[str, float] = {}

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        batch_size, max_len, _ = embedded.shape
        device = embedded.device

        # Phase 1: fused worker GRU (identical to flat GRU)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        gru_out_packed, h_n = self.worker(packed)
        h_T = h_n[-1]  # (B, hidden_dim)

        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            gru_out_packed, batch_first=True,
        )  # (B, T, hidden_dim)

        # Phase 2: boundary detection + controller (post-hoc)
        halt_logits = self.boundary_scorer(gru_out).squeeze(-1)  # (B, T)

        # Mask padded positions to large negative so sigmoid → ~0
        timesteps = torch.arange(max_len, device=device).unsqueeze(0)  # (1, T)
        pad_mask = timesteps >= lengths.unsqueeze(1)  # (B, T)
        halt_logits = halt_logits.masked_fill(pad_mask, -20.0)
        halt_probs = torch.sigmoid(halt_logits)  # (B, T)

        # ACT accumulation + segment emission (per-sequence loop on scalars)
        ctrl_states = torch.zeros(batch_size, self.config.ctrl_dim, device=device)
        ponder_cost = torch.zeros(batch_size, device=device)

        # Track telemetry
        segment_counts = torch.zeros(batch_size, device=device)

        # Accumulator for segment-weighted hidden states and halt mass
        seg_accum = torch.zeros(batch_size, self.config.hidden_dim, device=device)
        halt_accum = torch.zeros(batch_size, device=device)

        for t in range(max_len):
            active = ~pad_mask[:, t]  # (B,) bool
            if not active.any():
                break

            h_t = halt_probs[:, t]  # (B,)
            hidden_t = gru_out[:, t, :]  # (B, hidden_dim)

            # Accumulate weighted hidden state and halt mass
            seg_accum = seg_accum + h_t.unsqueeze(-1) * hidden_t
            halt_accum = halt_accum + h_t
            ponder_cost = ponder_cost + h_t * active.float()

            # Fire boundary where accumulated halt mass >= 1.0
            fire = (halt_accum >= 1.0) & active  # (B,)
            if fire.any():
                # Normalise segment representation by halt mass
                seg_repr = seg_accum[fire] / halt_accum[fire].unsqueeze(-1).clamp(min=1e-6)
                seg_input = self.seg_proj(seg_repr)  # (fired, ctrl_dim)
                ctrl_states[fire] = self.controller(seg_input, ctrl_states[fire])
                segment_counts[fire] += 1

                # Reset accumulators for fired sequences
                seg_accum[fire] = 0.0
                halt_accum[fire] -= 1.0  # carry remainder

        # Handle trailing segment (anything left in accumulator at sequence end)
        has_remainder = halt_accum > 0.01
        if has_remainder.any():
            seg_repr = seg_accum[has_remainder] / halt_accum[has_remainder].unsqueeze(-1).clamp(min=1e-6)
            seg_input = self.seg_proj(seg_repr)
            ctrl_states[has_remainder] = self.controller(seg_input, ctrl_states[has_remainder])
            segment_counts[has_remainder] += 1

        # Gated fusion: h_T + gate * ctrl_proj(c_T)
        c_T = ctrl_states  # (B, ctrl_dim)
        gate = torch.sigmoid(self.gate_linear(torch.cat([h_T, c_T], dim=-1)))  # (B, 1)
        fused = h_T + gate * self.ctrl_out(c_T)
        fused = self.dropout(fused)

        # Store ponder cost for training loop
        self._last_ponder_cost = ponder_cost.mean()

        # Boundary telemetry (detached, cheap)
        with torch.no_grad():
            valid_halt = halt_probs.masked_fill(pad_mask, 0.0)
            valid_count = (~pad_mask).sum().float().clamp(min=1.0)
            self._last_boundary_telemetry = {
                "halt_prob_mean": (valid_halt.sum() / valid_count).item(),
                "halt_prob_std": valid_halt[~pad_mask].std().item() if (~pad_mask).sum() > 1 else 0.0,
                "segments_per_seq_mean": segment_counts.mean().item(),
                "segments_per_seq_std": segment_counts.std().item(),
                "ponder_cost_mean": ponder_cost.mean().item(),
                "gate_mean": gate.mean().item(),
            }

        return self.head(fused)
