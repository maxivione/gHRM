"""Causal HRM — Hierarchical Reasoning Model for autoregressive language modeling.

Preserves the core HRM mechanics (H-level / L-level reasoning, ACT halting)
while adapting for causal text generation with GPT-2 tokenizer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class CausalHRMConfig:
    vocab_size: int = 50257        # GPT-2 tokenizer
    hidden_dim: int = 512
    num_heads: int = 8
    num_h_layers: int = 2          # H-level (slow, high-level reasoning)
    num_l_layers: int = 2          # L-level (fast, local patterns)
    mlp_expansion: float = 2.667   # SwiGLU effective expansion
    max_seq_len: int = 512
    max_act_steps: int = 8         # ACT: max reasoning iterations
    act_explore_prob: float = 0.1  # ACT: exploration probability during training
    rope_theta: float = 10000.0
    dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    unet_skip_scale: float = 0.1  # U-Net skip connection scale
    tie_weights: bool = True       # tie embedding and lm_head weights


def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.max_len = max_len
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len: int):
        if self._cos_cached is None or seq_len > self._cos_cached.shape[0]:
            t = torch.arange(max(seq_len, self.max_len), device=self.inv_freq.device)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class CausalSelfAttention(nn.Module):
    def __init__(self, config: CausalHRMConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, nh, hd)
        q, k = q.transpose(1, 2), k.transpose(1, 2)  # (B, nh, T, hd)
        v = v.transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=self.attn_dropout.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, expansion: float):
        super().__init__()
        inner = int(hidden_dim * expansion)
        self.w1 = nn.Linear(hidden_dim, inner, bias=False)
        self.w2 = nn.Linear(inner, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, inner, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class HRMBlock(nn.Module):
    """Single transformer block used in both H-level and L-level."""
    def __init__(self, config: CausalHRMConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = SwiGLU(config.hidden_dim, config.mlp_expansion)
        self.eps = config.rms_norm_eps

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Post-norm (matching HRM-official)
        x = rms_norm(x + self.attn(x, cos, sin), self.eps)
        x = rms_norm(x + self.mlp(x), self.eps)
        return x


class ReasoningLevel(nn.Module):
    """A stack of HRMBlocks forming one reasoning level (H or L)."""
    def __init__(self, config: CausalHRMConfig, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([HRMBlock(config) for _ in range(num_layers)])
        self.skip_scale = config.unet_skip_scale

    def forward(self, z: torch.Tensor, injection: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        z = z + injection
        n = len(self.layers)
        encoder_outs = []
        for i, layer in enumerate(self.layers):
            if i < n // 2:
                encoder_outs.append(z)
            z = layer(z, cos, sin)
            # U-Net skip: connect encoder layer i to decoder layer (n-1-i)
            decoder_idx = n - 1 - i
            if decoder_idx < len(encoder_outs) and i >= n // 2:
                skip_from = encoder_outs[decoder_idx]
                z = z + self.skip_scale * skip_from
        return z


class CausalHRM(nn.Module):
    """
    Causal Hierarchical Reasoning Model.

    Architecture:
    - Token embedding + RoPE
    - ACT loop:
        - L-level: fine-grained local reasoning (causal attention)
        - H-level: high-level structural reasoning (causal attention)
        - Q-head: decides whether to halt reasoning
    - LM head: next-token prediction from H-level output
    """
    def __init__(self, config: CausalHRMConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.embed_scale = math.sqrt(config.hidden_dim)
        self.embed_drop = nn.Dropout(config.dropout)

        self.rope = RotaryEmbedding(
            dim=config.hidden_dim // config.num_heads,
            max_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        self.h_level = ReasoningLevel(config, config.num_h_layers)
        self.l_level = ReasoningLevel(config, config.num_l_layers)

        # Learned initial states for H and L
        self.h_init = nn.Parameter(torch.randn(config.hidden_dim) * 0.02)
        self.l_init = nn.Parameter(torch.randn(config.hidden_dim) * 0.02)

        # LM output head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.embed_tokens.weight

        # ACT Q-head: 2 outputs (q_halt, q_continue)
        self.q_head = nn.Linear(config.hidden_dim, 2, bias=True)
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, -5.0)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.q_head:
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,         # (B, T)
        labels: Optional[torch.Tensor] = None,  # (B, T)
    ) -> dict:
        B, T = input_ids.shape
        device = input_ids.device

        # Embed
        x = self.embed_tokens(input_ids) * self.embed_scale
        x = self.embed_drop(x)

        cos, sin = self.rope(T)

        # Init H and L states
        z_H = self.h_init.unsqueeze(0).unsqueeze(0).expand(B, T, -1).clone()
        z_L = self.l_init.unsqueeze(0).unsqueeze(0).expand(B, T, -1).clone()

        # ACT loop — all steps without grad except the last one
        actual_steps = self.config.max_act_steps
        halted = torch.zeros(B, dtype=torch.bool, device=device)

        with torch.no_grad():
            for step in range(self.config.max_act_steps - 1):
                z_L = self.l_level(z_L, z_H + x, cos, sin)
                z_H = self.h_level(z_H, z_L, cos, sin)

                # Check halting (training only, eval always uses max steps for consistency)
                if self.training and self.config.max_act_steps > 1:
                    q_logits = self.q_head(z_H[:, 0])  # (B, 2) from first position
                    should_halt = q_logits[:, 0] > q_logits[:, 1]  # halt > continue

                    # Exploration
                    explore = torch.rand(B, device=device) < self.config.act_explore_prob
                    min_steps = torch.randint(2, self.config.max_act_steps + 1, (B,), device=device)
                    should_halt = should_halt & ~explore & (torch.tensor(step + 1, device=device) >= min_steps)

                    halted = halted | should_halt
                    if halted.all():
                        actual_steps = step + 1
                        break

        # Final step with gradients
        z_L = self.l_level(z_L, z_H + x, cos, sin)
        z_H = self.h_level(z_H, z_L, cos, sin)

        # LM output
        logits = self.lm_head(z_H)  # (B, T, vocab_size)

        # Q-head for ACT loss
        q_logits = self.q_head(z_H[:, 0])
        q_halt = q_logits[:, 0]
        q_continue = q_logits[:, 1]

        out = {
            'logits': logits,
            'q_halt': q_halt,
            'q_continue': q_continue,
            'act_steps': actual_steps,
        }

        if labels is not None:
            # Shift for next-token prediction: logits[:-1] predicts labels[1:]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            out['loss'] = loss

        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,     # (1, T) prompt
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Autoregressive text generation."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = input_ids[:, -self.config.max_seq_len:]

            out = self.forward(idx_cond)
            logits = out['logits'][:, -1, :]  # last position

            # Temperature
            logits = logits / temperature

            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

            # Stop on EOS (GPT-2 uses token 50256)
            if next_id.item() == 50256:
                break

        return input_ids
