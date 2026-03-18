from __future__ import annotations

import torch


def exact_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = logits.argmax(dim=-1)
    return (predictions == targets).float().mean()


def mean_confidence(confidence: torch.Tensor) -> torch.Tensor:
    return confidence.float().mean()
