from __future__ import annotations

import sys

import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baselines import FlatRecurrentBaseline, HierarchicalReasoner, ReasonerConfig


def main() -> int:
    batch_size = 4
    sequence_length = 6
    input_dim = 128
    output_dim = 16

    flat = FlatRecurrentBaseline(input_dim=input_dim, hidden_dim=256, output_dim=output_dim)
    flat_inputs = torch.randn(batch_size, sequence_length, input_dim)
    flat_outputs = flat(flat_inputs)

    hierarchical = HierarchicalReasoner(
        ReasonerConfig(
            input_dim=input_dim,
            workspace_dim=128,
            workspace_slots=8,
            hidden_dim=256,
            planner_interval=2,
            worker_steps=6,
            memory_slots=8,
            output_dim=output_dim,
        )
    )
    hierarchical_inputs = torch.randn(batch_size, input_dim)
    hierarchical_outputs = hierarchical(hierarchical_inputs)

    print("flat:", flat_outputs["answer_logits"].shape)
    print("hierarchical:", hierarchical_outputs["answer_logits"].shape)
    print("halt:", hierarchical_outputs["halt_probability"].shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
