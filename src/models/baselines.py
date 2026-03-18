from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.adapters.grids import GridAdapter, GridAdapterConfig
from src.models.halting import HaltingHead
from src.models.heads import OutputHeads
from src.models.memory import MemoryConfig, MemoryController
from src.models.planner import SlowPlanner
from src.models.worker import FastWorker
from src.models.workspace import SharedWorkspace, WorkspaceConfig


@dataclass(frozen=True)
class ReasonerConfig:
    input_dim: int
    workspace_dim: int
    workspace_slots: int
    hidden_dim: int
    planner_interval: int
    worker_steps: int
    memory_slots: int
    output_dim: int


class FlatRecurrentBaseline(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.recurrent = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.heads = OutputHeads(hidden_dim, output_dim)

    def forward(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        _, hidden_state = self.recurrent(sequence)
        return self.heads(hidden_state[-1])


class HierarchicalReasoner(nn.Module):
    def __init__(self, config: ReasonerConfig) -> None:
        super().__init__()
        self.config = config
        self.adapter = GridAdapter(
            GridAdapterConfig(input_dim=config.input_dim, workspace_dim=config.workspace_dim)
        )
        self.workspace = SharedWorkspace(
            WorkspaceConfig(slot_count=config.workspace_slots, slot_dim=config.workspace_dim)
        )
        self.planner = SlowPlanner(config.workspace_dim, config.hidden_dim)
        self.worker = FastWorker(config.workspace_dim + config.hidden_dim, config.hidden_dim)
        self.memory = MemoryController(
            MemoryConfig(slot_count=config.memory_slots, slot_dim=config.hidden_dim)
        )
        self.halting = HaltingHead(config.hidden_dim)
        self.heads = OutputHeads(config.hidden_dim, config.output_dim)
        self.read_projection = nn.Linear(config.hidden_dim, config.workspace_dim)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = inputs.shape[0]
        device = inputs.device

        encoded_input = self.adapter(inputs)
        workspace = self.workspace.initial_state(batch_size=batch_size, device=device)
        workspace = self.workspace.inject(workspace, encoded_input)

        planner_state = torch.zeros(batch_size, self.config.hidden_dim, device=device)
        worker_state = torch.zeros(batch_size, self.config.hidden_dim, device=device)
        memory_state = self.memory.empty_state(batch_size=batch_size, device=device)

        for step in range(self.config.worker_steps):
            workspace_summary = workspace.slots.mean(dim=1)

            if step % self.config.planner_interval == 0:
                planner_state = self.planner(workspace_summary, planner_state)

            worker_input = torch.cat([workspace_summary, planner_state], dim=-1)
            worker_state = self.worker(worker_input, worker_state)

            memory_read = self.memory.read(worker_state, memory_state)
            memory_state = self.memory.write(worker_state, memory_state)
            workspace.slots[:, 0, :] = workspace.slots[:, 0, :] + self.read_projection(memory_read)

        outputs = self.heads(worker_state)
        outputs["halt_probability"] = self.halting(worker_state)
        outputs["memory_state"] = memory_state
        return outputs
