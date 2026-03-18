from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    domain: str
    phase: str
    description: str


PHASE0_TASKS = [
    TaskSpec(
        name="sudoku",
        domain="grids",
        phase="phase0",
        description="Constraint completion baseline for controlled recurrent reasoning.",
    ),
    TaskSpec(
        name="maze",
        domain="grids",
        phase="phase0",
        description="Pathfinding with local structure and stepwise planning pressure.",
    ),
    TaskSpec(
        name="graph_traversal",
        domain="symbolic",
        phase="phase0",
        description="Shortest-path and reachability microtasks over small graphs.",
    ),
    TaskSpec(
        name="string_rewrite",
        domain="symbolic",
        phase="phase0",
        description="Rule-following symbolic edits with compact state transitions.",
    ),
]
