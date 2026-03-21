from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from datasets.phase0.graph import generate_graph_shortest_path_examples
from datasets.phase0.maze import generate_maze_examples
from datasets.phase0.sudoku import generate_sudoku_examples
from datasets.phase1.nested_arith import generate_nested_arith_examples
from datasets.phase1.graph_waypoint import generate_graph_waypoint_examples
from datasets.phase1.register_machine import generate_register_machine_examples
from datasets.phase1.segment_match import generate_segment_match_examples


GLOBAL_VOCAB_SIZE = 128
GLOBAL_NUM_CLASSES = 16


@dataclass(frozen=True)
class TaskSpec:
    name: str
    domain: str
    phase: str
    target_description: str
    generator: Callable[[int, int, bool], list[dict]]


E1_TASKS = [
    TaskSpec(
        name="maze_path_exists",
        domain="grids",
        phase="phase0",
        target_description="Binary answer for whether a path exists from start to goal.",
        generator=generate_maze_examples,
    ),
    TaskSpec(
        name="sudoku_cell_fill",
        domain="grids",
        phase="phase0",
        target_description="Masked-cell digit prediction on small Sudoku-style boards.",
        generator=generate_sudoku_examples,
    ),
    TaskSpec(
        name="graph_shortest_path_len",
        domain="symbolic",
        phase="phase0",
        target_description="Shortest-path length class between two nodes.",
        generator=generate_graph_shortest_path_examples,
    ),
]

PHASE1_TASKS = [
    TaskSpec(
        name="nested_arith",
        domain="compositional",
        phase="phase1",
        target_description="Evaluate nested arithmetic expression (result 0-15).",
        generator=generate_nested_arith_examples,
    ),
    TaskSpec(
        name="graph_waypoint",
        domain="compositional",
        phase="phase1",
        target_description="Shortest path through mandatory waypoints (total length 0-15).",
        generator=generate_graph_waypoint_examples,
    ),
    TaskSpec(
        name="register_machine",
        domain="sequential",
        phase="phase1",
        target_description="Final register value after a sequence of SET/ADD/SUB/SWAP ops (0-15).",
        generator=generate_register_machine_examples,
    ),
    TaskSpec(
        name="segment_match",
        domain="compositional",
        phase="phase1",
        target_description="Binary: do labelled segments satisfy cross-segment predicates?",
        generator=generate_segment_match_examples,
    ),
]

TASK_BY_NAME = {task.name: task for task in E1_TASKS + PHASE1_TASKS}

