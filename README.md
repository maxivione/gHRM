# gHRM-Lite

gHRM-Lite is a compact research repo for one falsifiable experiment first:

- `E1`: flat recurrent baseline vs hierarchical recurrent baseline
- four tiny synthetic phase-0 tasks
- reproducible local configs
- wall-clock and VRAM telemetry on a single RTX 3070 8GB

Everything outside E1 is intentionally deferred.

## Scope

In scope:

- flat GRU baseline
- hierarchical GRU baseline
- maze, Sudoku-style fill, graph shortest path, and string rewrite tasks
- exact accuracy, OOD accuracy, VRAM, and wall-clock logging

Out of scope:

- memory modules
- text adapters
- code adapters
- MoE
- RL loops
- full LM work

## Quick Start

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
python scripts\generate_phase0.py
python scripts\run_e1.py --model-config configs\model\flat_gru_v0.toml --train-config configs\train\e1_local_3070.toml --eval-config configs\eval\e1_phase0.toml
```

## E1 Layout

```text
configs/       Reproducible model, train, and eval config files
data/          Dataset manifest and generated synthetic data
datasets/      Tiny phase-0 task generators
src/models/    Flat and hierarchical recurrent baselines only
src/training/  One E1 train/eval path
src/eval/      Exact accuracy and aggregate metrics
src/telemetry/ Wall-clock and peak VRAM logging
scripts/       Dataset generation and E1 run entrypoints
reports/       One report template for E1 writeups
```

## Phase-0 Tasks

- `maze_path_exists`
- `sudoku_cell_fill`
- `graph_shortest_path_len`
- `string_rewrite_final_token`

Use `python -m src.cli print-phase0` to see the current task registry.

## Research Rule

E1 succeeds only if the hierarchical baseline beats the flat baseline by a real margin under matched scale and still fits the 3070 budget.
