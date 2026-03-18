# gHRM-Lite

gHRM-Lite is a compact research repo for testing whether a generalized hierarchical recurrent backbone can transfer reasoning across small grid, symbolic, text, and code-style tasks on consumer hardware.

The current repo state is a bootstrap scaffold for `E1` from the master brief:

- flat recurrent baseline vs hierarchical recurrent baseline
- phase-0 synthetic tasks first
- telemetry and reproducibility before scale
- RTX 3070 8GB as the main training target

## Scope

In scope:

- small falsifiable experiments
- synthetic dataset generation
- reproducible local training and evaluation
- telemetry for VRAM, wall-clock, control events, and failure cases

Out of scope:

- chatbot finetuning
- open-web pretraining
- large multimodal systems
- scale-driven benchmark farming

## Quick Start

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
python -m src.cli check-layout
python scripts\smoke_test.py
```

## Initial Layout

```text
configs/     Reproducible model, train, and eval config files
data/        Local raw, processed, synthetic, and manifest files
datasets/    Domain-specific dataset builders and dataset assets
src/         Adapters, models, training, eval, telemetry, and task registry
scripts/     Thin operational scripts
reports/     Experiment writeups, failures, and weekly notes
notebooks/   Scratch analysis only after scripted baselines exist
```

## Phase-0 Tasks

The first experiment cycle stays narrow:

- Sudoku
- maze
- graph traversal
- string rewrite

Use `python -m src.cli print-phase0` to see the current task registry.

## Research Rule

Every experiment should produce:

1. hypothesis
2. smallest config diff
3. datasets used
4. hardware used
5. peak VRAM
6. wall-clock
7. primary metrics
8. failure analysis
9. next action
