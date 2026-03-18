# gHRM-Lite Agent Notes

This repo exists to test a compact generalized hierarchical recurrent model, not to grow a general-purpose ML framework.

## Default Working Rules

- Patch only the modules touched by the current experiment.
- Prefer the smallest falsifiable experiment over a cleaner abstraction.
- Add telemetry before adding model complexity.
- Keep domain adapters small and keep gains attributable to the shared backbone.
- Prefer synthetic tasks and local reproducibility over larger datasets.
- Do not start the next experiment until the current one is written up in `reports/experiments/`.

## Baseline Gate

No architecture claim is valid without fair comparisons against:

1. flat recurrent baseline
2. small transformer baseline
3. no-memory hierarchical baseline
4. no-halting hierarchical baseline

## Experiment Writeup Contract

Every run note should include:

```text
Title:
Hypothesis:
Cause being tested:
Config diff:
Datasets:
Hardware:
Peak VRAM:
Wall-clock:
Primary metrics:
Failure analysis:
Conclusion:
Next action:
```

## Non-Goals

Do not widen the repo into:

- chatbot product work
- open-web pretraining
- large retrieval systems
- RL-heavy loops
- speculative subsystems without an ablation path
