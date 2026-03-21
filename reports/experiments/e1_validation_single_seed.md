Title: E1 phase-0 single-seed flat vs hierarchical GRU comparison
Hypothesis: The hierarchical GRU scaffold should show a clear accuracy advantage over the flat GRU baseline on the current E1 tasks.
Cause being tested: Architecture effect under the existing shared E1 optimizer, data, and evaluation scaffold.
Models compared: flat_gru_v0 vs hierarchical_gru_v0
Parameter counts: flat_gru_v0=3276304; hierarchical_gru_v0=1756304
Datasets: data/synthetic/e1_phase0 across maze_path_exists, sudoku_cell_fill, graph_shortest_path_len, string_rewrite_final_token
Train split sizes: 256 per task, 1024 total
Validation split sizes: 64 per task, 256 total
OOD split sizes: 64 per task, 256 total
Hardware: NVIDIA GeForce RTX 3070 8GB
Batch size: 64
Mixed precision: True
Peak VRAM: flat_gru_v0=134.00 MB; hierarchical_gru_v0=76.06 MB
Wall-clock per epoch: flat_gru_v0=0.33 sec mean train epoch, 8.7 sec total run; hierarchical_gru_v0=1.24 sec mean train epoch, 17.0 sec total run
Seeds: 7
val_exact_accuracy: flat_gru_v0=0.8438; hierarchical_gru_v0=0.8359
ood_exact_accuracy: flat_gru_v0=0.5469; hierarchical_gru_v0=0.5430
macro_exact_accuracy: flat_gru_v0=0.8438 val macro / 0.5469 ood macro; hierarchical_gru_v0=0.8359 val macro / 0.5430 ood macro
steps_to_best_val: flat_gru_v0=8; hierarchical_gru_v0=8
run_stability: Both runs completed 8/8 epochs on CUDA with instability_events=0; max observed train grad norm was 6.37 for flat_gru_v0 and 5.05 for hierarchical_gru_v0.
Failure analysis: The comparison does not isolate architecture cleanly because flat_gru_v0 has 1.87x more parameters, while hierarchical_gru_v0 performs extra planner-state updates every fourth token. The dataset also still has low difficulty spread on graph and a degenerate rewrite label support that only uses classes 0 and 2.
Conclusion: C) procedurally inconclusive. The final validation and OOD numbers are effectively tied, but the scaffold is not parameter-matched or recurrent-budget-matched, so this run does not support a clean architectural claim.
Kill criteria met?: No
Next action: Freeze E1 at this point, keep the small Sudoku OOD label-support fix, and make the next E1 pass parameter-matched and recurrent-budget-matched before claiming signal.

val_task_accuracies:

| task | flat_gru_v0 | hierarchical_gru_v0 |
| --- | ---: | ---: |
| maze_path_exists | 0.8125 | 0.8281 |
| sudoku_cell_fill | 1.0000 | 0.9688 |
| graph_shortest_path_len | 0.5625 | 0.5469 |
| string_rewrite_final_token | 1.0000 | 1.0000 |
| average | 0.8438 | 0.8359 |

ood_task_accuracies:

| task | flat_gru_v0 | hierarchical_gru_v0 |
| --- | ---: | ---: |
| maze_path_exists | 0.6719 | 0.6250 |
| sudoku_cell_fill | 0.7188 | 0.5156 |
| graph_shortest_path_len | 0.2969 | 0.5156 |
| string_rewrite_final_token | 0.5000 | 0.5156 |
| average | 0.5469 | 0.5430 |

artifacts:

- flat_gru_v0 summary: reports/experiments/20260318-010556-e1/summary.json
- flat_gru_v0 report: reports/experiments/20260318-010556-e1/report.md
- flat_gru_v0 checkpoint: reports/experiments/20260318-010556-e1/best.pt
- hierarchical_gru_v0 summary: reports/experiments/20260318-010610-e1/summary.json
- hierarchical_gru_v0 report: reports/experiments/20260318-010610-e1/report.md
- hierarchical_gru_v0 checkpoint: reports/experiments/20260318-010610-e1/best.pt
