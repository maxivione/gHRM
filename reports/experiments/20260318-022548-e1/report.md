Title: gated_sidecar_gru_v0 single-run report
Hypothesis: Shared E1 scaffold can complete the configured phase-0 run without procedural failures.
Cause being tested: Whether this model produces stable optimization signal on the current E1 tasks.
Models compared: gated_sidecar_gru_v0
Parameter counts: 3367953
Datasets: data/synthetic/e1_phase0
Train split sizes: 768
Validation split sizes: 192
OOD split sizes: 192
Hardware: cpu
Batch size: 64
Mixed precision: False
Peak VRAM: 0.00 MB
Wall-clock per epoch: 5.11 sec
Seeds: 42
val_exact_accuracy: 0.8802
ood_exact_accuracy: 0.6458
macro_exact_accuracy: 0.8802
steps_to_best_val: 23
run_stability: instability_events=0, train_max_grad_norm=3.4431
Failure analysis: Per-task accuracies are listed below this template block.
Conclusion: Single-run artifact only; compare both models before choosing an E1 bucket.
Kill criteria met?: No
Next action: Use this run alongside the matched baseline comparison.

val_task_accuracies:
{
  "graph_shortest_path_len": 0.8125,
  "maze_path_exists": 0.828125,
  "sudoku_cell_fill": 1.0
}

ood_task_accuracies:
{
  "graph_shortest_path_len": 0.484375,
  "maze_path_exists": 0.609375,
  "sudoku_cell_fill": 0.84375
}

checkpoint_path: reports\experiments\20260318-022548-e1\best.pt
