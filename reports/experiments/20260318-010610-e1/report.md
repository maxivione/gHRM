Title: hierarchical_gru_v0 single-run report
Hypothesis: Shared E1 scaffold can complete the configured phase-0 run without procedural failures.
Cause being tested: Whether this model produces stable optimization signal on the current E1 tasks.
Models compared: hierarchical_gru_v0
Parameter counts: 1756304
Datasets: data/synthetic/e1_phase0
Train split sizes: 1024
Validation split sizes: 256
OOD split sizes: 256
Hardware: cuda
Batch size: 64
Mixed precision: True
Peak VRAM: 76.06 MB
Wall-clock per epoch: 1.11 sec
Seeds: 7
val_exact_accuracy: 0.8359
ood_exact_accuracy: 0.5430
macro_exact_accuracy: 0.8359
steps_to_best_val: 8
run_stability: instability_events=0, train_max_grad_norm=3.5675
Failure analysis: Per-task accuracies are listed below this template block.
Conclusion: Single-run artifact only; compare both models before choosing an E1 bucket.
Kill criteria met?: No
Next action: Use this run alongside the matched baseline comparison.

val_task_accuracies:
{
  "graph_shortest_path_len": 0.546875,
  "maze_path_exists": 0.828125,
  "string_rewrite_final_token": 1.0,
  "sudoku_cell_fill": 0.96875
}

ood_task_accuracies:
{
  "graph_shortest_path_len": 0.515625,
  "maze_path_exists": 0.625,
  "string_rewrite_final_token": 0.515625,
  "sudoku_cell_fill": 0.515625
}

checkpoint_path: reports\experiments\20260318-010610-e1\best.pt
