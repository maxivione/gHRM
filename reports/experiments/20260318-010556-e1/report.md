Title: flat_gru_v0 single-run report
Hypothesis: Shared E1 scaffold can complete the configured phase-0 run without procedural failures.
Cause being tested: Whether this model produces stable optimization signal on the current E1 tasks.
Models compared: flat_gru_v0
Parameter counts: 3276304
Datasets: data/synthetic/e1_phase0
Train split sizes: 1024
Validation split sizes: 256
OOD split sizes: 256
Hardware: cuda
Batch size: 64
Mixed precision: True
Peak VRAM: 134.00 MB
Wall-clock per epoch: 0.26 sec
Seeds: 7
val_exact_accuracy: 0.8438
ood_exact_accuracy: 0.5469
macro_exact_accuracy: 0.8438
steps_to_best_val: 8
run_stability: instability_events=0, train_max_grad_norm=4.2972
Failure analysis: Per-task accuracies are listed below this template block.
Conclusion: Single-run artifact only; compare both models before choosing an E1 bucket.
Kill criteria met?: No
Next action: Use this run alongside the matched baseline comparison.

val_task_accuracies:
{
  "graph_shortest_path_len": 0.5625,
  "maze_path_exists": 0.8125,
  "string_rewrite_final_token": 1.0,
  "sudoku_cell_fill": 1.0
}

ood_task_accuracies:
{
  "graph_shortest_path_len": 0.296875,
  "maze_path_exists": 0.671875,
  "string_rewrite_final_token": 0.5,
  "sudoku_cell_fill": 0.71875
}

checkpoint_path: reports\experiments\20260318-010556-e1\best.pt
