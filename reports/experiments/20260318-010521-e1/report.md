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
Wall-clock per epoch: 1.42 sec
Seeds: 7
val_exact_accuracy: 0.4922
ood_exact_accuracy: 0.4648
macro_exact_accuracy: 0.4922
steps_to_best_val: 1
run_stability: instability_events=0, train_max_grad_norm=4.1863
Failure analysis: Per-task accuracies are listed below this template block.
Conclusion: Single-run artifact only; compare both models before choosing an E1 bucket.
Kill criteria met?: No
Next action: Use this run alongside the matched baseline comparison.

val_task_accuracies:
{
  "graph_shortest_path_len": 0.34375,
  "macro": 0.4921875,
  "maze_path_exists": 0.796875,
  "string_rewrite_final_token": 0.484375,
  "sudoku_cell_fill": 0.34375
}

ood_task_accuracies:
{
  "graph_shortest_path_len": 0.46875,
  "macro": 0.46484375,
  "maze_path_exists": 0.515625,
  "string_rewrite_final_token": 0.546875,
  "sudoku_cell_fill": 0.328125
}

checkpoint_path: reports\experiments\20260318-010521-e1\best.pt
