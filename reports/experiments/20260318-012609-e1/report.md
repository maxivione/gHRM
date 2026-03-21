Title: hierarchical_gru_v0 single-run report
Hypothesis: Shared E1 scaffold can complete the configured phase-0 run without procedural failures.
Cause being tested: Whether this model produces stable optimization signal on the current E1 tasks.
Models compared: hierarchical_gru_v0
Parameter counts: 3209232
Datasets: data/synthetic/e1_phase0
Train split sizes: 768
Validation split sizes: 192
OOD split sizes: 192
Hardware: cuda
Batch size: 64
Mixed precision: False
Peak VRAM: 148.53 MB
Wall-clock per epoch: 0.64 sec
Seeds: 7
val_exact_accuracy: 0.8854
ood_exact_accuracy: 0.5312
macro_exact_accuracy: 0.8854
steps_to_best_val: 22
run_stability: instability_events=0, train_max_grad_norm=4.4631
Failure analysis: Per-task accuracies are listed below this template block.
Conclusion: Single-run artifact only; compare both models before choosing an E1 bucket.
Kill criteria met?: No
Next action: Use this run alongside the matched baseline comparison.

val_task_accuracies:
{
  "graph_shortest_path_len": 0.796875,
  "maze_path_exists": 0.859375,
  "sudoku_cell_fill": 1.0
}

ood_task_accuracies:
{
  "graph_shortest_path_len": 0.28125,
  "maze_path_exists": 0.625,
  "sudoku_cell_fill": 0.6875
}

checkpoint_path: reports\experiments\20260318-012609-e1\best.pt
