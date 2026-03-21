Title: flat_gru_v0 single-run report
Hypothesis: Shared E1 scaffold can complete the configured phase-0 run without procedural failures.
Cause being tested: Whether this model produces stable optimization signal on the current E1 tasks.
Models compared: flat_gru_v0
Parameter counts: 3276304
Datasets: data/synthetic/e1_phase0
Train split sizes: 768
Validation split sizes: 192
OOD split sizes: 192
Hardware: cuda
Batch size: 64
Mixed precision: False
Peak VRAM: 155.50 MB
Wall-clock per epoch: 0.21 sec
Seeds: 7
val_exact_accuracy: 0.8802
ood_exact_accuracy: 0.5208
macro_exact_accuracy: 0.8802
steps_to_best_val: 23
run_stability: instability_events=0, train_max_grad_norm=3.8765
Failure analysis: Per-task accuracies are listed below this template block.
Conclusion: Single-run artifact only; compare both models before choosing an E1 bucket.
Kill criteria met?: No
Next action: Use this run alongside the matched baseline comparison.

val_task_accuracies:
{
  "graph_shortest_path_len": 0.84375,
  "maze_path_exists": 0.796875,
  "sudoku_cell_fill": 1.0
}

ood_task_accuracies:
{
  "graph_shortest_path_len": 0.390625,
  "maze_path_exists": 0.609375,
  "sudoku_cell_fill": 0.5625
}

checkpoint_path: reports\experiments\20260318-012643-e1\best.pt
