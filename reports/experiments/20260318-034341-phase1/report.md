Title: flat_gru_v0 single-run report
Hypothesis: Shared E1 scaffold can complete the configured phase-0 run without procedural failures.
Cause being tested: Whether this model produces stable optimization signal on the current E1 tasks.
Models compared: flat_gru_v0
Parameter counts: 3276304
Datasets: data/synthetic/phase1
Train split sizes: 8192
Validation split sizes: 1024
OOD split sizes: 1024
Hardware: cuda
Batch size: 64
Mixed precision: False
Peak VRAM: 153.48 MB
Wall-clock per epoch: 1.98 sec
Seeds: 42
val_exact_accuracy: 0.8164
ood_exact_accuracy: 0.3857
macro_exact_accuracy: 0.8164
steps_to_best_val: 27
run_stability: instability_events=0, train_max_grad_norm=2.8886
Failure analysis: Per-task accuracies are listed below this template block.
Conclusion: Single-run artifact only; compare both models before choosing an E1 bucket.
Kill criteria met?: No
Next action: Use this run alongside the matched baseline comparison.

val_task_accuracies:
{
  "graph_waypoint": 0.75390625,
  "nested_arith": 0.77734375,
  "register_machine": 0.73828125,
  "segment_match": 0.99609375
}

ood_task_accuracies:
{
  "graph_waypoint": 0.0703125,
  "nested_arith": 0.4609375,
  "register_machine": 0.51171875,
  "segment_match": 0.5
}

checkpoint_path: reports\experiments\20260318-034341-phase1\best.pt
