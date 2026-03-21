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
Peak VRAM: 158.68 MB
Wall-clock per epoch: 2.11 sec
Seeds: 7
val_exact_accuracy: 0.8184
ood_exact_accuracy: 0.4062
macro_exact_accuracy: 0.8184
steps_to_best_val: 30
run_stability: instability_events=0, train_max_grad_norm=3.0244
Failure analysis: Per-task accuracies are listed below this template block.
Conclusion: Single-run artifact only; compare both models before choosing an E1 bucket.
Kill criteria met?: No
Next action: Use this run alongside the matched baseline comparison.

val_task_accuracies:
{
  "graph_waypoint": 0.7578125,
  "nested_arith": 0.765625,
  "register_machine": 0.75,
  "segment_match": 1.0
}

ood_task_accuracies:
{
  "graph_waypoint": 0.05859375,
  "nested_arith": 0.4765625,
  "register_machine": 0.54296875,
  "segment_match": 0.546875
}

checkpoint_path: reports\experiments\20260318-034200-phase1\best.pt
