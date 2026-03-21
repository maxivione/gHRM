Title: small_transformer_v0 single-run report
Hypothesis: Shared E1 scaffold can complete the configured phase-0 run without procedural failures.
Cause being tested: Whether this model produces stable optimization signal on the current E1 tasks.
Models compared: small_transformer_v0
Parameter counts: 3214096
Datasets: data/synthetic/phase1
Train split sizes: 8192
Validation split sizes: 1024
OOD split sizes: 1024
Hardware: cuda
Batch size: 64
Mixed precision: False
Peak VRAM: 253.92 MB
Wall-clock per epoch: 2.32 sec
Seeds: 123
val_exact_accuracy: 0.7500
ood_exact_accuracy: 0.3008
macro_exact_accuracy: 0.7500
steps_to_best_val: 17
run_stability: instability_events=0, train_max_grad_norm=4.8665
Failure analysis: Per-task accuracies are listed below this template block.
Conclusion: Single-run artifact only; compare both models before choosing an E1 bucket.
Kill criteria met?: No
Next action: Use this run alongside the matched baseline comparison.

val_task_accuracies:
{
  "graph_waypoint": 0.6875,
  "nested_arith": 0.640625,
  "register_machine": 0.671875,
  "segment_match": 1.0
}

ood_task_accuracies:
{
  "graph_waypoint": 0.05078125,
  "nested_arith": 0.3828125,
  "register_machine": 0.48828125,
  "segment_match": 0.28125
}

checkpoint_path: reports\experiments\20260318-035927-phase1\best.pt
