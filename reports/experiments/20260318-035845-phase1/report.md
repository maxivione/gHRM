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
Wall-clock per epoch: 2.41 sec
Seeds: 42
val_exact_accuracy: 0.7559
ood_exact_accuracy: 0.4170
macro_exact_accuracy: 0.7559
steps_to_best_val: 11
run_stability: instability_events=0, train_max_grad_norm=3.8109
Failure analysis: Per-task accuracies are listed below this template block.
Conclusion: Single-run artifact only; compare both models before choosing an E1 bucket.
Kill criteria met?: No
Next action: Use this run alongside the matched baseline comparison.

val_task_accuracies:
{
  "graph_waypoint": 0.69921875,
  "nested_arith": 0.640625,
  "register_machine": 0.68359375,
  "segment_match": 1.0
}

ood_task_accuracies:
{
  "graph_waypoint": 0.0546875,
  "nested_arith": 0.37109375,
  "register_machine": 0.51953125,
  "segment_match": 0.72265625
}

checkpoint_path: reports\experiments\20260318-035845-phase1\best.pt
