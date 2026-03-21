Title: flat_gru_v0 single-run report
Hypothesis: Shared E1 scaffold can complete the configured phase-0 run without procedural failures.
Cause being tested: Whether this model produces stable optimization signal on the current E1 tasks.
Models compared: flat_gru_v0
Parameter counts: 3276304
Datasets: data/synthetic/phase1
Train split sizes: 8192
Validation split sizes: 1024
OOD split sizes: 1024
Hardware: cpu
Batch size: 64
Mixed precision: False
Peak VRAM: 0.00 MB
Wall-clock per epoch: 63.04 sec
Seeds: 7
val_exact_accuracy: 0.5371
ood_exact_accuracy: 0.3096
macro_exact_accuracy: 0.5371
steps_to_best_val: 1
run_stability: instability_events=0, train_max_grad_norm=5.3198
Failure analysis: Per-task accuracies are listed below this template block.
Conclusion: Single-run artifact only; compare both models before choosing an E1 bucket.
Kill criteria met?: No
Next action: Use this run alongside the matched baseline comparison.

val_task_accuracies:
{
  "graph_waypoint": 0.44921875,
  "nested_arith": 0.26171875,
  "register_machine": 0.73828125,
  "segment_match": 0.69921875
}

ood_task_accuracies:
{
  "graph_waypoint": 0.05859375,
  "nested_arith": 0.24609375,
  "register_machine": 0.56640625,
  "segment_match": 0.3671875
}

checkpoint_path: reports\experiments\20260318-031139-phase1\best.pt
