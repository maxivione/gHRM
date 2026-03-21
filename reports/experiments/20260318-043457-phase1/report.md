Title: boundary_hrm_v0 single-run report
Hypothesis: Shared E1 scaffold can complete the configured phase-0 run without procedural failures.
Cause being tested: Whether this model produces stable optimization signal on the current E1 tasks.
Models compared: boundary_hrm_v0
Parameter counts: 3508242
Datasets: data/synthetic/phase1
Train split sizes: 8192
Validation split sizes: 1024
OOD split sizes: 1024
Hardware: cuda
Batch size: 64
Mixed precision: False
Peak VRAM: 161.88 MB
Wall-clock per epoch: 8.02 sec
Seeds: 42
val_exact_accuracy: 0.7969
ood_exact_accuracy: 0.4141
macro_exact_accuracy: 0.7969
steps_to_best_val: 30
run_stability: instability_events=0, train_max_grad_norm=3.8276
Failure analysis: Per-task accuracies are listed below this template block.
Conclusion: Single-run artifact only; compare both models before choosing an E1 bucket.
Kill criteria met?: No
Next action: Use this run alongside the matched baseline comparison.

val_task_accuracies:
{
  "graph_waypoint": 0.71484375,
  "nested_arith": 0.7734375,
  "register_machine": 0.74609375,
  "segment_match": 0.953125
}

ood_task_accuracies:
{
  "graph_waypoint": 0.12109375,
  "nested_arith": 0.484375,
  "register_machine": 0.52734375,
  "segment_match": 0.5234375
}

checkpoint_path: reports\experiments\20260318-043457-phase1\best.pt
