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
Peak VRAM: 161.65 MB
Wall-clock per epoch: 10.07 sec
Seeds: 123
val_exact_accuracy: 0.7539
ood_exact_accuracy: 0.4043
macro_exact_accuracy: 0.7539
steps_to_best_val: 18
run_stability: instability_events=0, train_max_grad_norm=3.3015
Failure analysis: Per-task accuracies are listed below this template block.
Conclusion: Single-run artifact only; compare both models before choosing an E1 bucket.
Kill criteria met?: No
Next action: Use this run alongside the matched baseline comparison.

val_task_accuracies:
{
  "graph_waypoint": 0.69140625,
  "nested_arith": 0.7890625,
  "register_machine": 0.7265625,
  "segment_match": 0.80859375
}

ood_task_accuracies:
{
  "graph_waypoint": 0.0546875,
  "nested_arith": 0.46875,
  "register_machine": 0.5859375,
  "segment_match": 0.5078125
}

checkpoint_path: reports\experiments\20260318-044229-phase1\best.pt
