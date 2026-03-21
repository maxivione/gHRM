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
Peak VRAM: 163.09 MB
Wall-clock per epoch: 13.26 sec
Seeds: 7
val_exact_accuracy: 0.7578
ood_exact_accuracy: 0.4004
macro_exact_accuracy: 0.7578
steps_to_best_val: 23
run_stability: instability_events=0, train_max_grad_norm=3.0728
Failure analysis: Per-task accuracies are listed below this template block.
Conclusion: Single-run artifact only; compare both models before choosing an E1 bucket.
Kill criteria met?: No
Next action: Use this run alongside the matched baseline comparison.

val_task_accuracies:
{
  "graph_waypoint": 0.66796875,
  "nested_arith": 0.77734375,
  "register_machine": 0.73828125,
  "segment_match": 0.84765625
}

ood_task_accuracies:
{
  "graph_waypoint": 0.046875,
  "nested_arith": 0.484375,
  "register_machine": 0.53515625,
  "segment_match": 0.53515625
}

checkpoint_path: reports\experiments\20260318-042724-phase1\best.pt
