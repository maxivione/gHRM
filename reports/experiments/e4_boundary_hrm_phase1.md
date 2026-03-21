# E4 — Boundary-Detecting HRM vs Flat GRU on Phase 1

Title: E4 — learned-segmentation test: boundary_hrm_v0 vs flat_gru_v0 on Phase 1
Hypothesis: If OOD generalization on compositional tasks requires identifying problem sub-structure, then a model that learns data-dependent segment boundaries (rather than fixed chunking) should generalize better than a flat GRU on OOD instances.
Cause being tested: Whether learned boundary detection + controller GRU over variable-length segments improves OOD accuracy compared to a flat GRU, with graph_waypoint as the anchor task.
Pre-registered interpretation: E4 is a learned-segmentation test, not a general HRM success claim. If graph_waypoint improves but nested_arith does not, report as task-specific.
Config diff: new model boundary_hrm_v0 (same 2-layer 512-hidden fused nn.GRU worker as flat GRU, plus boundary scorer Linear(512→1), segment projection Linear(512→128), controller GRUCell(128→128), gated residual fusion, ponder_coeff=0.01, halt_bias=-1.0). Same train config (phase1_local_3070.toml), same eval config, same data splits.
Models compared: flat_gru_v0, boundary_hrm_v0
Parameter counts: flat_gru_v0=3,276,304; boundary_hrm_v0=3,508,242 (delta=+7.1%)
Datasets: data/synthetic/phase1 (nested_arith, graph_waypoint, register_machine, segment_match)
Hardware: cuda (RTX 3070)
Batch size: 64
Mixed precision: false
Seeds: 7, 42, 123

## Parameter Counts

| model | params_total | delta vs flat GRU |
| --- | ---: | ---: |
| flat_gru_v0 | 3,276,304 | 0.0% |
| boundary_hrm_v0 | 3,508,242 | +7.1% |

## Per-Seed Results

| model | seed | best_ep | stop_ep | val_acc | ood_acc | val_gw | ood_gw | val_na | ood_na | val_rm | ood_rm | val_sm | ood_sm | vram_mb | sec/ep |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat_gru_v0 | 7 | 30 | 30 | 0.8184 | 0.4062 | 0.7578 | 0.0586 | 0.7656 | 0.4766 | 0.7500 | 0.5430 | 1.0000 | 0.5469 | 158.7 | 2.33 |
| flat_gru_v0 | 42 | 27 | 30 | 0.8164 | 0.3857 | 0.7539 | 0.0703 | 0.7773 | 0.4609 | 0.7383 | 0.5117 | 0.9961 | 0.5000 | 153.4 | 1.99 |
| flat_gru_v0 | 123 | 25 | 30 | 0.7920 | 0.4199 | 0.7148 | 0.0977 | 0.7891 | 0.4648 | 0.7383 | 0.5312 | 0.9258 | 0.5859 | 155.8 | 2.24 |
| boundary_hrm_v0 | 7 | 23 | 28 | 0.7578 | 0.4004 | 0.6680 | 0.0469 | 0.7773 | 0.4844 | 0.7383 | 0.5352 | 0.8477 | 0.5352 | 163.1 | 13.26 |
| boundary_hrm_v0 | 42 | 30 | 30 | 0.7969 | 0.4141 | 0.7148 | 0.1211 | 0.7734 | 0.4844 | 0.7461 | 0.5273 | 0.9531 | 0.5234 | 161.9 | 8.02 |
| boundary_hrm_v0 | 123 | 18 | 23 | 0.7539 | 0.4043 | 0.6914 | 0.0547 | 0.7891 | 0.4688 | 0.7266 | 0.5859 | 0.8086 | 0.5078 | 161.6 | 10.07 |

## Mean ± Std Across Seeds

| model | val_acc | ood_acc | val_gw | ood_gw | val_na | ood_na | val_rm | ood_rm | val_sm | ood_sm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flat_gru_v0 | 0.8089 ± 0.0120 | 0.4040 ± 0.0140 | 0.7422 ± 0.0194 | 0.0755 ± 0.0164 | 0.7773 ± 0.0096 | 0.4674 ± 0.0066 | 0.7422 ± 0.0055 | 0.5286 ± 0.0129 | 0.9740 ± 0.0341 | 0.5443 ± 0.0351 |
| boundary_hrm_v0 | 0.7695 ± 0.0194 | 0.4062 ± 0.0057 | 0.6914 ± 0.0191 | 0.0742 ± 0.0333 | 0.7799 ± 0.0066 | 0.4792 ± 0.0074 | 0.7370 ± 0.0080 | 0.5495 ± 0.0260 | 0.8698 ± 0.0610 | 0.5221 ± 0.0112 |

## GPU Stats

| model | peak_vram_mb | sec_per_epoch |
| --- | --- | --- |
| flat_gru_v0 | 155.96 ± 2.15 | 2.19 ± 0.14 |
| boundary_hrm_v0 | 162.21 ± 0.63 | 10.45 ± 2.16 |

## Pre-Registered Pass/Fail Gate

| Criterion | Threshold | Observed | Result |
| --- | --- | --- | --- |
| 1. bhrm OOD > flat OOD | > 0.4040 | 0.4062 | **PASS** (marginal, +0.0022) |
| 2. bhrm gw OOD > flat gw OOD | > 0.0755 | 0.0742 | **FAIL** (-0.0013) |
| 3. bhrm OOD std ≤ 0.026 | ≤ 0.026 | 0.0057 | **PASS** |
| 4. bhrm sec/ep ≤ 6.6 | ≤ 6.6 | 10.45 | **FAIL** |

**Overall verdict: FAIL (2 of 4 criteria not met)**

## Kill Criteria Check

| Kill condition | Triggered? |
| --- | --- |
| gw OOD ≤ flat on all 3 seeds | No (seed 42 hit 0.121 > 0.0755) |
| val < 0.70 | No (0.7695) |
| inf/NaN on ≥2 seeds | No (0 instability events) |
| wall-clock > 11.0 | No (10.45 mean, seed 7 hit 13.26) |
| halt_prob_std < 0.01 all seeds by epoch 5 | Borderline (final halt_std: 0.0015, 0.0013, 0.0127) |

No hard kill triggered, but the result is a clear fail on the pass gate.

## Boundary Telemetry (at best epoch)

| seed | halt_prob_mean | halt_prob_std | segments_per_seq | seg_std | gate_mean |
| --- | --- | --- | --- | --- | --- |
| 7 | 0.000835 | 0.001525 | 0.80 | 0.41 | 0.2048 |
| 42 | 0.000728 | 0.001288 | 0.75 | 0.44 | 0.1003 |
| 123 | 0.003427 | 0.012728 | 0.88 | 0.33 | 0.4450 |

**The boundary scorer effectively collapsed.** halt_prob_mean < 0.004 on all seeds means boundary fire probability is near zero at every timestep. segments_per_seq < 1.0 means most sequences emit zero segments (only the trailing-remainder fallback contributes). The model learned to suppress boundary detection entirely and rely on the flat worker's h_T, exactly as predicted in the pre-registered failure mode.

## Interpretation

### What happened

The boundary scorer's halt probability converged to near-zero across all timesteps. This means:
1. The ponder cost regularizer (which penalises total halt probability) pushed halt_probs down
2. There was insufficient gradient signal from the downstream classification loss to push halt_probs up at structurally meaningful positions
3. The model found it easier to let the flat GRU's h_T do all the work and keep the gate small (gate_mean 0.10-0.45)

The result is that boundary_hrm_v0 effectively degenerated into a flat GRU with ~232K unused controller parameters. The overall OOD "improvement" of +0.0022 is noise, not signal.

### Why graph_waypoint did not improve

The boundary scorer needs to learn "fire at WAYPOINT_SEP tokens" from the classification loss alone. At halt_prob_mean < 0.001, no boundaries are firing, so the controller GRU processes zero segments, and the gate contributes nothing to the final representation. The model never had a chance to test whether segmentation helps because segmentation never happened.

### Why wall-clock failed

Despite using the same fused nn.GRU for the worker (phase 1), the per-timestep boundary accumulation loop in phase 2 adds ~8 sec/epoch overhead. The boundary scoring itself is cheap (one batched linear), but the sequential accumulator loop with conditional controller firing and tensor indexing is expensive in Python. At 10.45 sec/epoch, this is 4.8× the flat GRU — above the 3× pass criterion.

### Nested_arith check

nested_arith OOD: boundary_hrm 0.4792 vs flat GRU 0.4674 (+0.0118). This is a tiny improvement, well within noise given the boundary scorer wasn't actually segmenting. No evidence of hierarchy detection.

## Conclusion

**FAIL.** The learned-segmentation hypothesis is not supported at this scale. The boundary scorer collapsed to near-zero halt probabilities, the controller was effectively unused, and the model degenerated into a slower flat GRU. The marginal OOD improvement (+0.0022) is not statistically meaningful.

The pre-registered failure mode (boundary scorer collapse) is the primary cause. The ponder cost regularizer and weak gradient signal from the classification loss combined to suppress all boundary detection.

Kill criteria met? Not technically (no hard kill triggered), but the pass gate failed on 2 of 4 criteria, and the boundary telemetry confirms the mechanism is not working.

## Next action

Archive E4 as "mechanism not activated — boundary scorer collapsed." The learned-segmentation idea is not falsified in principle (the mechanism never actually fired), but the current training signal is insufficient to learn boundary placement at this scale. Possible causes:
1. ponder_coeff too high (suppressed all halt probability)
2. halt_bias too negative (initial halt probs too low to ever fire)
3. no direct supervision signal for boundary placement

However, per project rules (AGENTS.md): do not start E5 or propose tweaks without an explicit request. The current evidence does not support continuing the HRM line.

## Run Artifacts

- boundary_hrm_v0 seed 7: `reports/experiments/20260318-042724-phase1`
- boundary_hrm_v0 seed 42: `reports/experiments/20260318-043457-phase1`
- boundary_hrm_v0 seed 123: `reports/experiments/20260318-044229-phase1`
- flat_gru_v0 seed 7: `reports/experiments/20260318-035337-phase1`
- flat_gru_v0 seed 42: `reports/experiments/20260318-035454-phase1`
- flat_gru_v0 seed 123: `reports/experiments/20260318-035605-phase1`
