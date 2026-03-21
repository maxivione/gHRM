Title: G3 — parameter-matched small transformer baseline on Phase 1
Hypothesis: If attention closes the OOD gap that the flat GRU shows on Phase 1 tasks, then the benchmark is not hierarchy-sensitive — it is just recurrent-model-sensitive.
Cause being tested: Whether self-attention (full context access per layer) resolves the flat GRU's OOD failures, especially on graph_waypoint.
Config diff: new model small_transformer_v0 (d_model=384, nhead=6, 2 encoder layers, dim_feedforward=1024, max_seq_len=512, head_hidden_dim=512, dropout=0.1). Same train config (phase1_local_3070.toml), same eval config, same data splits.
Models compared: flat_gru_v0, small_transformer_v0
Parameter counts: flat_gru_v0=3,276,304; small_transformer_v0=3,214,096 (delta=-1.9%)
Datasets: data/synthetic/phase1 (nested_arith, graph_waypoint, register_machine, segment_match)
Hardware: cuda (RTX 3070)
Batch size: 64
Mixed precision: false
Seeds: 7, 42, 123

## Parameter Counts

| model | params_total | delta vs flat GRU |
| --- | ---: | ---: |
| flat_gru_v0 | 3,276,304 | 0.0% |
| small_transformer_v0 | 3,214,096 | -1.9% |

## Per-Seed Results

| model | seed | best_ep | stop_ep | val_acc | ood_acc | val_gw | ood_gw | val_na | ood_na | val_rm | ood_rm | val_sm | ood_sm | vram_mb | sec/ep |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat_gru_v0 | 7 | 30 | 30 | 0.8184 | 0.4062 | 0.7578 | 0.0586 | 0.7656 | 0.4766 | 0.7500 | 0.5430 | 1.0000 | 0.5469 | 158.7 | 2.33 |
| flat_gru_v0 | 42 | 27 | 30 | 0.8164 | 0.3857 | 0.7539 | 0.0703 | 0.7773 | 0.4609 | 0.7383 | 0.5117 | 0.9961 | 0.5000 | 153.4 | 1.99 |
| flat_gru_v0 | 123 | 25 | 30 | 0.7920 | 0.4199 | 0.7148 | 0.0977 | 0.7891 | 0.4648 | 0.7383 | 0.5312 | 0.9258 | 0.5859 | 155.8 | 2.24 |
| small_transformer_v0 | 7 | 27 | 30 | 0.7754 | 0.3252 | 0.7539 | 0.0781 | 0.6484 | 0.3672 | 0.6992 | 0.5156 | 1.0000 | 0.3398 | 253.9 | 2.37 |
| small_transformer_v0 | 42 | 11 | 16 | 0.7559 | 0.4170 | 0.6992 | 0.0547 | 0.6406 | 0.3711 | 0.6836 | 0.5195 | 1.0000 | 0.7227 | 253.9 | 2.41 |
| small_transformer_v0 | 123 | 17 | 22 | 0.7500 | 0.3008 | 0.6875 | 0.0508 | 0.6406 | 0.3828 | 0.6719 | 0.4883 | 1.0000 | 0.2812 | 253.9 | 2.32 |

## Mean ± Std Across Seeds

| model | val_acc | ood_acc | val_gw | ood_gw | val_na | ood_na | val_rm | ood_rm | val_sm | ood_sm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flat_gru_v0 | 0.8089 ± 0.0147 | 0.4040 ± 0.0172 | 0.7422 ± 0.0238 | 0.0755 ± 0.0200 | 0.7773 ± 0.0117 | 0.4674 ± 0.0081 | 0.7422 ± 0.0068 | 0.5286 ± 0.0158 | 0.9740 ± 0.0418 | 0.5443 ± 0.0430 |
| small_transformer_v0 | 0.7604 ± 0.0133 | 0.3477 ± 0.0613 | 0.7135 ± 0.0354 | 0.0612 ± 0.0148 | 0.6432 ± 0.0045 | 0.3737 ± 0.0081 | 0.6849 ± 0.0137 | 0.5078 ± 0.0170 | 1.0000 ± 0.0000 | 0.4479 ± 0.2397 |

## GPU Stats

| model | peak_vram_mb | sec_per_epoch |
| --- | --- | --- |
| flat_gru_v0 | 156.0 ± 2.6 | 2.19 ± 0.18 |
| small_transformer_v0 | 253.9 ± 0.0 | 2.37 ± 0.04 |

## Stability

All 6 runs completed with instability_events=0. No inf/NaN in gradients or losses.
The transformer early-stopped on 2 of 3 seeds (seed 42 at epoch 16, seed 123 at epoch 22), suggesting it converges faster but to a lower val accuracy — consistent with overfitting or capacity-limited generalization.

## Key Findings

1. **Transformer does NOT close the OOD gap.** The small transformer is strictly worse than the flat GRU on OOD accuracy (0.3477 vs 0.4040 mean), and on val accuracy (0.7604 vs 0.8089).

2. **graph_waypoint OOD remains near-zero for both models.** GRU: 0.0755 ± 0.02, Transformer: 0.0612 ± 0.01. Attention does not help here at all.

3. **Transformer is worse on nested_arith and register_machine** on both val and OOD. These are the sequential/compositional tasks where you might expect attention to help. It doesn't.

4. **segment_match is near-saturated on val for both** (GRU: 0.974, Transformer: 1.000) but diverges wildly on OOD for the transformer (0.4479 ± 0.2397 vs GRU 0.5443 ± 0.0430). The transformer's OOD on segment_match is extremely seed-dependent.

5. **register_machine is the only task where the models are comparable on OOD** (GRU: 0.5286, Transformer: 0.5078). This task may be less hierarchy-sensitive.

## Interpretation

This is **Bucket 2: transformer also fails on OOD**.

The transformer has full attention over the sequence — every token can attend to every other token — and it still cannot generalize OOD on these tasks. The flat GRU, despite only having recurrent context, does slightly *better* on OOD.

This means:
- The benchmark is **genuinely diagnostic**: plain attention at matched scale does not trivially solve it.
- The OOD gap is not just a "recurrent model limitation" — it is a structural generalization failure that attention alone doesn't fix either.
- A new HRM-family idea (hierarchical control, learned decomposition) is **justified** as the next step.

## Per-Task Commentary

- **graph_waypoint**: Both models collapse to near-random on OOD. This is the hardest task and the strongest anchor for HRM-style work.
- **nested_arith**: GRU is clearly better than transformer on both val and OOD. Recurrence may help here for left-to-right evaluation.
- **register_machine**: Close to a wash on OOD. This task's sequential nature may not require hierarchy.
- **segment_match**: Val is saturated for both. OOD variance for transformer is very high. Consider demoting or dropping this task; it may not be discriminative enough.

## Conclusion

Bucket 2 — transformer also fails on OOD. The Phase 1 benchmark is validated as hierarchy-sensitive. Designing a new HRM-family architecture for Phase 1 is justified.

Kill criteria met? No
Next action: Design a new HRM-family architecture targeting Phase 1, with graph_waypoint as the primary anchor task. segment_match should be demoted (kept but not used as a primary discriminator) due to val saturation and high OOD variance.

## Run Artifacts

- flat_gru_v0 seed 7: `reports/experiments/20260318-035337-phase1`
- flat_gru_v0 seed 42: `reports/experiments/20260318-035454-phase1`
- flat_gru_v0 seed 123: `reports/experiments/20260318-035605-phase1`
- small_transformer_v0 seed 7: `reports/experiments/20260318-035724-phase1`
- small_transformer_v0 seed 42: `reports/experiments/20260318-035845-phase1`
- small_transformer_v0 seed 123: `reports/experiments/20260318-035927-phase1`
