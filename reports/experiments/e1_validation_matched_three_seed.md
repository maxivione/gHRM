Title: E1 fair rerun: parameter-matched flat vs hierarchical GRU on the retained phase-0 tasks
Hypothesis: If the hierarchical GRU carries real architectural signal at matched scale, it should beat the flat GRU baseline on held-out accuracy across multiple seeds.
Cause being tested: Architecture effect after removing the original E1 procedural confounds: parameter-count mismatch, 8-epoch truncation, single-seed evidence, and the degenerate string_rewrite task.
Config diff: hierarchical_gru_v0 dims increased to embedding_dim=384, worker_hidden_dim=480, planner_hidden_dim=320, fusion_dim=512; train epochs increased to 30 with early_stopping_patience=5; seeds expanded to 7, 42, 123; string_rewrite_final_token removed from the active E1 suite; AMP disabled for the final controlled matrix after the matched-dim AMP attempt produced an inf grad norm on hierarchical_gru_v0 seed 7.
Datasets: data/synthetic/e1_phase0 across maze_path_exists, sudoku_cell_fill, graph_shortest_path_len
Hardware: cuda
Peak VRAM: flat_gru_v0=156.84 +/- 1.32 MB; hierarchical_gru_v0=148.53 +/- 0.00 MB
Wall-clock: flat_gru_v0=7.68 +/- 1.00 sec/run; hierarchical_gru_v0=17.01 +/- 6.10 sec/run
Primary metrics: flat_gru_v0 val_avg=0.8750 +/- 0.0052 and ood_avg=0.5694 +/- 0.0797; hierarchical_gru_v0 val_avg=0.8351 +/- 0.0552 and ood_avg=0.5330 +/- 0.0443
Failure analysis: The matched rerun removes the main procedural confounds, and the hierarchical GRU still does not outperform the flat GRU. The gap is most visible on graph_shortest_path_len, where the hierarchical model is both worse on the mean and much higher variance across seeds. The first matched-dim AMP attempt also exposed a stability problem on hierarchical_gru_v0 seed 7, so the final comparison had to move both models to full precision to finish the required three-seed matrix fairly.
Conclusion: B) architectural no-signal. Under a parameter-matched, three-seed rerun on the retained E1 tasks, the hierarchical GRU is slower, less AMP-stable, and worse than the flat GRU on both validation and OOD mean accuracy.
Next action: Do not start E2. Archive E1 as architectural no-signal at matched scale; only revisit this area with a narrowly scoped hierarchical optimization or stability ablation if a rerun is explicitly requested later.

## Final Parameter Counts

| model | params_total | delta vs flat |
| --- | ---: | ---: |
| flat_gru_v0 | 3,276,304 | 0.00% |
| hierarchical_gru_v0 | 3,209,232 | -2.05% |

## Seed Runs

| model | seed | best_epoch | stop_epoch | val_graph | val_maze | val_sudoku | val_avg | ood_graph | ood_maze | ood_sudoku | ood_avg | peak_vram_mb | wall_clock_sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat_gru_v0 | 7 | 23 | 28 | 0.8438 | 0.7969 | 1.0000 | 0.8802 | 0.3906 | 0.6094 | 0.5625 | 0.5208 | 155.50 | 8.70 |
| flat_gru_v0 | 42 | 25 | 30 | 0.8125 | 0.8125 | 1.0000 | 0.8750 | 0.4531 | 0.5781 | 0.9531 | 0.6615 | 158.14 | 7.64 |
| flat_gru_v0 | 123 | 21 | 26 | 0.7656 | 0.8438 | 1.0000 | 0.8698 | 0.4375 | 0.6562 | 0.4844 | 0.5260 | 156.87 | 6.69 |
| hierarchical_gru_v0 | 7 | 22 | 27 | 0.7969 | 0.8594 | 1.0000 | 0.8854 | 0.2812 | 0.6250 | 0.6875 | 0.5312 | 148.53 | 20.85 |
| hierarchical_gru_v0 | 42 | 7 | 12 | 0.5156 | 0.8125 | 1.0000 | 0.7760 | 0.3281 | 0.5938 | 0.5469 | 0.4896 | 148.53 | 9.97 |
| hierarchical_gru_v0 | 123 | 20 | 25 | 0.7188 | 0.8125 | 1.0000 | 0.8438 | 0.5156 | 0.6094 | 0.6094 | 0.5781 | 148.53 | 20.20 |

## Mean +/- Std Across Seeds

| model | val_graph | val_maze | val_sudoku | val_avg | ood_graph | ood_maze | ood_sudoku | ood_avg | peak_vram_mb | wall_clock_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| flat_gru_v0 | 0.8073 +/- 0.0393 | 0.8177 +/- 0.0239 | 1.0000 +/- 0.0000 | 0.8750 +/- 0.0052 | 0.4271 +/- 0.0325 | 0.6146 +/- 0.0393 | 0.6667 +/- 0.2511 | 0.5694 +/- 0.0797 | 156.84 +/- 1.32 | 7.68 +/- 1.00 |
| hierarchical_gru_v0 | 0.6771 +/- 0.1452 | 0.8281 +/- 0.0271 | 1.0000 +/- 0.0000 | 0.8351 +/- 0.0552 | 0.3750 +/- 0.1240 | 0.6094 +/- 0.0156 | 0.6146 +/- 0.0705 | 0.5330 +/- 0.0443 | 148.53 +/- 0.00 | 17.01 +/- 6.10 |

## Instability Notes

- The first matched-dimension rerun was attempted with AMP still enabled. `hierarchical_gru_v0` hit `inf` grad norm on seed 7 before the first epoch completed, so those AMP runs were excluded from the final metrics.
- The final controlled matrix disabled AMP for both models and completed all six runs with `instability_events=0`.
- Early stopping triggered on 5 of the 6 final runs. Only `flat_gru_v0` seed 42 reached the 30-epoch cap.

## Final Run Artifacts

- flat_gru_v0 seed 7: `reports/experiments/20260318-012643-e1`
- flat_gru_v0 seed 42: `reports/experiments/20260318-012658-e1`
- flat_gru_v0 seed 123: `reports/experiments/20260318-012711-e1`
- hierarchical_gru_v0 seed 7: `reports/experiments/20260318-012609-e1`
- hierarchical_gru_v0 seed 42: `reports/experiments/20260318-012723-e1`
- hierarchical_gru_v0 seed 123: `reports/experiments/20260318-012739-e1`
