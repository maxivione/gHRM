Title: Bootstrap repo scaffold smoke test
Hypothesis: The repo scaffold is wired well enough to build both baseline models and keep the experiment layout reproducible.
Cause being tested: Initial repo bootstrap and baseline module wiring
Config diff: TBD
Datasets: TBD
Hardware: RTX 3070 8GB
Peak VRAM: Not measured
Wall-clock: Local smoke run only
Primary metrics: Flat baseline forward pass succeeded; hierarchical baseline forward pass succeeded
Failure analysis: Direct script execution initially failed because `scripts/` was not adding the repo root to `sys.path`
Conclusion: The bootstrap scaffold is usable for the next step, but there is still no real training loop or dataset generator
Next action: Implement phase-0 synthetic dataset generation and the E1 train/eval harness
