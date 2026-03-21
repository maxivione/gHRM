"""G3: run the small transformer baseline against flat GRU on Phase 1.

Runs both models × 3 seeds (7, 42, 123) on GPU, same tasks/splits/stopping.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train_e1 import run_training
from src.tasks.registry import PHASE1_TASKS

SEEDS = [7, 42, 123]
MODELS = [
    "configs/model/flat_gru_v0.toml",
    "configs/model/small_transformer_v0.toml",
]
TRAIN_CONFIG = "configs/train/phase1_local_3070.toml"
EVAL_CONFIG = "configs/eval/e1_phase0.toml"


def main() -> int:
    results: list[dict] = []
    for model_config in MODELS:
        for seed in SEEDS:
            model_name = Path(model_config).stem
            print(f"\n{'='*60}")
            print(f"  G3: {model_name}  seed={seed}")
            print(f"{'='*60}")
            run_dir = run_training(
                model_config_path=model_config,
                train_config_path=TRAIN_CONFIG,
                eval_config_path=EVAL_CONFIG,
                device_override="cuda",
                seed_override=seed,
                tasks=PHASE1_TASKS,
            )
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            summary["model_name"] = model_name
            results.append(summary)
            print(f"  -> val_acc={summary['val_exact_accuracy']:.4f}  ood_acc={summary['ood_exact_accuracy']:.4f}")
            print(f"  -> run_dir={run_dir}")

    # Print comparison table
    print(f"\n\n{'='*80}")
    print("G3 COMPARISON: flat_gru_v0 vs small_transformer_v0 on Phase 1")
    print(f"{'='*80}")
    header = f"{'model':<28} {'seed':>4} {'params':>10} {'val_acc':>8} {'ood_acc':>8} {'val_gw':>7} {'ood_gw':>7} {'val_na':>7} {'ood_na':>7} {'val_rm':>7} {'ood_rm':>7} {'val_sm':>7} {'ood_sm':>7} {'vram_mb':>8} {'sec/ep':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model_name']:<28} {r['seed']:>4} {r['params_total']:>10,} "
            f"{r['val_exact_accuracy']:>8.4f} {r['ood_exact_accuracy']:>8.4f} "
            f"{r.get('val_graph_waypoint_exact_accuracy', 0):>7.4f} {r.get('ood_graph_waypoint_exact_accuracy', 0):>7.4f} "
            f"{r.get('val_nested_arith_exact_accuracy', 0):>7.4f} {r.get('ood_nested_arith_exact_accuracy', 0):>7.4f} "
            f"{r.get('val_register_machine_exact_accuracy', 0):>7.4f} {r.get('ood_register_machine_exact_accuracy', 0):>7.4f} "
            f"{r.get('val_segment_match_exact_accuracy', 0):>7.4f} {r.get('ood_segment_match_exact_accuracy', 0):>7.4f} "
            f"{r['peak_vram_mb']:>8.1f} {r['wall_clock_sec_per_epoch']:>7.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
