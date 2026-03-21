from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tasks.registry import PHASE1_TASKS


def main() -> int:
    manifest_path = Path("data/manifests/phase1.toml")
    with manifest_path.open("rb") as handle:
        manifest = tomllib.load(handle)

    output_root = Path(manifest["settings"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    for task in PHASE1_TASKS:
        task_config = manifest[task.name]
        task_root = output_root / task.name
        task_root.mkdir(parents=True, exist_ok=True)

        for split_name, ood in (("train", False), ("val", False), ("ood", True)):
            split_count = task_config[f"{split_name}_size"]
            split_seed = task_config["seed"] + {"train": 0, "val": 1, "ood": 2}[split_name]
            rows = task.generator(count=split_count, seed=split_seed, ood=ood)
            with (task_root / f"{split_name}.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
