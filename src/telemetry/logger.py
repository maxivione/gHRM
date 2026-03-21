from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class JsonlLogger:
    log_path: Path

    def log(self, event: str, **fields: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **fields,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def build_run_dir(root: Path, run_name: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    for suffix in range(100):
        candidate_name = f"{timestamp}-{run_name}" if suffix == 0 else f"{timestamp}-{run_name}-{suffix}"
        run_dir = root / candidate_name
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
    raise RuntimeError(f"Could not allocate a unique run directory under {root}")


def read_peak_vram_mb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
