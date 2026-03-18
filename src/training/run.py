from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class ExperimentSpec:
    slug: str
    title: str
    hypothesis: str
    cause: str
    config_diff: str = "TBD"
    datasets: str = "TBD"
    hardware: str = "RTX 3070 8GB"


def write_experiment_stub(root: Path, spec: ExperimentSpec) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{timestamp}-{spec.slug}"
    run_dir.mkdir(parents=True, exist_ok=False)

    report_path = run_dir / "report.md"
    report_path.write_text(
        "\n".join(
            [
                f"Title: {spec.title}",
                f"Hypothesis: {spec.hypothesis}",
                f"Cause being tested: {spec.cause}",
                f"Config diff: {spec.config_diff}",
                f"Datasets: {spec.datasets}",
                f"Hardware: {spec.hardware}",
                "Peak VRAM:",
                "Wall-clock:",
                "Primary metrics:",
                "Failure analysis:",
                "Conclusion:",
                "Next action:",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path
