from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.tasks.registry import PHASE0_TASKS


REQUIRED_PATHS = [
    Path("configs/model"),
    Path("configs/train"),
    Path("configs/eval"),
    Path("data/manifests"),
    Path("datasets/grids"),
    Path("datasets/symbolic"),
    Path("datasets/text"),
    Path("datasets/code"),
    Path("src/models"),
    Path("src/training"),
    Path("src/eval"),
    Path("src/tasks"),
    Path("src/telemetry"),
    Path("reports/experiments"),
]


def check_layout() -> int:
    missing = [str(path) for path in REQUIRED_PATHS if not path.exists()]
    if missing:
        print("Missing required paths:")
        for path in missing:
            print(f"  - {path}")
        return 1

    print("Repo layout looks complete.")
    return 0


def print_phase0() -> int:
    for task in PHASE0_TASKS:
        print(f"{task.name}: {task.domain} | {task.description}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="gHRM-Lite repo utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check-layout", help="Verify the expected repo scaffold exists")
    subparsers.add_parser("print-phase0", help="Print phase-0 tasks from the registry")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "check-layout":
        return check_layout()
    if args.command == "print-phase0":
        return print_phase0()

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
