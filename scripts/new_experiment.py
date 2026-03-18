from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.run import ExperimentSpec, write_experiment_stub


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a new experiment report stub")
    parser.add_argument("slug")
    parser.add_argument("title")
    parser.add_argument("hypothesis")
    parser.add_argument("cause")
    parser.add_argument(
        "--root",
        default="reports/experiments",
        help="Directory where the experiment report should be created",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report_path = write_experiment_stub(
        root=Path(args.root),
        spec=ExperimentSpec(
            slug=args.slug,
            title=args.title,
            hypothesis=args.hypothesis,
            cause=args.cause,
        ),
    )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
