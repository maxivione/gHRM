from __future__ import annotations

from collections import defaultdict

import torch


def exact_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = logits.argmax(dim=-1)
    return (predictions == targets).float().mean()


def collect_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"exact_accuracy": 0.0, "macro_exact_accuracy": 0.0}

    total_correct = sum(int(row["predicted"] == row["target"]) for row in rows)
    per_task: dict[str, list[int]] = defaultdict(list)

    for row in rows:
        per_task[row["task_name"]].append(int(row["predicted"] == row["target"]))

    task_accuracies = {
        task_name: sum(outcomes) / len(outcomes) for task_name, outcomes in per_task.items()
    }
    metrics = {
        "exact_accuracy": total_correct / len(rows),
        "macro_exact_accuracy": sum(task_accuracies.values()) / len(task_accuracies),
    }
    for task_name, task_accuracy in task_accuracies.items():
        metrics[f"{task_name}_exact_accuracy"] = task_accuracy
    return metrics
