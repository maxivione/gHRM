from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.boundary_hrm import BoundaryHRMConfig, BoundaryHRM
from src.models.flat_gru import FlatGRUConfig, FlatGRUBaseline
from src.models.gated_sidecar_gru import GatedSidecarGRUConfig, GatedSidecarGRU
from src.models.hierarchical_gru import HierarchicalGRUConfig, HierarchicalGRUBaseline
from src.models.lqb_gru import LQBConfig, LQBModel
from src.models.small_transformer import SmallTransformerConfig, SmallTransformerBaseline
from src.tasks.registry import GLOBAL_NUM_CLASSES, GLOBAL_VOCAB_SIZE, E1_TASKS, PHASE1_TASKS
from src.telemetry.logger import JsonlLogger, build_run_dir
from src.training.loop import run_eval_epoch, run_train_epoch
from src.utils.seed import seed_everything


@dataclass(frozen=True)
class TrainE1Config:
    seed: int
    device: str
    epochs: int
    early_stopping_patience: int | None
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    amp: bool
    data_root: str
    report_root: str
    run_name: str


class JsonlTaskDataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def _load_toml(path: str) -> dict:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _build_train_config(
    path: str,
    epochs_override: int | None,
    device_override: str | None,
    seed_override: int | None,
) -> TrainE1Config:
    config_data = _load_toml(path)
    return TrainE1Config(
        seed=seed_override if seed_override is not None else config_data["seed"],
        device=device_override or config_data["device"],
        epochs=epochs_override or config_data["epochs"],
        early_stopping_patience=config_data.get("early_stopping_patience"),
        batch_size=config_data["batch_size"],
        learning_rate=config_data["learning_rate"],
        weight_decay=config_data["weight_decay"],
        grad_clip_norm=config_data["grad_clip_norm"],
        amp=config_data["amp"],
        data_root=config_data["data_root"],
        report_root=config_data["report_root"],
        run_name=config_data["run_name"],
    )


def _build_model(model_config_path: str):
    config_data = _load_toml(model_config_path)
    model_name = config_data["name"]
    if model_name == "flat_gru_v0":
        return FlatGRUBaseline(
            FlatGRUConfig(
                vocab_size=config_data["vocab_size"],
                embedding_dim=config_data["embedding_dim"],
                hidden_dim=config_data["hidden_dim"],
                num_layers=config_data["num_layers"],
                dropout=config_data["dropout"],
                num_classes=config_data["num_classes"],
            )
        )
    if model_name == "hierarchical_gru_v0":
        return HierarchicalGRUBaseline(
            HierarchicalGRUConfig(
                vocab_size=config_data["vocab_size"],
                embedding_dim=config_data["embedding_dim"],
                worker_hidden_dim=config_data["worker_hidden_dim"],
                planner_hidden_dim=config_data["planner_hidden_dim"],
                fusion_dim=config_data["fusion_dim"],
                planner_interval=config_data["planner_interval"],
                dropout=config_data["dropout"],
                num_classes=config_data["num_classes"],
            )
        )
    if model_name == "gated_sidecar_gru_v0":
        return GatedSidecarGRU(
            GatedSidecarGRUConfig(
                vocab_size=config_data["vocab_size"],
                embedding_dim=config_data["embedding_dim"],
                hidden_dim=config_data["hidden_dim"],
                num_layers=config_data["num_layers"],
                ctrl_dim=config_data["ctrl_dim"],
                ctrl_layers=config_data["ctrl_layers"],
                chunk_size=config_data["chunk_size"],
                gate_init_bias=config_data["gate_init_bias"],
                dropout=config_data["dropout"],
                num_classes=config_data["num_classes"],
                freeze_gate=config_data.get("freeze_gate", False),
                freeze_controller=config_data.get("freeze_controller", False),
            )
        )
    if model_name == "lqb_gru_v0":
        return LQBModel(
            LQBConfig(
                vocab_size=config_data["vocab_size"],
                embedding_dim=config_data["embedding_dim"],
                hidden_dim=config_data["hidden_dim"],
                num_layers=config_data["num_layers"],
                d_attn=config_data["d_attn"],
                num_queries=config_data["num_queries"],
                gate_init_bias=config_data["gate_init_bias"],
                dropout=config_data["dropout"],
                num_classes=config_data["num_classes"],
                freeze_queries=config_data.get("freeze_queries", False),
            )
        )
    if model_name == "small_transformer_v0":
        return SmallTransformerBaseline(
            SmallTransformerConfig(
                vocab_size=config_data["vocab_size"],
                d_model=config_data["d_model"],
                nhead=config_data["nhead"],
                num_layers=config_data["num_layers"],
                dim_feedforward=config_data["dim_feedforward"],
                max_seq_len=config_data["max_seq_len"],
                dropout=config_data["dropout"],
                head_hidden_dim=config_data["head_hidden_dim"],
                num_classes=config_data["num_classes"],
            )
        )
    if model_name == "boundary_hrm_v0":
        return BoundaryHRM(
            BoundaryHRMConfig(
                vocab_size=config_data["vocab_size"],
                embedding_dim=config_data["embedding_dim"],
                hidden_dim=config_data["hidden_dim"],
                num_layers=config_data["num_layers"],
                ctrl_dim=config_data["ctrl_dim"],
                halt_bias=config_data["halt_bias"],
                ponder_coeff=config_data["ponder_coeff"],
                dropout=config_data["dropout"],
                num_classes=config_data["num_classes"],
            )
        )
    raise ValueError(f"Unknown E1 model config name {model_name!r} in {model_config_path}")


def _count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _read_rows(data_root: Path, split: str, tasks: list | None = None) -> list[dict]:
    if tasks is None:
        tasks = E1_TASKS
    rows = []
    for task in tasks:
        split_path = data_root / task.name / f"{split}.jsonl"
        with split_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                rows.append(json.loads(line))
    return rows


def _collate_rows(rows: list[dict]) -> dict:
    max_length = max(len(row["input_ids"]) for row in rows)
    input_ids = []
    lengths = []
    targets = []
    task_names = []

    for row in rows:
        lengths.append(len(row["input_ids"]))
        input_ids.append(row["input_ids"] + [0] * (max_length - len(row["input_ids"])))
        targets.append(row["target"])
        task_names.append(row["task_name"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "task_names": task_names,
    }


def run_training(
    model_config_path: str,
    train_config_path: str,
    eval_config_path: str,
    device_override: str | None = None,
    epochs_override: int | None = None,
    seed_override: int | None = None,
    tasks: list | None = None,
) -> Path:
    eval_config = _load_toml(eval_config_path)
    train_config = _build_train_config(train_config_path, epochs_override, device_override, seed_override)
    seed_everything(train_config.seed)

    resolved_device = train_config.device if train_config.device == "cpu" or torch.cuda.is_available() else "cpu"
    device = torch.device(resolved_device)

    model = _build_model(model_config_path).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    ponder_coeff = _load_toml(model_config_path).get("ponder_coeff", 0.0)

    data_root = Path(train_config.data_root)
    train_loader = DataLoader(
        JsonlTaskDataset(_read_rows(data_root, "train", tasks)),
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=_collate_rows,
    )
    val_loader = DataLoader(
        JsonlTaskDataset(_read_rows(data_root, "val", tasks)),
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=_collate_rows,
    )
    ood_loader = DataLoader(
        JsonlTaskDataset(_read_rows(data_root, "ood", tasks)),
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=_collate_rows,
    )

    run_dir = build_run_dir(Path(train_config.report_root), train_config.run_name)
    logger = JsonlLogger(run_dir / "events.jsonl")
    params_total = _count_parameters(model)

    logger.log(
        "run_started",
        model_config=model_config_path,
        train_config=train_config_path,
        eval_config=eval_config_path,
        seed=train_config.seed,
        device=device.type,
        params_total=params_total,
        early_stopping_patience=train_config.early_stopping_patience,
        vocab_size=GLOBAL_VOCAB_SIZE,
        num_classes=GLOBAL_NUM_CLASSES,
        metrics=eval_config["metrics"],
    )

    best_val_accuracy = -1.0
    best_epoch_summary: dict[str, float] = {}
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = run_dir / "best.pt"

    for epoch in range(1, train_config.epochs + 1):
        train_epoch = run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip_norm=train_config.grad_clip_norm,
            amp_enabled=train_config.amp,
            ponder_coeff=ponder_coeff,
        )
        val_epoch = run_eval_epoch(model=model, loader=val_loader, loss_fn=loss_fn, device=device)
        ood_epoch = run_eval_epoch(model=model, loader=ood_loader, loss_fn=loss_fn, device=device)

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_epoch.loss,
            "val_loss": val_epoch.loss,
            "ood_loss": ood_epoch.loss,
            "train_max_grad_norm": train_epoch.max_grad_norm,
            "instability_events": train_epoch.instability_events,
            "val_exact_accuracy": val_epoch.metrics["exact_accuracy"],
            "val_macro_exact_accuracy": val_epoch.metrics["macro_exact_accuracy"],
            "ood_exact_accuracy": ood_epoch.metrics["exact_accuracy"],
            "ood_macro_exact_accuracy": ood_epoch.metrics["macro_exact_accuracy"],
            "peak_vram_mb": max(
                train_epoch.peak_vram_mb,
                val_epoch.peak_vram_mb,
                ood_epoch.peak_vram_mb,
            ),
            "wall_clock_sec_per_epoch": train_epoch.wall_clock_sec,
        }
        # Boundary telemetry (only present for BoundaryHRM)
        boundary_telemetry = getattr(model, "_last_boundary_telemetry", {})
        if boundary_telemetry:
            epoch_summary["boundary_telemetry"] = boundary_telemetry
        for metric_name, metric_value in val_epoch.metrics.items():
            if metric_name not in {"exact_accuracy", "macro_exact_accuracy", "loss"}:
                epoch_summary[f"val_{metric_name}"] = metric_value
        for metric_name, metric_value in ood_epoch.metrics.items():
            if metric_name not in {"exact_accuracy", "macro_exact_accuracy", "loss"}:
                epoch_summary[f"ood_{metric_name}"] = metric_value
        logger.log("epoch_completed", **epoch_summary)

        if epoch_summary["val_exact_accuracy"] > best_val_accuracy:
            best_val_accuracy = epoch_summary["val_exact_accuracy"]
            best_epoch_summary = epoch_summary
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": model_config_path,
                    "train_config": train_config_path,
                    "eval_config": eval_config_path,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if (
            train_config.early_stopping_patience is not None
            and train_config.early_stopping_patience > 0
            and epochs_without_improvement >= train_config.early_stopping_patience
        ):
            logger.log(
                "early_stopped",
                stop_epoch=epoch,
                best_epoch=best_epoch,
                patience=train_config.early_stopping_patience,
                best_val_accuracy=best_val_accuracy,
            )
            break

    final_summary = {
        "model_config": model_config_path,
        "train_config": train_config_path,
        "seed": train_config.seed,
        "device": device.type,
        "params_total": params_total,
        "best_epoch": best_epoch,
        "steps_to_best_val": best_epoch,
        "stop_epoch": epoch,
        "stopped_early": epoch < train_config.epochs,
        "early_stopping_patience": train_config.early_stopping_patience,
        "checkpoint_path": str(checkpoint_path),
        **best_epoch_summary,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(final_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    template_path = Path("reports/experiments/e1_report_template.md")
    field_values = {
        "Title": f"{Path(model_config_path).stem} single-run report",
        "Hypothesis": "Shared E1 scaffold can complete the configured phase-0 run without procedural failures.",
        "Cause being tested": "Whether this model produces stable optimization signal on the current E1 tasks.",
        "Models compared": Path(model_config_path).stem,
        "Parameter counts": str(params_total),
        "Datasets": train_config.data_root,
        "Train split sizes": str(len(train_loader.dataset)),
        "Validation split sizes": str(len(val_loader.dataset)),
        "OOD split sizes": str(len(ood_loader.dataset)),
        "Hardware": device.type,
        "Batch size": str(train_config.batch_size),
        "Mixed precision": str(train_config.amp),
        "Peak VRAM": f"{final_summary['peak_vram_mb']:.2f} MB",
        "Wall-clock per epoch": f"{final_summary['wall_clock_sec_per_epoch']:.2f} sec",
        "Seeds": str(train_config.seed),
        "val_exact_accuracy": f"{final_summary['val_exact_accuracy']:.4f}",
        "ood_exact_accuracy": f"{final_summary['ood_exact_accuracy']:.4f}",
        "macro_exact_accuracy": f"{final_summary['val_macro_exact_accuracy']:.4f}",
        "steps_to_best_val": str(final_summary["steps_to_best_val"]),
        "run_stability": f"instability_events={final_summary['instability_events']}, train_max_grad_norm={final_summary['train_max_grad_norm']:.4f}",
        "Failure analysis": "Per-task accuracies are listed below this template block.",
        "Conclusion": "Single-run artifact only; compare both models before choosing an E1 bucket.",
        "Kill criteria met?": "No",
        "Next action": "Use this run alongside the matched baseline comparison.",
    }
    report_lines = []
    for line in template_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            report_lines.append(line)
            continue
        field_name, _ = line.split(":", 1)
        report_lines.append(f"{field_name}: {field_values.get(field_name, '').strip()}")
    report_lines.extend(
        [
            "",
            "val_task_accuracies:",
            json.dumps(
                {
                    key.removeprefix("val_").removesuffix("_exact_accuracy"): value
                    for key, value in final_summary.items()
                    if key.startswith("val_")
                    and key.endswith("_exact_accuracy")
                    and key not in {"val_exact_accuracy", "val_macro_exact_accuracy"}
                },
                indent=2,
                sort_keys=True,
            ),
            "",
            "ood_task_accuracies:",
            json.dumps(
                {
                    key.removeprefix("ood_").removesuffix("_exact_accuracy"): value
                    for key, value in final_summary.items()
                    if key.startswith("ood_")
                    and key.endswith("_exact_accuracy")
                    and key not in {"ood_exact_accuracy", "ood_macro_exact_accuracy"}
                },
                indent=2,
                sort_keys=True,
            ),
            "",
            f"checkpoint_path: {checkpoint_path}",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    logger.log("run_finished", **final_summary)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one E1 baseline")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--eval-config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tasks", default="e1", choices=["e1", "phase1"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task_list = PHASE1_TASKS if args.tasks == "phase1" else E1_TASKS
    run_dir = run_training(
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        eval_config_path=args.eval_config,
        device_override=args.device,
        epochs_override=args.epochs,
        seed_override=args.seed,
        tasks=task_list,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
