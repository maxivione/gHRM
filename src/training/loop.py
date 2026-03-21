from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from src.eval.metrics import collect_metrics
from src.telemetry.logger import read_peak_vram_mb


@dataclass(frozen=True)
class EpochResult:
    loss: float
    wall_clock_sec: float
    peak_vram_mb: float
    max_grad_norm: float
    instability_events: int
    metrics: dict[str, float]


def run_train_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device: torch.device,
    grad_clip_norm: float,
    amp_enabled: bool,
    ponder_coeff: float = 0.0,
) -> EpochResult:
    model.train()
    start_time = time.perf_counter()
    rows: list[dict] = []
    loss_total = 0.0
    max_grad_norm = 0.0
    instability_events = 0
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        targets = batch["targets"].to(device)

        with torch.autocast(device_type=device.type, enabled=amp_enabled and device.type == "cuda"):
            logits = model(input_ids=input_ids, lengths=lengths)
            loss = loss_fn(logits, targets)
            if ponder_coeff > 0.0:
                ponder_cost = getattr(model, "_last_ponder_cost", None)
                if ponder_cost is not None:
                    loss = loss + ponder_coeff * ponder_cost
        if not torch.isfinite(loss):
            instability_events += 1
            raise RuntimeError(
                f"Expected finite training loss on {device.type}, got {loss.item()} for tasks {sorted(set(batch['task_names']))}"
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        grad_norm_value = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        if not math.isfinite(grad_norm_value):
            instability_events += 1
            raise RuntimeError(
                f"Expected finite gradient norm on {device.type}, got {grad_norm_value} for tasks {sorted(set(batch['task_names']))}"
            )
        max_grad_norm = max(max_grad_norm, grad_norm_value)
        scaler.step(optimizer)
        scaler.update()

        loss_total += loss.item() * targets.size(0)
        predictions = logits.argmax(dim=-1).detach().cpu().tolist()
        for task_name, target, predicted in zip(batch["task_names"], targets.cpu().tolist(), predictions):
            rows.append({"task_name": task_name, "target": target, "predicted": predicted})

    metrics = collect_metrics(rows)
    metrics["loss"] = loss_total / len(loader.dataset)
    return EpochResult(
        loss=metrics["loss"],
        wall_clock_sec=time.perf_counter() - start_time,
        peak_vram_mb=read_peak_vram_mb(device),
        max_grad_norm=max_grad_norm,
        instability_events=instability_events,
        metrics=metrics,
    )


@torch.no_grad()
def run_eval_epoch(model, loader, loss_fn, device: torch.device) -> EpochResult:
    model.eval()
    start_time = time.perf_counter()
    rows: list[dict] = []
    loss_total = 0.0

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        targets = batch["targets"].to(device)

        logits = model(input_ids=input_ids, lengths=lengths)
        loss = loss_fn(logits, targets)
        loss_total += loss.item() * targets.size(0)

        predictions = logits.argmax(dim=-1).cpu().tolist()
        for task_name, target, predicted in zip(batch["task_names"], targets.cpu().tolist(), predictions):
            rows.append({"task_name": task_name, "target": target, "predicted": predicted})

    metrics = collect_metrics(rows)
    metrics["loss"] = loss_total / len(loader.dataset)
    return EpochResult(
        loss=metrics["loss"],
        wall_clock_sec=time.perf_counter() - start_time,
        peak_vram_mb=read_peak_vram_mb(device),
        max_grad_norm=0.0,
        instability_events=0,
        metrics=metrics,
    )
