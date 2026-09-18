from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from backend.constants import PER_CLASS_METRICS_FILENAME
from backend.core.metrics import TASK_METRIC_PROFILES, calculate_fitness, metric_column_names
from backend.models import Summary


def _safe_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    except (TypeError, ValueError):
        return None


def _load_results_rows(results_csv: Path) -> list[dict[str, Any]]:
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, Any]] = []
        for row in reader:
            epoch = _safe_float(row.get("epoch"))
            if epoch is None:
                continue
            if all(value in ("", None) for key, value in row.items() if key != "epoch"):
                continue
            rows.append(row)
        return rows


def _column_value(row: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in row:
            value = _safe_float(row[name])
            if value is not None:
                return value
    return None


def _metric_column_names(metric_name: str, suffixes: tuple[str, ...]) -> tuple[str, ...]:
    return metric_column_names(metric_name, suffixes)


def _extract_metric_set(row: dict[str, Any], suffixes: tuple[str, ...]) -> dict[str, float | None]:
    return {
        "precision": _column_value(row, _metric_column_names("precision", suffixes)),
        "recall": _column_value(row, _metric_column_names("recall", suffixes)),
        "map50": _column_value(row, _metric_column_names("mAP50", suffixes)),
        "map50_95": _column_value(row, _metric_column_names("mAP50-95", suffixes)),
    }


def _normalize_metric_set(metrics: dict[str, float | None]) -> dict[str, float]:
    return {
        key: round(float(value or 0.0), 6)
        for key, value in metrics.items()
    }


def _load_per_class_metrics(run_path: Path) -> list[dict[str, Any]]:
    metrics_path = run_path / PER_CLASS_METRICS_FILENAME
    if not metrics_path.exists():
        return []
    try:
        import json

        raw_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(raw_metrics, list):
        return []
    return [item for item in raw_metrics if isinstance(item, dict)]


def _loss_trend(rows: list[dict[str, Any]], loss_names: tuple[str, ...]) -> str:
    if len(rows) < 3:
        return "unknown"

    values = [_column_value(row, loss_names) for row in rows]
    values = [value for value in values if value is not None]
    if len(values) < 3:
        return "unknown"

    deltas = [values[index + 1] - values[index] for index in range(len(values) - 1)]
    positives = sum(1 for delta in deltas if delta > 0)
    negatives = sum(1 for delta in deltas if delta < 0)
    if negatives >= max(2, len(deltas) * 0.7):
        return "stable_down"
    if positives >= max(2, len(deltas) * 0.7):
        return "diverging"
    return "oscillating"


def _detect_overfitting(
    rows: list[dict[str, Any]],
    train_loss_names: tuple[str, ...],
    metric_suffixes: tuple[str, ...],
) -> str:
    if len(rows) < 5:
        return "none"
    tail = rows[-5:]
    train_losses = [_column_value(row, train_loss_names) for row in tail]
    metrics = [_column_value(row, _metric_column_names("mAP50-95", metric_suffixes)) for row in tail]
    train_losses = [value for value in train_losses if value is not None]
    metrics = [value for value in metrics if value is not None]
    if len(train_losses) < 3 or len(metrics) < 3:
        return "none"
    if train_losses[-1] < train_losses[0] and metrics[-1] <= metrics[0]:
        return "mild"
    return "none"


def _plateau(
    rows: list[dict[str, Any]],
    metric_suffixes: tuple[str, ...],
    min_delta: float = 0.002,
) -> tuple[bool, int | None]:
    if len(rows) < 6:
        return False, None
    values = [_column_value(row, _metric_column_names("mAP50-95", metric_suffixes)) for row in rows]
    values = [value for value in values if value is not None]
    if len(values) < 6:
        return False, None
    midpoint = len(values) // 2
    old_best = max(values[:midpoint])
    recent_best = max(values[midpoint:])
    if recent_best - old_best < min_delta:
        plateau_epoch = values.index(recent_best) + 1
        return True, plateau_epoch
    return False, None


def _semantic_summary(trial_id, run_path, rows, params, previous_summary):
    scored = [(row, calculate_fitness(row, "semantic")) for row in rows]
    valid = [(row, value) for row, value in scored if value is not None]
    best, fitness = max(valid, key=lambda item: item[1]) if valid else (rows[-1], None)
    metrics = {"miou": _column_value(best, ("metrics/mIoU",)),
               "pixel_accuracy": _column_value(best, ("metrics/pixel_acc",))}
    previous = (previous_summary or {}).get("final_metrics", {})
    delta = {key: round(value - previous[key], 6)
             if value is not None and isinstance(previous.get(key), (float, int)) else None
             for key, value in metrics.items()}
    loss_columns = tuple(key for key in rows[-1] if key.startswith("train/") and "loss" in key)
    total_time = _column_value(rows[-1], ("time",))
    # Reuse the trend checks with mIoU as their score, without publishing detection metrics.
    trend_rows = [{**row, "metrics/mAP50-95": row.get("metrics/mIoU")} for row in rows]
    plateau, plateau_epoch = _plateau(trend_rows, ("",))
    overfitting = _detect_overfitting(trend_rows, loss_columns, ("",))
    return Summary(
        trial_id=trial_id,
        basic_info={"epochs_planned": params["epochs"], "epochs_completed": len(rows),
                    "early_stop": len(rows) < int(params["epochs"]), "best_epoch": int(best["epoch"]),
                    "train_time_sec": total_time},
        metric_context={"task_type": "semantic", "primary_component": "semantic",
                        "available_components": ["semantic"], "selection_metric": "mIoU", "selection_fitness": fitness},
        final_metrics=metrics, per_class_metrics=_load_per_class_metrics(run_path),
        metric_breakdown={"semantic": metrics}, delta_vs_prev=delta,
        metric_breakdown_delta_vs_prev={"semantic": delta},
        training_dynamics={"loss_trend": _loss_trend(rows, loss_columns), "plateau": plateau,
                           "plateau_epoch": plateau_epoch, "overfitting": overfitting, "primary_component": "semantic"},
        warnings=["possible_overfitting"] if overfitting != "none" else [],
        resource={"avg_epoch_time": total_time / len(rows) if total_time is not None else None,
                  "gpu_mem_peak": _column_value(best, ("gpu_mem",))}, params=params,
    )


def build_summary(
    trial_id: str,
    task_type: str,
    run_dir: str,
    params: dict[str, Any],
    previous_summary: dict[str, Any] | None = None,
) -> Summary:
    run_path = Path(run_dir)
    results_csv = run_path / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found in run dir: {run_dir}")

    rows = _load_results_rows(results_csv)
    if not rows:
        raise ValueError("results.csv is empty")

    if task_type == "semantic":
        return _semantic_summary(trial_id, run_path, rows, params, previous_summary)

    profile = TASK_METRIC_PROFILES.get(task_type, TASK_METRIC_PROFILES["detection"])
    primary_component = profile["primary_component"]
    component_specs = profile["components"]
    primary_spec = component_specs[primary_component]
    scored_rows = [
        (index, row, calculate_fitness(row, task_type))
        for index, row in enumerate(rows)
    ]
    valid_scored_rows = [item for item in scored_rows if item[2] is not None]
    if valid_scored_rows:
        best_index, best_row, best_fitness = max(valid_scored_rows, key=lambda item: item[2])
    else:
        best_index, best_row, best_fitness = 0, rows[0], None
    best_epoch = int(_column_value(best_row, ("epoch",)) or best_index + 1)
    metric_breakdown_raw = {
        component: _extract_metric_set(best_row, spec["metric_suffixes"])
        for component, spec in component_specs.items()
    }
    available_components = [
        component
        for component, metrics in metric_breakdown_raw.items()
        if any(value is not None for value in metrics.values())
    ]
    primary_metrics = metric_breakdown_raw[primary_component]
    metric_breakdown = {
        component: _normalize_metric_set(metrics)
        for component, metrics in metric_breakdown_raw.items()
        if component in available_components
    }
    plateau, plateau_epoch = _plateau(rows, primary_spec["metric_suffixes"])
    overfitting = _detect_overfitting(rows, primary_spec["train_loss"], primary_spec["metric_suffixes"])
    total_time = _column_value(rows[-1], ("time",))
    epoch_time = _column_value(rows[-1], ("epoch_time",))
    if total_time is not None:
        train_time_sec = round(total_time, 3)
        avg_epoch_time = round(total_time / len(rows), 6)
    elif epoch_time is not None:
        train_time_sec = round(epoch_time * len(rows), 3)
        avg_epoch_time = epoch_time
    else:
        train_time_sec = None
        avg_epoch_time = None
    gpu_mem = _column_value(best_row, ("gpu_mem",))
    warnings: list[str] = []
    if gpu_mem and gpu_mem > 10_240:
        warnings.append("gpu_memory_high")
    if overfitting != "none":
        warnings.append("possible_overfitting")

    prev_metrics = (previous_summary or {}).get("final_metrics", {})
    prev_breakdown = (previous_summary or {}).get("metric_breakdown", {})
    delta_vs_prev = {
        "map50_95": round(float(primary_metrics.get("map50_95") or 0.0) - float(prev_metrics.get("map50_95", 0.0)), 6),
        "recall": round(float(primary_metrics.get("recall") or 0.0) - float(prev_metrics.get("recall", 0.0)), 6),
    }
    metric_breakdown_delta_vs_prev = {
        component: {
            "map50_95": round(
                float(metrics.get("map50_95", 0.0)) - float(prev_breakdown.get(component, {}).get("map50_95", 0.0)),
                6,
            ),
            "recall": round(
                float(metrics.get("recall", 0.0)) - float(prev_breakdown.get(component, {}).get("recall", 0.0)),
                6,
            ),
        }
        for component, metrics in metric_breakdown.items()
    }
    return Summary(
        trial_id=trial_id,
        basic_info={
            "epochs_planned": params["epochs"],
            "epochs_completed": len(rows),
            "early_stop": len(rows) < int(params["epochs"]),
            "best_epoch": best_epoch,
            "train_time_sec": train_time_sec,
        },
        metric_context={
            "task_type": task_type,
            "primary_component": primary_component,
            "available_components": available_components,
            "selection_metric": profile["selection_metric"],
            "selection_fitness": None if best_fitness is None else round(best_fitness, 6),
        },
        final_metrics={
            "precision": round(float(primary_metrics.get("precision") or 0.0), 6),
            "recall": round(float(primary_metrics.get("recall") or 0.0), 6),
            "map50": round(float(primary_metrics.get("map50") or 0.0), 6),
            "map50_95": round(float(primary_metrics.get("map50_95") or 0.0), 6),
        },
        per_class_metrics=_load_per_class_metrics(run_path),
        metric_breakdown=metric_breakdown,
        delta_vs_prev=delta_vs_prev,
        metric_breakdown_delta_vs_prev=metric_breakdown_delta_vs_prev,
        training_dynamics={
            "loss_trend": _loss_trend(rows, primary_spec["train_loss"]),
            "plateau": plateau,
            "plateau_epoch": plateau_epoch,
            "overfitting": overfitting,
            "primary_component": primary_component,
        },
        warnings=warnings,
        resource={
            "avg_epoch_time": avg_epoch_time,
            "gpu_mem_peak": gpu_mem,
        },
        params=params,
    )
