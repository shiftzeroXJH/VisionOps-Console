from __future__ import annotations

from typing import Any


TASK_METRIC_PROFILES: dict[str, dict[str, Any]] = {
    "detection": {
        "primary_component": "box",
        "fitness_components": ("box",),
        "selection_metric": "mAP50-95(B)",
        "components": {
            "box": {
                "metric_suffixes": ("B", ""),
                "train_loss": ("train/box_loss", "train/loss"),
                "val_loss": ("val/box_loss", "val/loss"),
            }
        },
    },
    "segment": {
        "primary_component": "mask",
        "fitness_components": ("box", "mask"),
        "selection_metric": "mAP50-95(B) + mAP50-95(M)",
        "components": {
            "box": {
                "metric_suffixes": ("B", ""),
                "train_loss": ("train/box_loss", "train/loss"),
                "val_loss": ("val/box_loss", "val/loss"),
            },
            "mask": {
                "metric_suffixes": ("M",),
                "train_loss": ("train/seg_loss", "train/loss"),
                "val_loss": ("val/seg_loss", "val/loss"),
            },
        },
    },
    "obb": {
        "primary_component": "obb",
        "fitness_components": ("obb",),
        "selection_metric": "mAP50-95(B)",
        "components": {
            "obb": {
                "metric_suffixes": ("O", "B", ""),
                "train_loss": ("train/box_loss", "train/loss"),
                "val_loss": ("val/box_loss", "val/loss"),
            }
        },
    },
}


def get_metric_profile(task_type: str) -> dict[str, Any]:
    return TASK_METRIC_PROFILES.get(task_type, TASK_METRIC_PROFILES["detection"])


def metric_column_names(metric_name: str, suffixes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        f"metrics/{metric_name}({suffix})" if suffix else f"metrics/{metric_name}"
        for suffix in suffixes
    )


def column_value(row: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name not in row or row[name] in ("", None):
            continue
        try:
            return float(row[name])
        except (TypeError, ValueError):
            continue
    return None


def fitness_metric(task_type: str) -> str:
    return get_metric_profile(task_type)["selection_metric"]


def calculate_fitness(row: dict[str, Any], task_type: str) -> float | None:
    """Calculate Ultralytics fitness for one results row."""
    profile = get_metric_profile(task_type)
    values: list[float] = []
    for component in profile["fitness_components"]:
        spec = profile["components"][component]
        value = column_value(row, metric_column_names("mAP50-95", spec["metric_suffixes"]))
        if value is None:
            return None
        values.append(value)
    return sum(values)
