from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from backend.constants import PER_CLASS_METRICS_FILENAME


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m backend.core.train_worker <request_json>", file=sys.stderr)
        return 2

    request_path = Path(sys.argv[1])
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"failed to read training request: {exc}", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print("ultralytics is not installed in the current environment", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1

    try:
        _train(request)
    except Exception as exc:  # pragma: no cover
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def _train(request: dict[str, Any]) -> None:
    from ultralytics import YOLO

    params = dict(request["params"])
    run_path = Path(request["run_dir"])
    run_path.mkdir(parents=True, exist_ok=True)
    model = YOLO(request["pretrained_model"])
    model.train(
        data=request["dataset_yaml"],
        epochs=int(params["epochs"]),
        patience=int(params["patience"]),
        imgsz=int(params["imgsz"]),
        batch=int(params["batch"]),
        workers=int(params["workers"]),
        device=0,
        optimizer=str(params["optimizer"]),
        lr0=float(params["lr0"]),
        lrf=float(params["lrf"]),
        momentum=float(params["momentum"]),
        weight_decay=float(params["weight_decay"]),
        warmup_epochs=float(params["warmup_epochs"]),
        cos_lr=bool(params["cos_lr"]),
        mosaic=float(params["mosaic"]),
        mixup=float(params["mixup"]),
        copy_paste=float(params["copy_paste"]),
        erasing=float(params["erasing"]),
        degrees=float(params["degrees"]),
        translate=float(params["translate"]),
        scale=float(params["scale"]),
        shear=float(params["shear"]),
        perspective=float(params["perspective"]),
        flipud=float(params["flipud"]),
        fliplr=float(params["fliplr"]),
        hsv_h=float(params["hsv_h"]),
        hsv_s=float(params["hsv_s"]),
        hsv_v=float(params["hsv_v"]),
        cache=False,
        seed=42,
        deterministic=True,
        pretrained=True,
        plots=True,
        save=True,
        save_period=10,
        val=True,
        project=str(run_path.parent),
        name=run_path.name,
        exist_ok=True,
        verbose=True,
    )
    _save_per_class_metrics(request, run_path)


def _save_per_class_metrics(request: dict[str, Any], run_path: Path) -> None:
    """Persist best-weight validation metrics without making training completion depend on them."""
    best_weight = run_path / "weights" / "best.pt"
    if not best_weight.exists():
        print("per-class metrics skipped: best.pt not found", file=sys.stderr)
        return
    try:
        from ultralytics import YOLO

        result = YOLO(str(best_weight)).val(
            data=request["dataset_yaml"],
            split="val",
            imgsz=int(request["params"]["imgsz"]),
            batch=int(request["params"]["batch"]),
            plots=False,
            save=False,
            verbose=False,
        )
        metrics = _extract_per_class_metrics(result, str(request.get("task_type", "detection")))
        (run_path / PER_CLASS_METRICS_FILENAME).write_text(
            json.dumps(metrics, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        print(f"per-class metrics skipped: {exc}", file=sys.stderr)


def _extract_per_class_metrics(result: Any, task_type: str) -> list[dict[str, Any]]:
    component_name = {"detection": "box", "segment": "seg", "obb": "obb"}.get(task_type, "box")
    component = getattr(result, component_name, None)
    if component is None:
        return []

    names = _class_names(getattr(result, "names", {}))
    precision = _numeric_list(getattr(component, "p", None))
    recall = _numeric_list(getattr(component, "r", None))
    map50 = _numeric_list(getattr(component, "ap50", None))
    map50_95 = _numeric_list(getattr(component, "maps", None))
    class_indexes = _class_indexes(getattr(component, "ap_class_index", None))
    if not class_indexes:
        class_indexes = list(range(max(len(precision), len(recall), len(map50))))

    metric_positions = {class_id: index for index, class_id in enumerate(class_indexes)}
    class_ids = sorted(set(names) | set(class_indexes))
    if not class_ids:
        return []

    rows: list[dict[str, Any]] = []
    for class_id in class_ids:
        metric_index = metric_positions.get(class_id)
        map_index = class_id if len(map50_95) > class_id else metric_index
        rows.append(
            {
                "class_id": class_id,
                "class_name": names.get(class_id, f"class_{class_id}"),
                "precision": _metric_value(precision, metric_index),
                "recall": _metric_value(recall, metric_index),
                "map50": _metric_value(map50, metric_index),
                "map50_95": _metric_value(map50_95, map_index),
            }
        )
    return rows


def _class_names(raw_names: Any) -> dict[int, str]:
    values = raw_names.items() if isinstance(raw_names, dict) else enumerate(raw_names or [])
    names: dict[int, str] = {}
    for raw_id, raw_name in values:
        try:
            names[int(raw_id)] = str(raw_name)
        except (TypeError, ValueError):
            continue
    return names


def _class_indexes(value: Any) -> list[int]:
    indexes: list[int] = []
    for item in _to_list(value):
        try:
            indexes.append(int(item))
        except (TypeError, ValueError):
            continue
    return indexes


def _numeric_list(value: Any) -> list[float | None]:
    return [_finite_float(item) for item in _to_list(value)]


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        return list(value)
    except TypeError:
        return [value]


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _metric_value(values: list[float | None], index: int | None) -> float | None:
    if index is None or index < 0 or index >= len(values):
        return None
    return values[index]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
