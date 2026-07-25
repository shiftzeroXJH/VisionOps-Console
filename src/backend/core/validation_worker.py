from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
BOX_COLOR = (42, 157, 143)
PRED_COLOR = (231, 111, 81)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m backend.core.validation_worker <request_json>", file=sys.stderr)
        return 2

    request_path = Path(sys.argv[1])
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"failed to read validation request: {exc}", file=sys.stderr)
        return 2

    try:
        result = run_validation_preview(request)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:  # pragma: no cover
        print(str(exc), file=sys.stderr)
        return 1


def run_validation_preview(request: dict[str, Any]) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(f"validation dependencies are missing: {exc}") from exc

    del ImageDraw, ImageFont

    dataset_yaml = Path(request["dataset_yaml"]).resolve()
    dataset_config = _parse_dataset_yaml(dataset_yaml)
    base_path = _resolve_dataset_base_path(dataset_yaml, dataset_config.get("path"))
    split_value = str(dataset_config.get("val", "") or "").strip()
    if not split_value:
        raise RuntimeError("dataset yaml has no val split")
    image_dir = _resolve_split_path(base_path, dataset_yaml.parent, split_value)
    label_dir = _resolve_label_dir(image_dir, split_value)
    if not image_dir.exists():
        raise RuntimeError(f"val image dir not found: {image_dir}")

    image_limit = int(request["image_limit"])
    conf = float(request["conf"])
    imgsz = int(request["imgsz"])
    batch = int(request["batch"])
    names = _normalize_names(dataset_config.get("names"))
    output_dir = Path(request["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(request["model_path"])
    metrics = _extract_metrics(
        model.val(
            data=str(dataset_yaml),
            split="val",
            imgsz=imgsz,
            batch=batch,
            plots=False,
            save=False,
            verbose=False,
        ),
        str(request.get("task_type", "detection")),
    )

    image_paths = sorted(
        path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )[:image_limit]
    rendered: list[dict[str, Any]] = []
    for index, image_path in enumerate(image_paths, start=1):
        label_path = _label_path_for_image(image_path, image_dir, label_dir)
        with Image.open(image_path) as opened:
            base = opened.convert("RGB")
        label_image = base.copy()
        predict_image = base.copy()
        _draw_label_file(label_image, label_path, names, request["task_type"])
        prediction = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            verbose=False,
            save=False,
        )[0]
        _draw_prediction(predict_image, prediction, names, request["task_type"])
        base_filename = f"{index:04d}_{_safe_stem(image_path.stem)}"
        label_filename = f"{base_filename}_label.jpg"
        predict_filename = f"{base_filename}_predict.jpg"
        label_image.save(output_dir / label_filename, quality=92)
        predict_image.save(output_dir / predict_filename, quality=92)
        rendered.append(
            {
                "label_filename": label_filename,
                "predict_filename": predict_filename,
                "source_image": str(image_path),
                "label_file": str(label_path) if label_path.exists() else "",
            }
        )

    return {
        "metrics": metrics,
        "images": rendered,
    }


def _extract_metrics(result: Any, task_type: str) -> dict[str, float]:
    primary_component = {
        "segment": "seg",
        "obb": "obb",
    }.get(task_type, "box")
    components = (primary_component,) + tuple(
        component for component in ("box", "seg", "obb") if component != primary_component
    )
    metrics: dict[str, float] = {}
    for key, attribute in {
        "precision": "mp",
        "recall": "mr",
        "map50": "map50",
        "map50_95": "map",
    }.items():
        for component in components:
            candidate = f"{component}.{attribute}"
            value = _nested_attr(result, candidate)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                metrics[key] = float(value)
                break
    return metrics


def _nested_attr(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _draw_label_file(image: Any, label_path: Path, names: dict[int, str], task_type: str) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    width, height = image.size
    if not label_path.exists():
        _draw_corner_text(draw, "No label", BOX_COLOR)
        return
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            values = [float(part) for part in parts[1:]]
        except ValueError:
            continue
        label = names.get(class_id, f"class_{class_id}")
        if task_type in {"segment", "obb"} and len(values) >= 6:
            points = _normalized_points(values, width, height)
            _draw_polygon(draw, points, BOX_COLOR, label)
        else:
            cx, cy, box_w, box_h = values[:4]
            x1 = (cx - box_w / 2) * width
            y1 = (cy - box_h / 2) * height
            x2 = (cx + box_w / 2) * width
            y2 = (cy + box_h / 2) * height
            _draw_box(draw, (x1, y1, x2, y2), BOX_COLOR, label)


def _draw_prediction(image: Any, prediction: Any, names: dict[int, str], task_type: str) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    width, height = image.size
    if task_type == "segment" and getattr(prediction, "masks", None) is not None and prediction.masks is not None:
        boxes = _boxes_list(prediction)
        polygons = getattr(prediction.masks, "xyn", []) or []
        for index, polygon in enumerate(polygons):
            class_id, conf = _box_class_conf(boxes[index] if index < len(boxes) else None)
            label = _prediction_label(names, class_id, conf)
            _draw_polygon(draw, [(float(x) * width, float(y) * height) for x, y in polygon], PRED_COLOR, label)
        return
    if task_type == "obb" and getattr(prediction, "obb", None) is not None and prediction.obb is not None:
        obb = prediction.obb
        classes = _tensor_list(getattr(obb, "cls", None))
        confs = _tensor_list(getattr(obb, "conf", None))
        polygons = _tensor_list(getattr(obb, "xyxyxyxy", None))
        for index, polygon in enumerate(polygons):
            class_id = int(classes[index]) if index < len(classes) else None
            conf = float(confs[index]) if index < len(confs) else None
            flat = polygon.reshape(-1).tolist() if hasattr(polygon, "reshape") else polygon
            points = [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat) - 1, 2)]
            _draw_polygon(draw, points, PRED_COLOR, _prediction_label(names, class_id, conf))
        return
    for box in _boxes_list(prediction):
        coords = _tensor_list(getattr(box, "xyxy", None))
        if coords and isinstance(coords[0], list):
            coords = coords[0]
        if len(coords) < 4:
            continue
        class_id, conf = _box_class_conf(box)
        _draw_box(draw, tuple(float(v) for v in coords[:4]), PRED_COLOR, _prediction_label(names, class_id, conf))


def _boxes_list(prediction: Any) -> list[Any]:
    boxes = getattr(prediction, "boxes", None)
    if boxes is None:
        return []
    try:
        return list(boxes)
    except TypeError:
        return []


def _box_class_conf(box: Any) -> tuple[int | None, float | None]:
    if box is None:
        return None, None
    cls_values = _tensor_list(getattr(box, "cls", None))
    conf_values = _tensor_list(getattr(box, "conf", None))
    class_id = int(cls_values[0]) if cls_values else None
    conf = float(conf_values[0]) if conf_values else None
    return class_id, conf


def _tensor_list(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        if hasattr(value, "tolist"):
            output = value.tolist()
        else:
            output = list(value)
        return output if isinstance(output, list) else [output]
    except Exception:
        return []


def _draw_box(draw: Any, box: tuple[float, float, float, float], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    _draw_text(draw, (x1 + 3, max(0, y1 - 16)), label, color)


def _draw_polygon(draw: Any, points: list[tuple[float, float]], color: tuple[int, int, int], label: str) -> None:
    if len(points) < 2:
        return
    draw.line(points + [points[0]], fill=color, width=3)
    _draw_text(draw, points[0], label, color)


def _draw_text(draw: Any, position: tuple[float, float], text: str, color: tuple[int, int, int]) -> None:
    x, y = position
    bbox = draw.textbbox((x, y), text)
    draw.rectangle((bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1), fill=color)
    draw.text((x, y), text, fill=(255, 255, 255))


def _draw_corner_text(draw: Any, text: str, color: tuple[int, int, int]) -> None:
    _draw_text(draw, (8, 8), text, color)


def _prediction_label(names: dict[int, str], class_id: int | None, conf: float | None) -> str:
    name = names.get(class_id, f"class_{class_id}") if class_id is not None else "object"
    return f"{name} {conf:.2f}" if conf is not None else name


def _normalized_points(values: list[float], width: int, height: int) -> list[tuple[float, float]]:
    return [(values[index] * width, values[index + 1] * height) for index in range(0, len(values) - 1, 2)]


def _label_path_for_image(image_path: Path, image_dir: Path, label_dir: Path) -> Path:
    relative = image_path.relative_to(image_dir)
    return (label_dir / relative).with_suffix(".txt")


def _safe_stem(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return cleaned[:80] or "image"


def _parse_dataset_yaml(dataset_yaml: Path) -> dict[str, Any]:
    from backend.core.dataset import _parse_dataset_yaml as parse_dataset_yaml

    return parse_dataset_yaml(dataset_yaml)


def _resolve_dataset_base_path(dataset_yaml: Path, raw_path: Any) -> Path:
    from backend.core.dataset import _resolve_dataset_base_path as resolve_dataset_base_path

    return resolve_dataset_base_path(dataset_yaml, raw_path)


def _resolve_split_path(base_path: Path, yaml_dir: Path, split_value: str) -> Path:
    from backend.core.dataset import _resolve_split_path as resolve_split_path

    return resolve_split_path(base_path, yaml_dir, split_value)


def _resolve_label_dir(image_dir: Path, split_value: str) -> Path:
    from backend.core.dataset import _resolve_label_dir as resolve_label_dir

    return resolve_label_dir(image_dir, split_value)


def _normalize_names(raw_names: Any) -> dict[int, str]:
    from backend.core.dataset import _normalize_names as normalize_names

    warnings: list[str] = []
    return normalize_names(raw_names, warnings)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

