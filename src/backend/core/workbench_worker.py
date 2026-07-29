from __future__ import annotations

import json
import math
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image

from backend.core.dataset import (
    IMAGE_EXTENSIONS,
    _normalize_names,
    _parse_dataset_yaml,
    _resolve_dataset_base_path,
    _resolve_label_dir,
    _resolve_split_path,
)
from backend.core.train_worker import _extract_per_class_metrics
from backend.core.validation_worker import _extract_metrics
from backend.workbench import _normalize_roi


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m backend.core.workbench_worker <request_json>", file=sys.stderr)
        return 2
    try:
        request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        mode = request.get("mode")
        if mode == "infer":
            result = run_inference(request)
        elif mode == "inspect_dataset":
            result = inspect_evaluation_dataset(Path(request["dataset_path"]))
        elif mode == "evaluate":
            result = run_evaluation(request)
        else:
            raise RuntimeError(f"unsupported workbench mode: {mode}")
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:  # pragma: no cover - worker boundary
        print(str(exc), file=sys.stderr)
        return 1


def _load_model(model_path: str) -> Any:
    from ultralytics import YOLO

    model = YOLO(model_path)
    task = str(getattr(model, "task", "detect") or "detect")
    if task not in {"detect", "detection"}:
        raise RuntimeError(f"only detection models are supported, got: {task}")
    return model


def _names(raw: Any) -> dict[int, str]:
    warnings: list[str] = []
    return _normalize_names(raw, warnings)


def run_inference(request: dict[str, Any]) -> dict[str, Any]:
    model = _load_model(request["model_path"])
    model_names = _names(getattr(model, "names", {}))
    output: list[dict[str, Any]] = []
    for item in request.get("images", []):
        record = {"image_id": item["image_id"], "status": "completed", "detections": [], "error": ""}
        try:
            source, normalized_roi, crop_size = _inference_source(item["path"], item.get("roi"))
            result = model.predict(
                source=source,
                conf=float(request["conf"]),
                imgsz=int(request["imgsz"]),
                save=False,
                verbose=False,
            )[0]
            names = _names(getattr(result, "names", model_names)) or model_names
            detections = _serialize_boxes(result, names)
            record["detections"] = (
                [_map_roi_detection(detection, normalized_roi, crop_size) for detection in detections]
                if normalized_roi is not None else detections
            )
            model_names = names or model_names
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
        output.append(record)
    return {
        "classes": [{"class_id": class_id, "class_name": name} for class_id, name in sorted(model_names.items())],
        "images": output,
    }


def _roi_corners(roi: dict[str, float]) -> list[tuple[float, float]]:
    angle = math.radians(roi["angle"])
    ux = (math.cos(angle), math.sin(angle))
    uy = (-math.sin(angle), math.cos(angle))
    half_width = roi["width"] / 2
    half_height = roi["height"] / 2
    cx, cy = roi["cx"], roi["cy"]
    return [
        (cx - ux[0] * half_width - uy[0] * half_height, cy - ux[1] * half_width - uy[1] * half_height),
        (cx - ux[0] * half_width + uy[0] * half_height, cy - ux[1] * half_width + uy[1] * half_height),
        (cx + ux[0] * half_width + uy[0] * half_height, cy + ux[1] * half_width + uy[1] * half_height),
        (cx + ux[0] * half_width - uy[0] * half_height, cy + ux[1] * half_width - uy[1] * half_height),
    ]


def _inference_source(path: str, raw_roi: Any) -> tuple[Any, dict[str, float] | None, tuple[int, int] | None]:
    if raw_roi is None:
        return path, None, None
    with Image.open(path) as source:
        image = source.convert("RGB")
    roi = _normalize_roi(raw_roi, image.width, image.height)
    if roi is None:
        return path, None, None
    output_size = (max(2, round(roi["width"])), max(2, round(roi["height"])))
    corners = _roi_corners(roi)
    quad = tuple(value for point in corners for value in point)
    crop = image.transform(output_size, Image.Transform.QUAD, quad, resample=Image.Resampling.BICUBIC)
    return crop, roi, output_size


def _map_roi_detection(
    detection: dict[str, Any], roi: dict[str, float], crop_size: tuple[int, int] | None
) -> dict[str, Any]:
    if crop_size is None:
        return detection
    crop_width, crop_height = crop_size
    angle = math.radians(roi["angle"])
    ux = (math.cos(angle), math.sin(angle))
    uy = (-math.sin(angle), math.cos(angle))

    def map_point(x: float, y: float) -> list[float]:
        local_x = x / crop_width * roi["width"] - roi["width"] / 2
        local_y = y / crop_height * roi["height"] - roi["height"] / 2
        return [
            roi["cx"] + ux[0] * local_x + uy[0] * local_y,
            roi["cy"] + ux[1] * local_x + uy[1] * local_y,
        ]

    polygon = [
        map_point(detection["x1"], detection["y1"]),
        map_point(detection["x1"], detection["y2"]),
        map_point(detection["x2"], detection["y2"]),
        map_point(detection["x2"], detection["y1"]),
    ]
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return {**detection, "x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys), "polygon": polygon}


def inspect_evaluation_dataset(dataset_path: Path) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    if dataset_path.is_file():
        if dataset_path.suffix.lower() not in {".yaml", ".yml"}:
            raise RuntimeError("standard YOLO dataset must be selected by its YAML file")
        config = _parse_dataset_yaml(dataset_path)
        base = _resolve_dataset_base_path(dataset_path, config.get("path"))
        split = str(config.get("val", "") or "").strip()
        if not split:
            raise RuntimeError("dataset YAML has no val split")
        image_dir = _resolve_split_path(base, dataset_path.parent, split)
        label_dir = _resolve_label_dir(image_dir, split)
        images = _image_files(image_dir, recursive=True)
        names = _names(config.get("names"))
        if not images:
            raise RuntimeError(f"no validation images found: {image_dir}")
        return {
            "dataset_type": "yolo",
            "dataset_root": str(base),
            "dataset_yaml": str(dataset_path),
            "image_dir": str(image_dir),
            "label_dir": str(label_dir),
            "image_count": len(images),
            "classes": _class_rows(names),
        }

    images = _image_files(dataset_path, recursive=False)
    if not images:
        raise RuntimeError(f"no images found in dataset folder: {dataset_path}")
    xml_count = sum((path.with_suffix(".xml")).is_file() for path in images)
    json_count = sum((path.with_suffix(".json")).is_file() for path in images)
    if xml_count and json_count:
        raise RuntimeError("mixed XML and JSON labels are not supported in one folder")
    label_type = "voc" if xml_count else "labelme" if json_count else ""
    if not label_type:
        raise RuntimeError("no same-name XML or JSON labels found")
    missing = [path.name for path in images if not path.with_suffix(".xml" if label_type == "voc" else ".json").is_file()]
    if missing:
        sample = ", ".join(missing[:8])
        raise RuntimeError(f"missing labels for {len(missing)} images: {sample}")
    class_names: set[str] = set()
    instance_count = 0
    for image_path in images:
        boxes = _simple_labels(image_path, label_type)
        instance_count += len(boxes)
        class_names.update(str(box["class_name"]) for box in boxes)
    return {
        "dataset_type": label_type,
        "dataset_root": str(dataset_path),
        "image_dir": str(dataset_path),
        "image_count": len(images),
        "instance_count": instance_count,
        "classes": [{"class_name": name} for name in sorted(class_names)],
    }


def run_evaluation(request: dict[str, Any]) -> dict[str, Any]:
    dataset_path = Path(request["dataset_path"]).resolve()
    inspection = inspect_evaluation_dataset(dataset_path)
    model = _load_model(request["model_path"])
    model_names = _names(getattr(model, "names", {}))
    if not model_names:
        raise RuntimeError("model has no usable class metadata")

    adapter_dir: Path | None = None
    if inspection["dataset_type"] == "yolo":
        dataset_yaml = Path(inspection["dataset_yaml"])
        dataset_names = {row["class_id"]: row["class_name"] for row in inspection["classes"]}
        _validate_class_names(model_names, dataset_names)
        images = _image_files(Path(inspection["image_dir"]), recursive=True)
        label_dir = Path(inspection["label_dir"])
        image_dir = Path(inspection["image_dir"])
        label_loader = lambda image_path: _yolo_labels(image_path, image_dir, label_dir, model_names)
    else:
        dataset_names_set = {row["class_name"] for row in inspection["classes"]}
        unknown_names = dataset_names_set - set(model_names.values())
        if unknown_names:
            raise RuntimeError(f"dataset contains classes not present in model: {sorted(unknown_names)}")
        images = _image_files(dataset_path, recursive=False)
        label_type = inspection["dataset_type"]
        adapter_dir, dataset_yaml = _build_simple_adapter(
            dataset_path, images, label_type, model_names, request["evaluation_id"]
        )
        label_loader = lambda image_path: _simple_labels(image_path, label_type, model_names)

    try:
        validation = model.val(
            data=str(dataset_yaml),
            split="val",
            conf=0.001,
            imgsz=int(request["imgsz"]),
            batch=int(request["batch"]),
            workers=0,
            plots=False,
            save=False,
            verbose=False,
        )
        metrics = _extract_metrics(validation, "detection")
        per_class = _extract_per_class_metrics(validation, "detection")
        predictions_dir = Path(inspection["dataset_root"]) / "predictions_xml" / request["evaluation_id"]
        predictions_dir.mkdir(parents=True, exist_ok=False)
        image_results: list[dict[str, Any]] = []
        for index, image_path in enumerate(images):
            with Image.open(image_path) as image:
                width, height = image.size
            labels = label_loader(image_path)
            prediction = model.predict(
                source=str(image_path),
                conf=float(request["display_conf"]),
                imgsz=int(request["imgsz"]),
                save=False,
                verbose=False,
            )[0]
            names = _names(getattr(prediction, "names", model_names)) or model_names
            detections = _serialize_boxes(prediction, names)
            xml_path = predictions_dir / f"{image_path.stem}.xml"
            _write_prediction_xml(xml_path, image_path, width, height, detections)
            image_results.append(
                {
                    "image_id": f"eval_img_{index:06d}",
                    "name": image_path.name,
                    "source_path": str(image_path),
                    "width": width,
                    "height": height,
                    "labels": labels,
                    "detections": detections,
                    "xml_path": str(xml_path),
                }
            )
        return {
            "dataset": inspection,
            "classes": _class_rows(model_names),
            "metrics": metrics,
            "per_class_metrics": per_class,
            "images": image_results,
            "predictions_dir": str(predictions_dir),
        }
    finally:
        if adapter_dir is not None:
            shutil.rmtree(adapter_dir, ignore_errors=True)


def _serialize_boxes(result: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    detections: list[dict[str, Any]] = []
    for box in boxes:
        xyxy = _to_list(getattr(box, "xyxy", None))
        if xyxy and isinstance(xyxy[0], list):
            xyxy = xyxy[0]
        classes = _to_list(getattr(box, "cls", None))
        confidences = _to_list(getattr(box, "conf", None))
        if len(xyxy) < 4:
            continue
        class_id = int(classes[0]) if classes else -1
        detections.append(
            {
                "class_id": class_id,
                "class_name": names.get(class_id, f"class_{class_id}"),
                "confidence": float(confidences[0]) if confidences else None,
                "x1": float(xyxy[0]),
                "y1": float(xyxy[1]),
                "x2": float(xyxy[2]),
                "y2": float(xyxy[3]),
            }
        )
    return detections


def _simple_labels(image_path: Path, label_type: str, model_names: dict[int, str] | None = None) -> list[dict[str, Any]]:
    if label_type == "voc":
        root = ET.parse(image_path.with_suffix(".xml")).getroot()
        boxes = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            bounds = obj.find("bndbox")
            if not name or bounds is None:
                continue
            boxes.append(
                {
                    "class_name": name,
                    "x1": float(bounds.findtext("xmin", "0")),
                    "y1": float(bounds.findtext("ymin", "0")),
                    "x2": float(bounds.findtext("xmax", "0")),
                    "y2": float(bounds.findtext("ymax", "0")),
                }
            )
    else:
        data = json.loads(image_path.with_suffix(".json").read_text(encoding="utf-8"))
        boxes = []
        for shape in data.get("shapes", []):
            shape_type = str(shape.get("shape_type", "polygon"))
            if shape_type not in {"rectangle", "polygon"}:
                raise RuntimeError(f"unsupported LabelMe shape_type '{shape_type}' in {image_path.name}")
            points = shape.get("points") or []
            if len(points) < 2:
                continue
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            boxes.append(
                {
                    "class_name": str(shape.get("label", "")).strip(),
                    "x1": min(xs),
                    "y1": min(ys),
                    "x2": max(xs),
                    "y2": max(ys),
                }
            )
    if model_names:
        by_name = {name: class_id for class_id, name in model_names.items()}
        for box in boxes:
            if box["class_name"] not in by_name:
                raise RuntimeError(f"unknown dataset class: {box['class_name']}")
            box["class_id"] = by_name[box["class_name"]]
    return boxes


def _yolo_labels(image_path: Path, image_dir: Path, label_dir: Path, names: dict[int, str]) -> list[dict[str, Any]]:
    relative = image_path.relative_to(image_dir)
    label_path = (label_dir / relative).with_suffix(".txt")
    if not label_path.is_file():
        return []
    with Image.open(image_path) as image:
        width, height = image.size
    boxes: list[dict[str, Any]] = []
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, box_width, box_height = (float(value) for value in parts[1:5])
        boxes.append(
            {
                "class_id": class_id,
                "class_name": names.get(class_id, f"class_{class_id}"),
                "x1": (cx - box_width / 2) * width,
                "y1": (cy - box_height / 2) * height,
                "x2": (cx + box_width / 2) * width,
                "y2": (cy + box_height / 2) * height,
            }
        )
    return boxes


def _build_simple_adapter(
    root: Path,
    images: list[Path],
    label_type: str,
    names: dict[int, str],
    evaluation_id: str,
) -> tuple[Path, Path]:
    adapter = root / f".workbench_adapter_{evaluation_id}"
    image_dir = adapter / "images" / "val"
    label_dir = adapter / "labels" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index, source in enumerate(images):
        unique_name = f"{index:06d}_{source.name}"
        target = image_dir / unique_name
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        with Image.open(source) as image:
            width, height = image.size
        lines: list[str] = []
        for box in _simple_labels(source, label_type, names):
            box_width = max(0.0, box["x2"] - box["x1"])
            box_height = max(0.0, box["y2"] - box["y1"])
            cx = box["x1"] + box_width / 2
            cy = box["y1"] + box_height / 2
            lines.append(
                f"{box['class_id']} {cx / width:.8f} {cy / height:.8f} {box_width / width:.8f} {box_height / height:.8f}"
            )
        (label_dir / Path(unique_name).with_suffix(".txt")).write_text("\n".join(lines), encoding="utf-8")
    yaml_path = adapter / "data.yaml"
    names_yaml = ", ".join(json.dumps(names[index], ensure_ascii=False) for index in sorted(names))
    yaml_path.write_text(
        f"path: {adapter.as_posix()}\ntrain: images/val\nval: images/val\nnames: [{names_yaml}]\n",
        encoding="utf-8",
    )
    return adapter, yaml_path


def _write_prediction_xml(
    output: Path, image_path: Path, width: int, height: int, detections: list[dict[str, Any]]
) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = image_path.parent.name
    ET.SubElement(root, "filename").text = image_path.name
    ET.SubElement(root, "path").text = str(image_path)
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    for detection in detections:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = str(detection["class_name"])
        ET.SubElement(obj, "score").text = f"{float(detection['confidence'] or 0):.6f}"
        bounds = ET.SubElement(obj, "bndbox")
        ET.SubElement(bounds, "xmin").text = str(max(0, round(detection["x1"])))
        ET.SubElement(bounds, "ymin").text = str(max(0, round(detection["y1"])))
        ET.SubElement(bounds, "xmax").text = str(min(width, round(detection["x2"])))
        ET.SubElement(bounds, "ymax").text = str(min(height, round(detection["y2"])))
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


def _validate_class_names(model_names: dict[int, str], dataset_names: dict[int, str]) -> None:
    if model_names != dataset_names:
        _raise_class_mismatch(model_names, set(dataset_names.values()))


def _raise_class_mismatch(model_names: dict[int, str], dataset_names: set[str]) -> None:
    model_set = set(model_names.values())
    missing = sorted(model_set - dataset_names)
    unknown = sorted(dataset_names - model_set)
    raise RuntimeError(f"dataset/model classes differ; missing={missing}, unknown={unknown}")


def _image_files(root: Path, recursive: bool) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(path.resolve() for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _class_rows(names: dict[int, str]) -> list[dict[str, Any]]:
    return [{"class_id": class_id, "class_name": name} for class_id, name in sorted(names.items())]


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else [value]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
