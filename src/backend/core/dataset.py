from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

YAML_NAMES = ("data.yaml", "dataset.yaml", "detect.yaml")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def inspect_dataset(dataset_root: str) -> list[str]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    candidates: list[str] = []
    for yaml_name in YAML_NAMES:
        candidates.extend(str(path.resolve()) for path in root.rglob(yaml_name))
    return sorted(set(candidates))


def analyze_dataset(dataset_yaml: str) -> dict[str, Any]:
    yaml_path = Path(dataset_yaml).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset yaml not found: {dataset_yaml}")

    config = _parse_dataset_yaml(yaml_path)
    base_path = _resolve_dataset_base_path(yaml_path, config.get("path"))
    warnings: list[str] = []
    split_stats: dict[str, Any] = {}
    aggregate_counts: dict[int, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0})

    for split_name in ("train", "val"):
        split_value = str(config.get(split_name, "") or "").strip()
        stats = _empty_split_stats()
        if not split_value:
            warnings.append(f"{split_name}_split_missing")
            split_stats[split_name] = stats
            continue

        image_dir = _resolve_split_path(base_path, yaml_path.parent, split_value)
        label_dir = _resolve_label_dir(image_dir, split_value)
        stats["image_dir"] = str(image_dir)
        stats["label_dir"] = str(label_dir)

        if not image_dir.exists():
            warnings.append(f"{split_name}_image_dir_missing")
            split_stats[split_name] = stats
            continue
        if not label_dir.exists():
            warnings.append(f"{split_name}_label_dir_missing")
            stats["image_count"] = _count_images(image_dir)
            split_stats[split_name] = stats
            continue

        stats["image_count"] = _count_images(image_dir)
        label_files = sorted(path for path in label_dir.rglob("*.txt") if path.is_file())
        stats["label_file_count"] = len(label_files)
        split_instance_count = 0
        for label_file in label_files:
            class_ids = _read_label_class_ids(label_file)
            split_instance_count += len(class_ids)
            for class_id in class_ids:
                aggregate_counts[class_id][split_name] += 1
        stats["instance_count"] = split_instance_count
        split_stats[split_name] = stats

    names_map = _normalize_names(config.get("names"), warnings)
    classes: list[dict[str, Any]] = []
    total_instances = sum(stats["train"] + stats["val"] for stats in aggregate_counts.values())
    for class_id in sorted(aggregate_counts):
        if class_id not in names_map:
            warnings.append(f"unknown_class_id:{class_id}")
        train_instances = aggregate_counts[class_id]["train"]
        val_instances = aggregate_counts[class_id]["val"]
        total = train_instances + val_instances
        classes.append(
            {
                "class_id": class_id,
                "class_name": names_map.get(class_id, f"class_{class_id}"),
                "train_instances": train_instances,
                "val_instances": val_instances,
                "total_instances": total,
                "total_ratio": 0.0 if total_instances <= 0 else round(total / total_instances, 6),
            }
        )

    if config.get("names") in (None, "", {}):
        warnings.append("names_missing")

    return {
        "dataset_yaml": str(yaml_path),
        "splits": split_stats,
        "classes": classes,
        "totals": {
            "class_count": len(classes),
            "train_instances": split_stats.get("train", {}).get("instance_count", 0),
            "val_instances": split_stats.get("val", {}).get("instance_count", 0),
            "total_instances": total_instances,
        },
        "warnings": sorted(set(warnings)),
    }


def _empty_split_stats() -> dict[str, Any]:
    return {
        "image_dir": "",
        "label_dir": "",
        "image_count": 0,
        "label_file_count": 0,
        "instance_count": 0,
    }


def _resolve_dataset_base_path(dataset_yaml: Path, raw_path: Any) -> Path:
    value = str(raw_path or "").strip()
    if not value:
        return dataset_yaml.parent
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (dataset_yaml.parent / path).resolve()


def _resolve_split_path(base_path: Path, yaml_dir: Path, split_value: str) -> Path:
    split_path = Path(split_value)
    if split_path.is_absolute():
        return split_path.resolve()
    candidate = (base_path / split_path).resolve()
    if candidate.exists():
        return candidate
    return (yaml_dir / split_path).resolve()


def _resolve_label_dir(image_dir: Path, split_value: str) -> Path:
    image_parts = [part.lower() for part in image_dir.parts]
    if "images" in image_parts:
        parts = list(image_dir.parts)
        for index, part in enumerate(parts):
            if part.lower() == "images":
                parts[index] = "labels"
                return Path(*parts)
    split_name = Path(split_value).name or image_dir.name
    return image_dir.parent / "labels" / split_name


def _count_images(image_dir: Path) -> int:
    return sum(1 for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _read_label_class_ids(label_file: Path) -> list[int]:
    class_ids: list[int] = []
    for line in label_file.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        class_token = stripped.split()[0]
        try:
            class_ids.append(int(float(class_token)))
        except ValueError:
            continue
    return class_ids


def _parse_dataset_yaml(dataset_yaml: Path) -> dict[str, Any]:
    lines = dataset_yaml.read_text(encoding="utf-8").splitlines()
    data: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            index += 1
            continue
        if ":" not in raw_line:
            index += 1
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key != "names":
            if value:
                data[key] = _parse_scalar(value)
            index += 1
            continue

        if value:
            data["names"] = _parse_inline_names(value)
            index += 1
            continue

        block_items: list[str] = []
        index += 1
        while index < len(lines):
            nested = lines[index]
            nested_stripped = nested.strip()
            nested_indent = len(nested) - len(nested.lstrip(" "))
            if not nested_stripped:
                index += 1
                continue
            if nested_indent <= indent:
                break
            block_items.append(nested[nested_indent:])
            index += 1
        data["names"] = _parse_block_names(block_items)
    return data


def _parse_inline_names(value: str) -> Any:
    stripped = value.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1].strip()
        if not inner:
            return []
        return [_strip_quotes(part.strip()) for part in inner.split(",")]
    if stripped.startswith("{") and stripped.endswith("}"):
        inner = stripped[1:-1].strip()
        result: dict[int, str] = {}
        if not inner:
            return result
        for part in inner.split(","):
            key, item_value = part.split(":", 1)
            result[int(key.strip())] = _strip_quotes(item_value.strip())
        return result
    return stripped


def _parse_block_names(lines: list[str]) -> Any:
    if not lines:
        return {}
    if all(line.lstrip().startswith("-") for line in lines):
        values: list[str] = []
        for line in lines:
            item = line.lstrip()[1:].strip()
            values.append(_strip_quotes(item))
        return values
    result: dict[int, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            result[int(key.strip())] = _strip_quotes(value.strip())
        except ValueError:
            continue
    return result


def _normalize_names(raw_names: Any, warnings: list[str]) -> dict[int, str]:
    if isinstance(raw_names, list):
        return {index: str(value) for index, value in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        normalized: dict[int, str] = {}
        for key, value in raw_names.items():
            try:
                normalized[int(key)] = str(value)
            except (TypeError, ValueError):
                warnings.append(f"invalid_name_key:{key}")
        return normalized
    if raw_names not in (None, ""):
        warnings.append("names_unsupported")
    return {}


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    if (stripped.startswith("'") and stripped.endswith("'")) or (
        stripped.startswith('"') and stripped.endswith('"')
    ):
        return stripped[1:-1]
    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    return stripped


def _strip_quotes(value: str) -> str:
    stripped = value.strip()
    if (stripped.startswith("'") and stripped.endswith("'")) or (
        stripped.startswith('"') and stripped.endswith('"')
    ):
        return stripped[1:-1]
    return stripped
