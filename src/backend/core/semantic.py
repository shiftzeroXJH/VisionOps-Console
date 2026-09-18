"""Semantic dataset and raster adapters shared by training and workbench workers."""
from __future__ import annotations

import base64
import io
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path


def semantic_dataset(yaml_path, split="val"):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset, add_polygon_background
    data = add_polygon_background(check_det_dataset(str(yaml_path), autodownload=False))
    cfg = get_cfg(overrides={"task": "semantic", "imgsz": 64, "cache": False, "workers": 0})
    dataset = build_yolo_dataset(cfg, data[split], 1, data, mode="val")
    import numpy as np
    valid_ids = set(range(max(2, data["nc"]))) | {255}
    for index, label in enumerate(dataset.labels):
        mask = dataset.load_mask(index, image_shape=tuple(label["shape"]))
        if mask.shape != tuple(label["shape"]):
            raise ValueError(f"Semantic mask dimensions differ from image: {label['im_file']}")
        invalid = set(int(value) for value in np.unique(mask)) - valid_ids
        if invalid:
            raise ValueError(f"Invalid semantic class IDs {sorted(invalid)}: {label['im_file']}")
    if data.get("masks_dir"):
        # Ultralytics skips corrupt image/mask pairs; surface omissions instead of silently changing the dataset.
        all_images = dataset.get_img_files(data[split])
        if len(all_images) != len(dataset.labels):
            raise ValueError("Semantic dataset contains missing or corrupt images/masks; see dataset scan warnings")
    return dataset, data


def display_names(names):
    names = {int(key): str(value) for key, value in names.items()}
    return {0: "background", 1: next(iter(names.values()))} if len(names) == 1 else names


class SemanticLabels(Mapping):
    """Load one full-resolution mask at a time, including for large evaluation sets."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = {str(Path(label["im_file"]).resolve()): index for index, label in enumerate(dataset.labels)}

    def __getitem__(self, path):
        index = self.indices[path]
        return self.dataset.load_mask(index, image_shape=tuple(self.dataset.labels[index]["shape"]))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def label_masks(yaml_path):
    dataset, data = semantic_dataset(yaml_path)
    return SemanticLabels(dataset), display_names(data["names"])


def analyze_semantic_dataset(yaml_path):
    import numpy as np
    classes, splits, warnings = {}, {}, []
    for split in ("train", "val"):
        dataset, data = semantic_dataset(yaml_path, split)
        names = display_names(data["names"])
        total, ignored = 0, 0
        for class_id, name in names.items():
            classes.setdefault(class_id, {"class_id": class_id, "class_name": name,
                                         "train_pixels": 0, "val_pixels": 0, "train_images": 0, "val_images": 0})
        for index, label in enumerate(dataset.labels):
            mask = dataset.load_mask(index, image_shape=tuple(label["shape"]))
            ignored += int((mask == 255).sum())
            for raw_id, count in zip(*np.unique(mask, return_counts=True)):
                class_id = int(raw_id)
                if class_id == 255:
                    continue
                if class_id not in classes:
                    warnings.append(f"unknown_class_id:{class_id}")
                    continue
                classes[class_id][f"{split}_pixels"] += int(count)
                classes[class_id][f"{split}_images"] += 1
                total += int(count)
        splits[split] = {"image_count": len(dataset.labels), "label_file_count": len(dataset.labels),
                         "pixel_count": total, "ignored_pixels": ignored}
    total = sum(item["pixel_count"] for item in splits.values())
    for row in classes.values():
        row["total_pixels"] = row["train_pixels"] + row["val_pixels"]
        row["total_ratio"] = row["total_pixels"] / total if total else 0
    return {"task_type": "semantic", "dataset_yaml": str(yaml_path), "splits": splits,
            "classes": list(classes.values()), "warnings": sorted(set(warnings)),
            "totals": {"class_count": len(classes), "total_pixels": total,
                       "train_pixels": splits["train"]["pixel_count"], "val_pixels": splits["val"]["pixel_count"]}}


def serialize_mask(mask, names, output_path, image_size=None, roi=None, crop_size=None):
    import numpy as np
    from PIL import Image
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    raster = Image.fromarray(np.asarray(mask, dtype=np.uint8))
    if roi is not None and image_size is not None:
        cw, ch = crop_size
        angle = math.radians(roi["angle"])
        c, s = math.cos(angle), math.sin(angle)
        sx, sy = cw / roi["width"], ch / roi["height"]
        transform = (c * sx, s * sx, cw / 2 - (c * roi["cx"] + s * roi["cy"]) * sx,
                     -s * sy, c * sy, ch / 2 + (s * roi["cx"] - c * roi["cy"]) * sy)
        raster = raster.transform(image_size, Image.Transform.AFFINE, transform,
                                  resample=Image.Resampling.NEAREST, fillcolor=255)
    elif image_size and raster.size != image_size:
        raster = raster.resize(image_size, Image.Resampling.NEAREST)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raster.save(output_path)
    values = np.asarray(raster)
    valid = int((values != 255).sum())
    palette = [(34, 197, 94), (56, 189, 248), (245, 158, 11), (244, 63, 94), (167, 139, 250), (20, 184, 166)]
    layers = []
    for class_id, name in display_names(names).items():
        pixels = int((values == class_id).sum())
        if not pixels:
            continue
        rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
        rgba[values == class_id] = (*palette[class_id % len(palette)], 255)
        buffer = io.BytesIO()
        Image.fromarray(rgba).save(buffer, format="PNG")
        layers.append({"class_id": class_id, "class_name": name, "pixels": pixels,
                       "ratio": pixels / valid if valid else 0,
                       "url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")})
    return {"width": raster.width, "height": raster.height, "mask_path": str(output_path), "layers": layers}


def overlay_mask(image, mask):
    import numpy as np
    from PIL import Image
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    values = np.asarray(mask)
    palette = np.asarray([(34, 197, 94), (56, 189, 248), (245, 158, 11), (244, 63, 94), (167, 139, 250)])
    rgba = np.zeros((*values.shape, 4), dtype=np.uint8)
    valid = values != 255
    rgba[valid, :3] = palette[values[valid] % len(palette)]
    rgba[valid, 3] = 125
    image.paste(Image.alpha_composite(image.convert("RGBA"), Image.fromarray(rgba).resize(image.size, Image.Resampling.NEAREST)).convert("RGB"))


if __name__ == "__main__":
    print(json.dumps(analyze_semantic_dataset(sys.argv[1]), ensure_ascii=True))
