"""Capture validation predictions without changing metric inputs or running predict again."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def capture_validator(task: str, confidence: float, emit: Callable[[str, Any, tuple[int, int]], None],
                      original_shapes: dict[str, tuple[int, int]] | None = None) -> type:
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.models.yolo.segment.val import SegmentationValidator
    from ultralytics.models.yolo.obb.val import OBBValidator
    from ultralytics.models.yolo.semantic.val import SemanticSegmentationValidator

    base = {"detection": DetectionValidator, "segment": SegmentationValidator,
            "obb": OBBValidator, "semantic": SemanticSegmentationValidator}[task]

    class CapturingValidator(base):
        def update_metrics(self, preds, batch):
            # Metrics see the complete low-confidence predictions, exactly as before.
            super().update_metrics(preds, batch)
            try:
                for index, path in enumerate(batch["im_file"]):
                    if task == "semantic":
                        shape = (original_shapes or {}).get(str(Path(path).resolve())) or self.image_shapes.get(path)
                        if shape is None:
                            raise RuntimeError(f"missing original image dimensions: {path}")
                        mask = preds[index].detach().cpu().numpy().copy()
                        emit(str(Path(path).resolve()), mask, tuple(shape))
                        continue
                    original_shape = tuple(batch["ori_shape"][index])
                    pred = preds[index]
                    keep = pred["conf"] >= confidence
                    # Clone before scaling: neither exports nor filtering may mutate metric inputs.
                    filtered = {key: value[keep].clone() for key, value in pred.items()}
                    prepared = {"imgsz": batch["img"].shape[2:], "ori_shape": original_shape,
                                "ratio_pad": batch["ratio_pad"][index]}
                    scaled = self.scale_preds(filtered, prepared) if keep.any() else filtered
                    emit(str(Path(path).resolve()), scaled, original_shape)
            except (KeyError, TypeError, AttributeError, IndexError) as exc:
                raise RuntimeError(f"validation prediction capture incompatible for {task}: {exc}") from exc

    return CapturingValidator


def serialize_validation_prediction(pred: dict[str, Any], shape: tuple[int, int], names: dict[int, str], task: str):
    import numpy as np
    import torch
    from ultralytics.engine.results import Results
    from backend.core.workbench_worker import _serialize_boxes

    # Results only needs image dimensions to turn masks into original-coordinate contours.
    image = np.empty((*shape, 0), dtype=np.uint8)
    boxes = torch.cat((pred["bboxes"], pred["conf"][:, None], pred["cls"][:, None]), dim=1).cpu()
    if task == "obb":
        result = Results(image, path="", names=names, obb=boxes)
    else:
        masks = pred.get("masks")
        result = Results(image, path="", names=names, boxes=boxes,
                         masks=masks.cpu() if masks is not None and len(boxes) else None)
    return _serialize_boxes(result, names, task)
