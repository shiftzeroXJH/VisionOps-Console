from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m openclaw_yolo.core.train_worker <request_json>", file=sys.stderr)
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
