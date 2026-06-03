from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m backend.core.export_worker <request_json>", file=sys.stderr)
        return 2

    request_path = Path(sys.argv[1])
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"failed to read export request: {exc}", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print("ultralytics is not installed in the current environment", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1

    try:
        model = YOLO(request["model_path"])
        exported_path = model.export(
            format="onnx",
            imgsz=int(request["imgsz"]),
            opset=13,
            simplify=True,
            dynamic=False,
            device="cpu",
        )
        source_path = Path(str(exported_path)).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"exported onnx not found: {source_path}")

        output_path = Path(request["output_path"]).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            raise FileExistsError(f"target file already exists: {output_path}")
        shutil.copy2(source_path, output_path)
        print(str(output_path))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

