"""Self-contained worker uploaded to a remote YOLO host."""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _status(path: Path, value: str, **extra: object) -> None:
    payload = {"status": value, "updated_at": _now(), **extra}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: remote_train_worker.py request.json", file=sys.stderr)
        return 2
    request_path = Path(sys.argv[1]).resolve()
    run_dir = Path(json.loads(request_path.read_text(encoding="utf-8"))["run_dir"]).resolve()
    status_path = run_dir / "status.json"
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        run_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(run_dir)
        _status(status_path, "running", started_at=_now(), pid=os.getpid())
        from ultralytics import YOLO

        params = dict(request["params"])
        model = YOLO(request["pretrained_model"])
        train_params = {
            "data": request["dataset_yaml"],
            "device": 0,
            "cache": False,
            "seed": 42,
            "deterministic": True,
            "pretrained": True,
            "plots": True,
            "save": True,
            "save_period": 10,
            "val": True,
            "project": str(run_dir.parent),
            "name": run_dir.name,
            "exist_ok": True,
            "verbose": True,
            "amp": True,
        }
        train_params.update(params)
        model.train(**train_params)
        _status(status_path, "completed", started_at=None, finished_at=_now(), pid=os.getpid())
        return 0
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _status(status_path, "failed", finished_at=_now(), error=str(exc), pid=os.getpid())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
