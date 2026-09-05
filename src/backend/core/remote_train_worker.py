"""Self-contained worker uploaded to a remote YOLO host."""
from __future__ import annotations

import json
import math
import os
import sys
import traceback
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _status(path: Path, value: str, **extra: object) -> None:
    _atomic_json(path, {"status": value, "updated_at": _now(), **extra})


def _identity(pid: int) -> dict:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "starttime": fields[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def _legacy_status(run_dir: Path, status: dict) -> dict:
    """Adopt old managed workers only when their limited evidence is conclusive."""
    unknown = {"state": "unknown", "error": "missing or ambiguous legacy worker evidence"}
    # Never downgrade incomplete new-format evidence into a legacy launch.
    if "identity" in status or "trial_id" in status:
        return unknown
    pid = status.get("pid")
    if type(pid) is not int or pid <= 0:
        return unknown
    request_path = run_dir / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if (not isinstance(request, dict) or "trial_id" in request
            or not isinstance(request.get("run_dir"), str)
            or Path(request["run_dir"]).resolve() != run_dir.resolve()):
        return unknown
    # stat, rather than exists, preserves permission/read failures as ambiguity.
    proc = Path("/proc")
    proc.stat()
    process_dir = proc / str(pid)
    try:
        process_dir.stat()
    except FileNotFoundError:
        if status.get("status") in {"completed", "failed"}:
            return {"state": status["status"], "error": str(status.get("error") or "")}
        return unknown
    if status.get("status") == "running":
        argv = process_dir.joinpath("cmdline").read_bytes().rstrip(b"\x00").split(b"\x00")
        expected = [os.fsencode(str(run_dir.resolve() / "remote_train_worker.py")),
                    os.fsencode(str(request_path.resolve()))]
        if len(argv) >= 3 and argv[-2:] == expected:
            return {"state": "running", "error": ""}
    # An occupied/reused PID cannot establish legacy terminal process identity.
    return unknown


def check_status(run_dir: Path, trial_id: str) -> dict:
    unknown = {"state": "unknown", "error": "missing or ambiguous worker identity/status"}
    try:
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if not isinstance(status, dict):
            return unknown
        try:
            marker = json.loads((run_dir / "started.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            return _legacy_status(run_dir, status)
        import fcntl
        if status.get("trial_id") != trial_id or marker.get("trial_id") != trial_id:
            return unknown
        if status.get("status") not in {"running", "completed", "failed"}:
            return unknown
        identity = status["identity"]
        if (identity != marker["identity"] or type(identity.get("pid")) is not int or identity["pid"] <= 0
                or not isinstance(identity.get("starttime"), str) or not identity["starttime"].isdigit()
                or not isinstance(identity.get("boot_id"), str) or not identity["boot_id"]):
            return unknown
        try:
            alive = _identity(identity["pid"]) == identity
        except FileNotFoundError:
            alive = False
        # Opening an existing lock only: missing evidence must not create new evidence.
        with (run_dir / "worker.lock").open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"state": "running", "error": ""} if alive else unknown
            if alive:
                return {"state": "running", "error": ""}
            if status.get("status") in {"completed", "failed"}:
                return {"state": status["status"], "error": str(status.get("error") or "")}
            if status.get("status") == "running":
                return {"state": "failed", "error": "远程训练进程已退出，未写入完成状态"}
        return unknown
    except Exception:
        return unknown


def _save_per_class_metrics(request: dict, run_dir: Path) -> None:
    """Validate best.pt and persist the same per-class metrics as local training."""
    best_weight = run_dir / "weights" / "best.pt"
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
        _atomic_json(run_dir / "per_class_metrics.json", metrics)
    except Exception as exc:
        print(f"per-class metrics skipped: {exc}", file=sys.stderr)


def _extract_per_class_metrics(result: object, task_type: str) -> list[dict]:
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
    rows = []
    for class_id in sorted(set(names) | set(class_indexes)):
        metric_index = metric_positions.get(class_id)
        map_index = class_id if len(map50_95) > class_id else metric_index
        rows.append({
            "class_id": class_id,
            "class_name": names.get(class_id, f"class_{class_id}"),
            "precision": _metric_value(precision, metric_index),
            "recall": _metric_value(recall, metric_index),
            "map50": _metric_value(map50, metric_index),
            "map50_95": _metric_value(map50_95, map_index),
        })
    return rows


def _class_names(raw_names: object) -> dict[int, str]:
    values = raw_names.items() if isinstance(raw_names, dict) else enumerate(raw_names or [])
    names = {}
    for raw_id, raw_name in values:
        try:
            names[int(raw_id)] = str(raw_name)
        except (TypeError, ValueError):
            continue
    return names


def _class_indexes(value: object) -> list[int]:
    indexes = []
    for item in _to_list(value):
        try:
            indexes.append(int(item))
        except (TypeError, ValueError):
            continue
    return indexes


def _numeric_list(value: object) -> list[float | None]:
    return [_finite_float(item) for item in _to_list(value)]


def _to_list(value: object) -> list:
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


def _finite_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _metric_value(values: list[float | None], index: int | None) -> float | None:
    if index is None or index < 0 or index >= len(values):
        return None
    return values[index]


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--status":
        print(json.dumps(check_status(Path(sys.argv[2]), sys.argv[3])))
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "--metrics":
        request_path = Path(sys.argv[2]).resolve()
        request = json.loads(request_path.read_text(encoding="utf-8"))
        _save_per_class_metrics(request, Path(request["run_dir"]).resolve())
        return 0
    if len(sys.argv) != 2:
        print("usage: remote_train_worker.py request.json", file=sys.stderr)
        return 2
    request_path = Path(sys.argv[1]).resolve()
    run_dir = Path(json.loads(request_path.read_text(encoding="utf-8"))["run_dir"]).resolve()
    status_path = run_dir / "status.json"
    import fcntl
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / "worker.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return 0
    # Persistent marker blocks even completed or crashed launches from being retrained.
    if (run_dir / "started.json").exists() or status_path.exists():
        lock.close()
        return 0
    request = json.loads(request_path.read_text(encoding="utf-8"))
    trial_id = request.get("trial_id")
    if not isinstance(trial_id, str) or not trial_id:
        lock.close()
        return 2
    identity = _identity(os.getpid())
    common = {"trial_id": trial_id, "identity": identity, "pid": os.getpid()}
    _atomic_json(run_dir / "started.json", {**common, "started_at": _now()})
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        run_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(run_dir)
        _status(status_path, "running", started_at=_now(), **common)
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
        train_params.update(project=str(run_dir.parent), name=run_dir.name, exist_ok=True)
        model.train(**train_params)
        _save_per_class_metrics(request, run_dir)
        _status(status_path, "completed", started_at=None, finished_at=_now(), **common)
        return 0
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        _status(status_path, "failed", finished_at=_now(), error=str(exc), **common)
        return 1
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
