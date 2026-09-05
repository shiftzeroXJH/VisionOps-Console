"""Self-contained worker uploaded to a remote YOLO host."""
from __future__ import annotations

import json
import os
import sys
import traceback
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
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


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--status":
        print(json.dumps(check_status(Path(sys.argv[2]), sys.argv[3])))
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
