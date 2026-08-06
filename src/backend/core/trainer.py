from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

class TrainingError(RuntimeError):
    pass


class TrainingCancelledError(TrainingError):
    pass


@dataclass
class _TrainingHandle:
    process: subprocess.Popen[str]
    cancel_requested: bool = False


_TRAINING_HANDLES: dict[str, _TrainingHandle] = {}
_TRAINING_HANDLES_LOCK = Lock()


def cancel_training_process(process_key: str) -> bool:
    with _TRAINING_HANDLES_LOCK:
        handle = _TRAINING_HANDLES.get(process_key)
        if handle is None:
            return False
        handle.cancel_requested = True
        process = handle.process
    _terminate_process_tree(process)
    return True


def run_training(
    pretrained_model: str,
    dataset_yaml: str,
    run_dir: str,
    trial_name: str,
    params: dict[str, Any],
    *,
    task_type: str = "detection",
    python_executable: str | None = None,
    src_root: str | None = None,
    process_key: str | None = None,
) -> dict[str, str]:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    run_path = run_path.resolve()
    _prepare_amp_check_weight(run_path)
    stdout_log = run_path / "stdout.log"
    stderr_log = run_path / "stderr.log"
    request_path = run_path / ".train_request.json"
    request_path.write_text(
        json.dumps(
            {
                "pretrained_model": pretrained_model,
                "dataset_yaml": dataset_yaml,
                "run_dir": str(run_path),
                "trial_name": trial_name,
                "params": params,
                "task_type": task_type,
            }
        ),
        encoding="utf-8",
    )

    command = [python_executable or sys.executable, "-m", "backend.core.train_worker", str(request_path.resolve())]
    env = dict(os.environ)
    if src_root:
        env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else f"{src_root}{os.pathsep}{env['PYTHONPATH']}"
    popen_kwargs: dict[str, Any] = {
        "cwd": str(run_path),
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "env": env,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    process: subprocess.Popen[str] | None = None
    cancelled = False
    with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open("w", encoding="utf-8") as stderr_handle:
        try:
            process = subprocess.Popen(command, stdout=stdout_handle, stderr=stderr_handle, **popen_kwargs)
        except FileNotFoundError as exc:
            raise TrainingError("python executable not found while starting training worker") from exc

        if process_key:
            _register_training_process(process_key, process)

        try:
            return_code = process.wait()
        finally:
            if process_key:
                cancelled = _unregister_training_process(process_key)

    if cancelled:
        raise TrainingCancelledError("training cancelled by user")

    if return_code != 0:
        raise TrainingError(_read_training_error(stderr_log) or f"training worker exited with code {return_code}")

    results_csv = run_path / "results.csv"
    if not results_csv.exists():
        raise TrainingError(f"training finished but results.csv not found for {trial_name}")
    return {
        "run_dir": str(run_path.resolve()),
        "stdout_log": str(stdout_log.resolve()),
        "stderr_log": str(stderr_log.resolve()),
    }


def _prepare_amp_check_weight(run_path: Path) -> None:
    source = _package_models_dir() / "yolo26n.pt"
    target = run_path / "yolo26n.pt"
    if not source.exists() or target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _package_models_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models"


def _register_training_process(process_key: str, process: subprocess.Popen[str]) -> None:
    with _TRAINING_HANDLES_LOCK:
        _TRAINING_HANDLES[process_key] = _TrainingHandle(process=process)


def _unregister_training_process(process_key: str) -> bool:
    with _TRAINING_HANDLES_LOCK:
        handle = _TRAINING_HANDLES.pop(process_key, None)
    # A stop request can race with a process that has already exited cleanly.
    # A zero exit code means training completed and must not be relabeled as cancelled.
    return bool(handle and handle.cancel_requested and handle.process.returncode != 0)


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return


def _read_training_error(stderr_log: Path) -> str:
    if not stderr_log.exists():
        return ""
    try:
        lines = stderr_log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    tail = [line.strip() for line in lines[-20:] if line.strip()]
    return tail[-1] if tail else ""
