from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
import subprocess
import shutil
import stat
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

from openclaw_yolo.constants import (
    EXPERIMENT_FILENAME,
    OPENCLAW_MAX_TRAINING_RUNS,
    SEARCH_SPACE,
    STATE_ANALYZING,
    STATE_CANCELLED,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_READY,
    STATE_RETRAINING,
    STATE_TRAINING,
    STATE_WAITING,
    STOP_CONDITIONS,
    SUMMARY_FILENAME,
    TASK_BASELINES,
    TRIAL_CONFIG_FILENAME,
)
from openclaw_yolo.core.analyzer import build_summary
from openclaw_yolo.core.baseline import build_initial_params
from openclaw_yolo.core.constraints import validate_param_value
from openclaw_yolo.core.dataset import analyze_dataset, inspect_dataset
from openclaw_yolo.core.param_search import ProposalValidationError, validate_continue_request
from openclaw_yolo.core.trainer import (
    TrainingCancelledError,
    TrainingError,
    cancel_training_process,
    run_training,
)
from openclaw_yolo.db.repository import Repository, default_project_name
from openclaw_yolo.models import ExperimentConfig, GoalConfig, RemoteServer, TrialRecord
from openclaw_yolo.utils import ensure_dir, read_json, utc_now_iso, write_json


class ServiceError(RuntimeError):
    pass


OPENCLAW_SESSIONS_PATH = "~/.openclaw/agents/main/sessions/sessions.json"
OPENCLAW_NOTIFY_SCRIPT = "/home/shiftzero/bin/openclaw_notify.sh"
STALE_EMPTY_TASK_TTL_HOURS = 2
REMOTE_SOURCE = "remote_sftp"
REMOTE_SYNC_PENDING = "pending"
REMOTE_SYNC_SYNCED = "synced"
REMOTE_SYNC_FAILED = "failed"
REMOTE_TRAINING_RUNNING = "running"
REMOTE_TRAINING_COMPLETED = "completed"
REMOTE_TRAINING_MAYBE_STOPPED = "maybe_stopped"
REMOTE_TRAINING_UNKNOWN = "unknown"
MODEL_FILENAME_ALIASES = {
    "yolov11n.pt": "yolo11n.pt",
    "yolov11n-seg.pt": "yolo11n-seg.pt",
    "yolov11n-obb.pt": "yolo11n-obb.pt",
}
DEFAULT_YOLO_PYTHON = r"D:\apps\miniforge\envs\yolo_env\python.exe"
ILLEGAL_WINDOWS_NAME_CHARS = set('<>:"/\\|?*')


def _parse_scalar_yaml_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return ""
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_args_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        if not key or key.startswith("-"):
            continue
        data[key] = _parse_scalar_yaml_value(value)
    return data


def _model_basename(model: str) -> str:
    raw = str(model or "").strip()
    if not raw:
        return ""
    normalized = raw.replace("\\", "/")
    name = PureWindowsPath(raw).name if "\\" in raw else Path(normalized).name
    return name or raw


def _model_stem(model: str) -> str:
    basename = _model_basename(model)
    stem = Path(basename).stem or basename
    cleaned = []
    for char in stem.lower():
        cleaned.append(char if char.isalnum() else "_")
    value = "".join(cleaned).strip("_")
    return value or "model"


def _validate_export_model_name(model_name: str) -> str:
    normalized = str(model_name or "").strip()
    if not normalized:
        raise ServiceError("model_name is required")
    if normalized[-1] in {".", " "}:
        raise ServiceError("model_name cannot end with a dot or space")
    illegal = sorted({char for char in normalized if char in ILLEGAL_WINDOWS_NAME_CHARS or ord(char) < 32})
    if illegal:
        raise ServiceError(f"model_name contains illegal characters: {' '.join(illegal)}")
    return normalized


def _export_weight_path(run_dir: str) -> Path:
    base = Path(run_dir)
    candidates = [
        base / "weights" / "best.pt",
        base / "weights" / "last.pt",
        base / "best.pt",
        base / "last.pt",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    raise ServiceError(f"no exportable weight file found under run_dir: {run_dir}")


def _trial_weight_path(run_dir: str) -> Path:
    try:
        return _export_weight_path(run_dir)
    except ServiceError as exc:
        raise ServiceError(f"no validation weight file found under run_dir: {run_dir}") from exc


def _export_filename(model_name: str, model: str, task_type: str, imgsz: int) -> str:
    del task_type
    return f"{model_name}-{_model_stem(model)}-{int(imgsz)}.onnx"


def _python_for_yolo() -> str:
    configured = os.environ.get("OPENCLAW_YOLO_PYTHON", "").strip()
    if configured:
        return configured
    if Path(DEFAULT_YOLO_PYTHON).exists():
        return DEFAULT_YOLO_PYTHON
    return sys.executable


def _safe_validation_id() -> str:
    return datetime.now(timezone.utc).strftime("val_%Y%m%d_%H%M%S_%f")


def _validate_preview_filename(filename: str) -> str:
    normalized = str(filename or "").strip()
    if not normalized:
        raise ServiceError("filename is required")
    if "/" in normalized or "\\" in normalized or normalized in {".", ".."}:
        raise ServiceError("invalid filename")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", normalized):
        raise ServiceError("invalid filename")
    return normalized


def _directory_size(path: Path) -> tuple[int, int]:
    total_files = 0
    total_bytes = 0
    if not path.exists():
        return total_files, total_bytes
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        total_files += 1
        try:
            total_bytes += item.stat().st_size
        except OSError:
            continue
    return total_files, total_bytes


def _params_from_args(args: dict[str, Any]) -> dict[str, Any]:
    return {
        key: args[key]
        for key in SEARCH_SPACE
        if key in args and args[key] is not None
    }


def _valid_epoch_count(results_csv: Path) -> int:
    if not results_csv.exists():
        return 0
    import csv

    count = 0
    try:
        with results_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    float(row.get("epoch", ""))
                except (TypeError, ValueError):
                    continue
                count += 1
    except OSError:
        return 0
    return count


def _compact_search_space(search_space: dict[str, Any]) -> dict[str, str]:
    compact: dict[str, str] = {}
    for name, spec in search_space.items():
        if not isinstance(spec, dict):
            compact[name] = str(spec)
            continue
        spec_type = spec.get("type")
        if spec_type == "choice":
            values = ", ".join(str(value) for value in spec.get("values", []))
            compact[name] = f"choice[{values}]"
            continue
        if spec_type in {"int", "float"}:
            parts = [spec_type]
            if "min" in spec and "max" in spec:
                parts.append(f"{spec['min']}..{spec['max']}")
            if "step" in spec:
                parts.append(f"step {spec['step']}")
            compact[name] = " ".join(parts)
            continue
        compact[name] = str(spec)
    return compact


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "metric_context": summary.get("metric_context", {}),
        "final_metrics": summary.get("final_metrics", {}),
        "metric_breakdown": summary.get("metric_breakdown", {}),
        "delta_vs_prev": summary.get("delta_vs_prev", {}),
        "metric_breakdown_delta_vs_prev": summary.get("metric_breakdown_delta_vs_prev", {}),
        "training_dynamics": summary.get("training_dynamics", {}),
        "warnings": summary.get("warnings", []),
        "resource": summary.get("resource", {}),
        "basic_info": summary.get("basic_info", {}),
        "params": summary.get("params", {}),
    }


def _notification_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "basic_info": summary.get("basic_info", {}),
        "metric_context": summary.get("metric_context", {}),
        "final_metrics": summary.get("final_metrics", {}),
        "metric_breakdown": summary.get("metric_breakdown", {}),
        "delta_vs_prev": summary.get("delta_vs_prev", {}),
        "metric_breakdown_delta_vs_prev": summary.get("metric_breakdown_delta_vs_prev", {}),
        "training_dynamics": summary.get("training_dynamics", {}),
        "warnings": summary.get("warnings", []),
        "resource": summary.get("resource", {}),
        "params": summary.get("params", {}),
    }


def _resolve_session_id(session_key: str) -> str:
    script = (
        "import json, sys, pathlib; "
        "path = pathlib.Path(sys.argv[1]).expanduser(); "
        "session_key = sys.argv[2]; "
        "data = json.loads(path.read_text(encoding='utf-8')); "
        "entry = data.get(session_key); "
        "session_id = None if entry is None else entry.get('sessionId'); "
        "sys.exit(2) if entry is None or not session_id else print(session_id)"
    )
    try:
        process = subprocess.run(
            ["wsl", "python3", "-c", script, OPENCLAW_SESSIONS_PATH, session_key],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ServiceError("failed to invoke WSL while validating session_key") from exc

    stdout = process.stdout.strip()
    stderr = process.stderr.strip()
    if process.returncode == 0 and stdout:
        return stdout
    if process.returncode == 2:
        raise ServiceError("invalid session_key: not found in OpenClaw sessions.json")
    if "No such file" in stderr:
        raise ServiceError(f"OpenClaw sessions file not found: {OPENCLAW_SESSIONS_PATH}")
    raise ServiceError(stderr or "failed to resolve session_key via WSL")


def _build_notify_message(
    config: ExperimentConfig,
    trial_name: str,
    compact_summary: dict[str, Any],
) -> str:
    metric_value = compact_summary.get("final_metrics", {}).get(config.goal.metric)
    target_reached = isinstance(metric_value, (int, float)) and float(metric_value) >= config.goal.target
    payload = {
        "experiment_id": config.experiment_id,
        "trial_id": trial_name,
        "goal": {
            "metric": config.goal.metric,
            "target": config.goal.target,
            "value": metric_value,
            "target_reached": target_reached,
        },
        "summary": compact_summary,
    }
    return (
        "这是一次 YOLO 训练完成后的自动回调，结果已随消息提供。默认不要调用工具。\n\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
        "请直接用中文总结关键指标、训练动态、是否达到目标和下一步建议。"
        "只有在结果缺字段、训练失败、用户要求细节，或确需续训前确认最新状态时，才调用 openclaw-yolo 工具。"
        "如果目标已达到，不要调用 continue。"
    )


def _notify_status(latest_event: dict[str, Any] | None) -> dict[str, Any] | None:
    if latest_event is None:
        return None
    payload = latest_event.get("payload", {})
    status = "sent" if latest_event.get("event_type") == "OPENCLAW_NOTIFY_SENT" else "failed"
    result = {
        "status": status,
        "created_at": latest_event.get("created_at"),
        "trial_id": payload.get("trial_id"),
        "session_key": payload.get("session_key"),
    }
    if status == "sent":
        result["session_id"] = payload.get("session_id")
    else:
        result["error"] = payload.get("error")
    return result


def _stale_task_cutoff_iso(max_age_hours: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    return dt.replace(microsecond=0, tzinfo=None).isoformat()


def _notify_openclaw_session(session_id: str, message: str) -> None:
    command = (
        f"{shlex.quote(OPENCLAW_NOTIFY_SCRIPT)} "
        f"{shlex.quote(session_id)} "
        f"{shlex.quote(message)}"
    )
    try:
        process = subprocess.run(
            ["wsl", "bash", "-ic", command],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ServiceError("failed to invoke WSL while notifying OpenClaw") from exc
    if process.returncode != 0:
        raise ServiceError(process.stderr.strip() or "openclaw_notify.sh failed")


def _resolve_pretrained_model(pretrained: str) -> str:
    normalized_pretrained = MODEL_FILENAME_ALIASES.get(pretrained.strip().lower(), pretrained)
    pretrained_path = Path(normalized_pretrained)
    if pretrained_path.is_absolute() and pretrained_path.exists():
        return str(pretrained_path.resolve())

    package_model_path = Path(__file__).resolve().parent / "models" / normalized_pretrained
    if package_model_path.exists():
        return str(package_model_path.resolve())

    if pretrained_path.exists():
        return str(pretrained_path.resolve())

    return normalized_pretrained


def _validate_pretrained_model(pretrained_model: str) -> None:
    model_path = Path(pretrained_model)
    if not model_path.exists():
        return
    if model_path.is_dir():
        raise ServiceError(f"pretrained model path is a directory: {pretrained_model}")
    size = model_path.stat().st_size
    if size < 1024:
        raise ServiceError(
            f"pretrained model file looks invalid or truncated: {pretrained_model} ({size} bytes)"
        )


def _handle_rmtree_error(func: Any, path: str, exc_info: Any) -> None:
    try:
        os_path = Path(path)
        os_path.chmod(stat.S_IWRITE)
        func(path)
    except Exception:
        raise exc_info[1].with_traceback(exc_info[2])


class OrchestratorService:
    def __init__(self, db_path: str | None = None) -> None:
        if db_path == ":memory:":
            repo_path = ":memory:"
        else:
            repo_path = str(Path(db_path or "openclaw_yolo_state.sqlite").resolve())
        self.repo = Repository(repo_path)

    def inspect_dataset(self, dataset_root: str) -> dict[str, Any]:
        return {"yaml_candidates": inspect_dataset(dataset_root)}

    def list_remote_servers(self) -> dict[str, Any]:
        return {
            "remote_servers": [
                {
                    "remote_server_id": server.remote_server_id,
                    "name": server.name,
                    "host": server.host,
                    "port": server.port,
                    "username": server.username,
                    "auth_type": server.auth_type,
                    "private_key_path": server.private_key_path,
                    "password_ref": server.password_ref,
                    "default_runs_root": server.default_runs_root,
                }
                for server in self.repo.list_remote_servers()
            ]
        }

    def create_remote_server(
        self,
        *,
        name: str,
        host: str,
        port: int = 22,
        username: str,
        auth_type: str = "key",
        private_key_path: str | None = None,
        password_ref: str | None = None,
        default_runs_root: str | None = None,
    ) -> dict[str, Any]:
        normalized_auth_type = auth_type.strip().lower()
        if normalized_auth_type not in {"key", "password"}:
            raise ServiceError("auth_type must be 'key' or 'password'")
        if normalized_auth_type == "key" and not (private_key_path or "").strip():
            raise ServiceError("private_key_path is required for key auth")
        if normalized_auth_type == "password" and not (password_ref or "").strip():
            raise ServiceError("password_ref is required for password auth")

        existing = self.repo.list_remote_servers()
        server_id = f"remote_{len(existing) + 1:03d}"
        existing_ids = {server.remote_server_id for server in existing}
        index = len(existing) + 1
        while server_id in existing_ids:
            index += 1
            server_id = f"remote_{index:03d}"
        server = RemoteServer(
            remote_server_id=server_id,
            name=name.strip() or server_id,
            host=host.strip(),
            port=int(port),
            username=username.strip(),
            auth_type=normalized_auth_type,
            private_key_path=(private_key_path or "").strip(),
            password_ref=(password_ref or "").strip(),
            default_runs_root=(default_runs_root or "").strip(),
        )
        if not server.host:
            raise ServiceError("host is required")
        if not server.username:
            raise ServiceError("username is required")
        self.repo.create_remote_server(server)
        return {"remote_server": server.__dict__}

    def test_remote_server(self, remote_server_id: str) -> dict[str, Any]:
        server = self.repo.get_remote_server(remote_server_id)
        client, sftp = self._open_sftp(server)
        try:
            root = server.default_runs_root or "."
            try:
                sftp.stat(root)
                root_exists = True
            except OSError:
                root_exists = False
            return {
                "remote_server_id": remote_server_id,
                "status": "ok",
                "default_runs_root": root,
                "default_runs_root_exists": root_exists,
            }
        finally:
            sftp.close()
            client.close()

    def create_experiment(
        self,
        *,
        description: str,
        project: str | None = None,
        task_type: str,
        dataset_root: str,
        dataset_yaml: str | None,
        pretrained: str,
        save_root: str,
        goal: dict[str, Any],
        initial_params: dict[str, Any] | None = None,
        session_key: str | None = None,
        auto_iterate: bool = False,
        confirm_timeout: int = 60,
        require_session: bool = False,
    ) -> dict[str, Any]:
        self._cleanup_stale_empty_tasks()
        normalized_session_key = (session_key or "").strip()
        if require_session and not normalized_session_key:
            raise ServiceError("session_key is required")
        if normalized_session_key:
            _resolve_session_id(normalized_session_key)

        candidates = inspect_dataset(dataset_root)
        if dataset_yaml is None:
            if len(candidates) != 1:
                return {
                    "status": "needs_dataset_yaml",
                    "yaml_candidates": candidates,
                    "message": "dataset yaml must be specified when zero or multiple candidates are found",
                }
            dataset_yaml = candidates[0]
        elif not Path(dataset_yaml).exists():
            raise ServiceError(f"dataset yaml not found: {dataset_yaml}")

        normalized_description = description.strip()
        normalized_project = (project or "").strip() or default_project_name(normalized_description)
        resolved_pretrained = _resolve_pretrained_model(pretrained)
        _validate_pretrained_model(resolved_pretrained)
        experiment_id = self.repo.next_experiment_id()
        experiment_dir = ensure_dir(Path(save_root) / "experiments" / experiment_id)
        initial_overrides = dict(initial_params or {})
        config = ExperimentConfig(
            experiment_id=experiment_id,
            description=normalized_description,
            project=normalized_project,
            session_key=normalized_session_key,
            task_type=task_type,
            dataset_root=str(Path(dataset_root).resolve()),
            dataset_yaml=str(Path(dataset_yaml).resolve()),
            pretrained_model=resolved_pretrained,
            save_root=str(Path(save_root).resolve()),
            goal=GoalConfig(metric=str(goal["metric"]), target=float(goal["target"])),
            auto_iterate=auto_iterate,
            confirm_timeout=confirm_timeout,
            status=STATE_READY,
            initial_params=build_initial_params(task_type, initial_overrides),
            search_space=SEARCH_SPACE,
            stop_conditions=STOP_CONDITIONS,
        )
        self.repo.create_experiment(config)
        write_json(experiment_dir / EXPERIMENT_FILENAME, config.to_dict())
        self.repo.add_event(experiment_id, "EXPERIMENT_CREATED", config.to_dict())
        return {
            "status": config.status,
            "experiment_id": experiment_id,
            "description": config.description,
            "project": config.project,
            "session_key": config.session_key,
            "dataset_yaml": config.dataset_yaml,
            "initial_params": config.initial_params,
            "experiment_dir": str(experiment_dir),
        }

    def create_task(
        self,
        *,
        description: str,
        session_key: str,
        task_type: str,
        dataset_root: str,
        dataset_yaml: str | None,
        pretrained: str,
        save_root: str,
        goal: dict[str, Any],
        auto_iterate: bool,
        confirm_timeout: int,
        initial_overrides: dict[str, Any],
    ) -> dict[str, Any]:
        return self.create_experiment(
            description=description,
            project=initial_overrides.pop("project", None),
            session_key=session_key,
            task_type=task_type,
            dataset_root=dataset_root,
            dataset_yaml=dataset_yaml,
            pretrained=pretrained,
            save_root=save_root,
            goal=goal,
            auto_iterate=auto_iterate,
            confirm_timeout=confirm_timeout,
            initial_params=initial_overrides,
            require_session=True,
        )

    def list_tasks(self, compact: bool = False) -> dict[str, Any]:
        self._cleanup_stale_empty_tasks()
        experiments = self.repo.list_experiments()
        if compact:
            return {
                "experiments": [
                    {
                        "experiment_id": item.experiment_id,
                        "description": item.description,
                        "project": item.project,
                        "session_key": item.session_key,
                        "status": item.status,
                    }
                    for item in experiments
                ]
            }
        return {
            "experiments": [
                {
                    "experiment_id": item.experiment_id,
                    "description": item.description,
                    "project": item.project,
                    "session_key": item.session_key,
                    "task_type": item.task_type,
                    "status": item.status,
                    "goal": item.goal.__dict__,
                    "dataset_yaml": item.dataset_yaml,
                    "pretrained_model": item.pretrained_model,
                }
                for item in experiments
            ]
        }

    def list_experiments(self) -> dict[str, Any]:
        self._cleanup_stale_empty_tasks()
        experiments = self.repo.list_experiments()
        items: list[dict[str, Any]] = []
        for config in experiments:
            trials = self.repo.list_trials(config.experiment_id)
            latest_trial = trials[-1] if trials else None
            metric = config.goal.metric
            best_trial_info = None
            best_value = None
            for trial in trials:
                value = trial.metrics.get(metric) if trial.metrics else None
                if isinstance(value, (int, float)) and (best_value is None or value > best_value):
                    best_value = float(value)
                    best_trial_info = {
                        "trial_id": trial.trial_id,
                        "iteration": trial.iteration,
                        "metric": metric,
                        "value": best_value,
                    }
            items.append(
                {
                    "experiment_id": config.experiment_id,
                    "description": config.description,
                    "project": config.project,
                    "status": config.status,
                    "task_type": config.task_type,
                    "dataset_root": config.dataset_root,
                    "dataset_yaml": config.dataset_yaml,
                    "pretrained_model": config.pretrained_model,
                    "goal": config.goal.__dict__,
                    "trial_count": len(trials),
                    "best_metric": best_trial_info,
                    "latest_trial": None
                    if latest_trial is None
                    else {
                        "trial_id": latest_trial.trial_id,
                        "iteration": latest_trial.iteration,
                        "status": latest_trial.status,
                        "metrics": latest_trial.metrics,
                        "source": latest_trial.source,
                        "model": _model_basename(latest_trial.model or config.pretrained_model),
                        "remote_training_status": latest_trial.remote_training_status,
                    },
                }
            )
        return {"experiments": items}

    def get_experiment_detail(self, experiment_id: str) -> dict[str, Any]:
        config = self.repo.get_experiment(experiment_id)
        trials = self.repo.list_trials(experiment_id)
        return {
            "experiment": config.to_dict(),
            "trial_count": len(trials),
            "latest_params": self._latest_params(config, trials),
            "default_model": config.pretrained_model,
            "search_space": config.search_space,
            "trials": [self._trial_row(trial) for trial in trials],
        }

    def update_experiment(
        self,
        experiment_id: str,
        *,
        description: str | None = None,
        project: str | None = None,
    ) -> dict[str, Any]:
        config = self.repo.get_experiment(experiment_id)
        if description is None and project is None:
            return {"experiment": config.to_dict()}

        normalized_description = None
        if description is not None:
            normalized_description = description.strip()
            if not normalized_description:
                raise ServiceError("description cannot be empty")

        normalized_project = None
        if project is not None:
            normalized_project = project.strip() or default_project_name(normalized_description or config.description)

        self.repo.update_experiment(
            experiment_id,
            description=normalized_description,
            project=normalized_project,
        )
        if normalized_description is not None:
            config.description = normalized_description
        if normalized_project is not None:
            config.project = normalized_project

        experiment_json = Path(config.save_root) / "experiments" / experiment_id / EXPERIMENT_FILENAME
        if experiment_json.exists():
            write_json(experiment_json, config.to_dict())

        self.repo.add_event(
            experiment_id,
            "EXPERIMENT_UPDATED",
            {"description": config.description, "project": config.project},
        )
        return {"experiment": config.to_dict()}

    def show_task(self, experiment_id: str, compact: bool = False) -> dict[str, Any]:
        config = self.repo.get_experiment(experiment_id)
        trials = self.repo.list_trials(experiment_id)
        notify_status = _notify_status(
            self.repo.latest_event_for_types(
                experiment_id,
                ["OPENCLAW_NOTIFY_SENT", "OPENCLAW_NOTIFY_FAILED"],
            )
        )
        if compact:
            latest_trial = trials[-1] if trials else None
            return {
                "experiment_id": config.experiment_id,
                "description": config.description,
                "session_key": config.session_key,
                "status": config.status,
                "goal": config.goal.__dict__,
                "trial_count": len(trials),
                "openclaw_notify": notify_status,
                "latest_trial": None
                if latest_trial is None
                else {
                    "trial_id": latest_trial.trial_id,
                    "display_name": latest_trial.display_name,
                    "iteration": latest_trial.iteration,
                    "status": latest_trial.status,
                    "metrics": latest_trial.metrics,
                },
            }
        return {
            "experiment": config.to_dict(),
            "openclaw_notify": notify_status,
            "trials": [
                {
                    "trial_id": trial.trial_id,
                    "iteration": trial.iteration,
                    "status": trial.status,
                    "params": trial.params,
                    "metrics": trial.metrics,
                    "summary_path": trial.summary_path,
                    "model": trial.model,
                }
                for trial in trials
            ],
        }

    def get_param_metadata(self, experiment_id: str) -> dict[str, Any]:
        config = self.repo.get_experiment(experiment_id)
        trials = self.repo.list_trials(experiment_id)
        merged_search_space = dict(SEARCH_SPACE)
        merged_search_space.update(config.search_space or {})
        return {
            "experiment_id": experiment_id,
            "task_type": config.task_type,
            "baseline": TASK_BASELINES.get(config.task_type, {}),
            "initial_params": config.initial_params,
            "latest_params": self._latest_params(config, trials),
            "default_model": config.pretrained_model,
            "editable_schema": self._editable_schema(merged_search_space),
            "search_space": merged_search_space,
        }

    def validate_params(
        self,
        experiment_id: str,
        *,
        params: dict[str, Any] | None = None,
        param_updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = self.repo.get_experiment(experiment_id)
        trials = self.repo.list_trials(experiment_id)
        base = self._latest_params(config, trials)
        candidate = dict(params) if params is not None else dict(base)
        if param_updates:
            candidate.update(param_updates)

        normalized: dict[str, Any] = {}
        errors: dict[str, str] = {}
        warnings: list[str] = []
        for key in candidate:
            if key not in SEARCH_SPACE:
                errors[key] = "unsupported parameter"
        for key in SEARCH_SPACE:
            if key not in candidate:
                errors[key] = "missing required parameter"
                continue
            try:
                normalized[key] = validate_param_value(key, candidate[key])
            except ValueError as exc:
                errors[key] = str(exc)
        if int(normalized.get("workers", 1) or 0) == 0:
            warnings.append("workers is 0; data loading may be slow")
        if errors:
            return {
                "valid": False,
                "normalized_params": normalized,
                "errors": errors,
                "warnings": warnings,
            }
        return {
            "valid": True,
            "normalized_params": normalized,
            "errors": {},
            "warnings": warnings,
        }

    def cancel_task(self, experiment_id: str, reason: str | None = None) -> dict[str, Any]:
        config = self.repo.get_experiment(experiment_id)
        if config.status in {STATE_COMPLETED, STATE_CANCELLED}:
            return {
                "experiment_id": experiment_id,
                "status": config.status,
                "message": "task already finalized",
            }
        normalized_reason = reason or "cancelled by user"
        process_terminated = cancel_training_process(experiment_id)
        for trial in self.repo.list_trials(experiment_id):
            if trial.status in {STATE_TRAINING, STATE_RETRAINING, STATE_ANALYZING, STATE_WAITING}:
                self.repo.update_trial(trial.trial_id, status=STATE_CANCELLED)
        self.repo.update_experiment_status(experiment_id, STATE_CANCELLED)
        self.repo.add_event(
            experiment_id,
            "TASK_CANCELLED",
            {"reason": normalized_reason, "process_terminated": process_terminated},
        )
        return {
            "experiment_id": experiment_id,
            "status": STATE_CANCELLED,
            "reason": normalized_reason,
            "process_terminated": process_terminated,
        }

    def delete_task(
        self,
        experiment_id: str,
        *,
        keep_files: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        config = self.repo.get_experiment(experiment_id)
        finalized_states = {STATE_COMPLETED, STATE_CANCELLED, STATE_FAILED}
        if not force and config.status not in finalized_states:
            raise ServiceError(
                f"task {experiment_id} is in status {config.status}; cancel or finalize it first, or use --force"
            )

        trials = self.repo.list_trials(experiment_id)
        experiment_dir = Path(config.save_root).resolve() / "experiments" / experiment_id
        files_deleted = False
        warnings: list[str] = []

        deleted_trials = self.repo.delete_trials_for_experiment(experiment_id)
        deleted_events = self.repo.delete_events_for_experiment(experiment_id)
        self.repo.delete_experiment(experiment_id)

        if not keep_files and experiment_dir.exists():
            save_root = Path(config.save_root).resolve()
            experiments_root = (save_root / "experiments").resolve()
            resolved_experiment_dir = experiment_dir.resolve()
            if experiments_root not in resolved_experiment_dir.parents:
                raise ServiceError(f"refusing to delete path outside experiments root: {resolved_experiment_dir}")
            try:
                shutil.rmtree(resolved_experiment_dir, onerror=_handle_rmtree_error)
                files_deleted = True
            except Exception as exc:
                warnings.append(f"failed to delete files: {exc}")

        return {
            "experiment_id": experiment_id,
            "deleted": True,
            "deleted_trials": deleted_trials,
            "deleted_events": deleted_events,
            "files_deleted": files_deleted,
            "kept_files": keep_files,
            "previous_status": config.status,
            "trial_ids": [trial.trial_id for trial in trials],
            "warnings": warnings,
        }

    def delete_trial(
        self,
        trial_id: str,
        *,
        keep_files: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        trial = self.repo.get_trial(trial_id)
        config = self.repo.get_experiment(trial.experiment_id)
        protected_states = {STATE_TRAINING, STATE_RETRAINING, STATE_ANALYZING}
        if not force and trial.status in protected_states:
            raise ServiceError(
                f"trial {trial_id} is in status {trial.status}; wait until it finishes or use --force"
            )

        warnings: list[str] = []
        deleted_paths: list[str] = []
        experiment_dir = (Path(config.save_root).resolve() / "experiments" / trial.experiment_id).resolve()

        if not keep_files:
            candidate_dirs: list[Path] = []
            run_dir = Path(trial.run_dir).resolve()
            if experiment_dir in run_dir.parents or run_dir == experiment_dir:
                candidate_dirs.append(run_dir)
            if trial.summary_path:
                summary_dir = Path(trial.summary_path).resolve().parent
                if summary_dir not in candidate_dirs and experiment_dir in summary_dir.parents:
                    candidate_dirs.append(summary_dir)

            for candidate in candidate_dirs:
                if experiment_dir not in candidate.parents:
                    warnings.append(f"skipped path outside experiment directory: {candidate}")
                    continue
                if candidate.exists():
                    try:
                        shutil.rmtree(candidate, onerror=_handle_rmtree_error)
                        deleted_paths.append(str(candidate))
                    except Exception as exc:
                        warnings.append(f"failed to delete files for {candidate}: {exc}")

        deleted_events = self.repo.delete_events_for_trial(trial_id)
        deleted_trials = self.repo.delete_trial(trial_id)
        remaining_trials = self.repo.list_trials(trial.experiment_id)
        if remaining_trials:
            self.repo.update_experiment_status(trial.experiment_id, remaining_trials[-1].status)
        else:
            self.repo.update_experiment_status(trial.experiment_id, STATE_READY)

        return {
            "experiment_id": trial.experiment_id,
            "trial_id": trial_id,
            "deleted": deleted_trials == 1,
            "deleted_trials": deleted_trials,
            "deleted_events": deleted_events,
            "files_deleted": bool(deleted_paths),
            "deleted_paths": deleted_paths,
            "kept_files": keep_files,
            "previous_status": trial.status,
            "remaining_trial_count": len(remaining_trials),
            "warnings": warnings,
        }

    def clear_validation_preview_cache(self) -> dict[str, Any]:
        experiments = self.repo.list_experiments()
        deleted_dirs = 0
        deleted_files = 0
        deleted_bytes = 0
        warnings: list[str] = []
        for config in experiments:
            experiment_dir = (Path(config.save_root).resolve() / "experiments" / config.experiment_id).resolve()
            for trial in self.repo.list_trials(config.experiment_id):
                run_dir = Path(trial.run_dir).resolve()
                preview_dir = (run_dir / ".validation_previews").resolve()
                if not preview_dir.exists():
                    continue
                if experiment_dir not in preview_dir.parents and run_dir not in preview_dir.parents:
                    warnings.append(f"skipped unsafe validation cache path: {preview_dir}")
                    continue
                files, size = _directory_size(preview_dir)
                try:
                    shutil.rmtree(preview_dir, onerror=_handle_rmtree_error)
                    deleted_dirs += 1
                    deleted_files += files
                    deleted_bytes += size
                except Exception as exc:
                    warnings.append(f"failed to delete {preview_dir}: {exc}")
        return {
            "status": "cleared",
            "deleted_dirs": deleted_dirs,
            "deleted_files": deleted_files,
            "deleted_bytes": deleted_bytes,
            "warnings": warnings,
        }

    def run_trial(
        self,
        experiment_id: str,
        params: dict[str, Any] | None = None,
        *,
        pretrained: str | None = None,
        note: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        config = self.repo.get_experiment(experiment_id)
        trials = self.repo.list_trials(experiment_id)
        self._ensure_openclaw_training_limit(config, trials)
        iteration = self._next_iteration(trials)
        validation = self.validate_params(experiment_id, params=params or config.initial_params)
        if not validation["valid"]:
            raise ServiceError(f"invalid trial params: {validation['errors']}")
        trial_params = validation["normalized_params"]
        trial_model = _resolve_pretrained_model(pretrained or config.pretrained_model)
        _validate_pretrained_model(trial_model)
        display_name = self._next_trial_display_name(experiment_id, trial_model, trial_params)
        trial_id = self.repo.next_trial_id()
        trial_dir = ensure_dir(Path(config.save_root) / "experiments" / experiment_id / display_name)
        status = STATE_TRAINING if iteration == 1 else STATE_RETRAINING
        dataset_analysis = analyze_dataset(config.dataset_yaml)
        trial = TrialRecord(
            trial_id=trial_id,
            display_name=display_name,
            experiment_id=experiment_id,
            iteration=iteration,
            params=trial_params,
            status=status,
            run_dir=str(trial_dir),
            source="trained",
            note=(note or "").strip(),
            reason=(reason or "").strip(),
            model=trial_model,
            model_source="manual" if pretrained else "experiment_default",
            params_source="manual",
            dataset_analysis=dataset_analysis,
        )
        write_json(trial_dir / TRIAL_CONFIG_FILENAME, trial_params)
        self.repo.create_trial(trial)
        self.repo.update_experiment_status(experiment_id, status)
        self.repo.add_event(
            experiment_id,
            "TRIAL_STARTED",
            {
                "trial_id": trial_id,
                "display_name": display_name,
                "params": trial_params,
                "model": trial_model,
                "note": trial.note,
                "reason": trial.reason,
            },
            trial_id,
        )

        try:
            training_result = run_training(
                pretrained_model=trial_model,
                dataset_yaml=config.dataset_yaml,
                run_dir=str(trial_dir),
                trial_name=display_name,
                params=trial_params,
                process_key=experiment_id,
            )
            if self.repo.get_experiment(experiment_id).status == STATE_CANCELLED:
                self.repo.update_trial(trial_id, status=STATE_CANCELLED)
                self.repo.add_event(
                    experiment_id,
                    "TRIAL_CANCELLED",
                    {"trial_id": trial_id, "reason": "cancelled by user"},
                    trial_id,
                )
                return {
                    "status": STATE_CANCELLED,
                    "trial_id": trial_id,
                    "display_name": display_name,
                    "run_dir": training_result["run_dir"],
                    "stdout_log": training_result["stdout_log"],
                    "stderr_log": training_result["stderr_log"],
                }
            run_dir = training_result["run_dir"]
            previous_summary = None
            summaries = self.repo.recent_summaries(experiment_id, limit=1)
            if summaries:
                previous_summary = summaries[-1]
            self.repo.update_experiment_status(experiment_id, STATE_ANALYZING)
            summary = build_summary(
                trial_id,
                config.task_type,
                run_dir,
                trial_params,
                previous_summary,
            ).to_dict()
            summary_path = trial_dir / SUMMARY_FILENAME
            write_json(summary_path, summary)
            next_status = self._completion_status(config, summary, len(trials) + 1)
            self.repo.update_trial(
                trial_id,
                status=next_status,
                metrics=summary["final_metrics"],
                summary_path=str(summary_path),
            )
            self.repo.update_experiment_status(experiment_id, next_status)
            self.repo.add_event(experiment_id, "TRIAL_COMPLETED", summary, trial_id)
            self._notify_training_result(config, trial_id, summary)
            return {
                "status": next_status,
                "trial_id": trial_id,
                "display_name": display_name,
                "run_dir": run_dir,
                "stdout_log": training_result["stdout_log"],
                "stderr_log": training_result["stderr_log"],
                "summary_path": str(summary_path),
                "final_metrics": summary["final_metrics"],
            }
        except TrainingCancelledError as exc:
            self.repo.update_trial(trial_id, status=STATE_CANCELLED)
            self.repo.update_experiment_status(experiment_id, STATE_CANCELLED)
            self.repo.add_event(
                experiment_id,
                "TRIAL_CANCELLED",
                {"trial_id": trial_id, "error": str(exc)},
                trial_id,
            )
            return {
                "status": STATE_CANCELLED,
                "trial_id": trial_id,
                "display_name": display_name,
                "run_dir": str(trial_dir),
            }
        except TrainingError as exc:
            self.repo.update_trial(trial_id, status=STATE_FAILED)
            self.repo.update_experiment_status(experiment_id, STATE_FAILED)
            self.repo.add_event(experiment_id, "TRIAL_FAILED", {"trial_id": trial_id, "error": str(exc)}, trial_id)
            raise ServiceError(str(exc)) from exc

    def get_summary(self, trial_id: str, compact: bool = False) -> dict[str, Any]:
        trial = self.repo.get_trial(trial_id)
        config = self.repo.get_experiment(trial.experiment_id)
        if not trial.summary_path:
            summary: dict[str, Any] = {
                "trial_id": trial_id,
                "final_metrics": {},
                "metric_breakdown": {},
                "delta_vs_prev": {},
                "metric_breakdown_delta_vs_prev": {},
                "training_dynamics": {},
                "warnings": ["summary_not_available"],
                "resource": {},
                "params": trial.params,
            }
        else:
            summary = read_json(trial.summary_path)
        logs = self._trial_logs(trial.run_dir)
        if compact:
            return {
                "trial_id": trial_id,
                "display_name": trial.display_name,
                "run_dir": trial.run_dir,
                "summary_path": trial.summary_path,
                "source": trial.source,
                "note": trial.note,
                "reason": trial.reason,
                "model": trial.model,
                "model_display": _model_basename(trial.model),
                "model_source": trial.model_source,
                "params_source": trial.params_source,
                "remote_server_id": trial.remote_server_id,
                "remote_run_dir": trial.remote_run_dir,
                "sync_status": trial.sync_status,
                "sync_error": trial.sync_error,
                "remote_training_status": trial.remote_training_status,
                "last_synced_at": trial.last_synced_at,
                "last_synced_epoch_count": trial.last_synced_epoch_count,
                "logs": logs,
                "metric_context": summary.get("metric_context", {}),
                "final_metrics": summary.get("final_metrics", {}),
                "metric_breakdown": summary.get("metric_breakdown", {}),
                "delta_vs_prev": summary.get("delta_vs_prev", {}),
                "metric_breakdown_delta_vs_prev": summary.get("metric_breakdown_delta_vs_prev", {}),
                "training_dynamics": summary.get("training_dynamics", {}),
                "warnings": summary.get("warnings", []),
                "dataset_analysis": trial.dataset_analysis,
            }
        summary["trial"] = {
            "trial_id": trial.trial_id,
            "display_name": trial.display_name,
            "experiment_id": trial.experiment_id,
            "task_type": config.task_type,
            "iteration": trial.iteration,
            "status": trial.status,
            "run_dir": trial.run_dir,
            "summary_path": trial.summary_path,
            "source": trial.source,
            "note": trial.note,
            "reason": trial.reason,
            "model": trial.model,
            "model_display": _model_basename(trial.model),
            "model_source": trial.model_source,
            "params_source": trial.params_source,
            "remote_server_id": trial.remote_server_id,
            "remote_run_dir": trial.remote_run_dir,
            "sync_status": trial.sync_status,
            "sync_error": trial.sync_error,
            "remote_training_status": trial.remote_training_status,
            "last_synced_at": trial.last_synced_at,
            "last_synced_epoch_count": trial.last_synced_epoch_count,
            "logs": logs,
            "imgsz": int(summary.get("params", {}).get("imgsz") or trial.params.get("imgsz") or 0),
        }
        summary["dataset_analysis"] = trial.dataset_analysis
        return summary

    def rename_trial(self, trial_id: str, display_name: str) -> dict[str, Any]:
        trial = self.repo.get_trial(trial_id)
        normalized = self._validate_trial_display_name(display_name)
        if normalized == trial.display_name:
            return {
                "trial_id": trial.trial_id,
                "display_name": trial.display_name,
                "experiment_id": trial.experiment_id,
            }
        if self.repo.trial_display_name_exists(trial.experiment_id, normalized, exclude_trial_id=trial_id):
            raise ServiceError(f"trial name already exists in experiment: {normalized}")
        self.repo.update_trial(trial_id, display_name=normalized)
        self.repo.add_event(
            trial.experiment_id,
            "TRIAL_RENAMED",
            {
                "trial_id": trial_id,
                "display_name": normalized,
                "previous_display_name": trial.display_name,
            },
            trial_id,
        )
        return {
            "trial_id": trial.trial_id,
            "display_name": normalized,
            "experiment_id": trial.experiment_id,
        }

    def export_trial_onnx(
        self,
        trial_id: str,
        *,
        model_name: str,
        output_dir: str,
    ) -> dict[str, Any]:
        trial = self.repo.get_trial(trial_id)
        config = self.repo.get_experiment(trial.experiment_id)
        normalized_model_name = _validate_export_model_name(model_name)
        raw_output_dir = str(output_dir or "").strip()
        if not raw_output_dir:
            raise ServiceError("output_dir is required")
        normalized_output_dir = str(Path(raw_output_dir).expanduser().resolve())

        imgsz = int(trial.params.get("imgsz") or 0)
        if imgsz <= 0:
            raise ServiceError(f"invalid imgsz for trial {trial_id}: {trial.params.get('imgsz')}")

        weight_path = _export_weight_path(trial.run_dir)
        output_filename = _export_filename(normalized_model_name, trial.model, config.task_type, imgsz)
        output_path = Path(normalized_output_dir) / output_filename
        if output_path.exists():
            raise ServiceError(f"target file already exists: {output_path}")

        request_path = Path(trial.run_dir) / ".export_onnx_request.json"
        write_json(
            request_path,
            {
                "model_path": str(weight_path),
                "imgsz": imgsz,
                "output_path": str(output_path),
            },
        )

        python_executable = _python_for_yolo()
        env = dict(os.environ)
        src_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else f"{src_root}{os.pathsep}{env['PYTHONPATH']}"
        process = subprocess.run(
            [python_executable, "-m", "openclaw_yolo.core.export_worker", str(request_path)],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
            env=env,
        )
        if process.returncode != 0:
            stderr = (process.stderr or "").strip()
            stdout = (process.stdout or "").strip()
            raise ServiceError(stderr or stdout or "failed to export onnx")

        self.repo.add_event(
            trial.experiment_id,
            "TRIAL_ONNX_EXPORTED",
            {
                "trial_id": trial_id,
                "model_name": normalized_model_name,
                "weight_path": str(weight_path),
                "output_path": str(output_path),
                "python_executable": python_executable,
            },
            trial_id,
        )
        return {
            "trial_id": trial_id,
            "status": "exported",
            "output_path": str(output_path),
            "output_filename": output_filename,
            "weight_path": str(weight_path),
            "python_executable": python_executable,
            "imgsz": imgsz,
            "task_type": config.task_type,
        }

    def validate_trial_preview(
        self,
        trial_id: str,
        *,
        image_limit: int = 50,
        conf: float = 0.25,
    ) -> dict[str, Any]:
        trial = self.repo.get_trial(trial_id)
        config = self.repo.get_experiment(trial.experiment_id)
        run_dir = Path(trial.run_dir)
        if not run_dir.exists():
            raise ServiceError(f"trial run_dir not found: {trial.run_dir}")
        weight_path = _trial_weight_path(trial.run_dir)
        normalized_limit = int(image_limit)
        if normalized_limit < 1 or normalized_limit > 500:
            raise ServiceError("image_limit must be between 1 and 500")
        normalized_conf = float(conf)
        if normalized_conf < 0.001 or normalized_conf > 1.0:
            raise ServiceError("conf must be between 0.001 and 1.0")

        validation_id = _safe_validation_id()
        output_dir = ensure_dir(run_dir / ".validation_previews" / validation_id)
        request_path = output_dir / ".validation_request.json"
        stdout_log = output_dir / "stdout.log"
        stderr_log = output_dir / "stderr.log"
        imgsz = int(trial.params.get("imgsz", config.initial_params.get("imgsz", 640)))
        batch = int(trial.params.get("batch", config.initial_params.get("batch", 16)))
        write_json(
            request_path,
            {
                "trial_id": trial_id,
                "validation_id": validation_id,
                "task_type": config.task_type,
                "dataset_yaml": config.dataset_yaml,
                "model_path": str(weight_path),
                "output_dir": str(output_dir),
                "image_limit": normalized_limit,
                "conf": normalized_conf,
                "imgsz": imgsz,
                "batch": batch,
            },
        )

        python_executable = _python_for_yolo()
        env = dict(os.environ)
        src_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else f"{src_root}{os.pathsep}{env['PYTHONPATH']}"
        completed = subprocess.run(
            [python_executable, "-m", "openclaw_yolo.core.validation_worker", str(request_path)],
            capture_output=True,
            text=True,
            cwd=str(run_dir),
            check=False,
            env=env,
        )
        stdout_log.write_text(completed.stdout or "", encoding="utf-8")
        stderr_log.write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "failed to run validation preview").strip()
            raise ServiceError(message.splitlines()[-1] if message else "failed to run validation preview")
        try:
            worker_result = json.loads((completed.stdout or "").strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise ServiceError("validation worker did not return valid JSON") from exc

        return {
            "trial_id": trial_id,
            "display_name": trial.display_name,
            "validation_id": validation_id,
            "split": "val",
            "conf": normalized_conf,
            "image_limit": normalized_limit,
            "imgsz": imgsz,
            "batch": batch,
            "weight_path": str(weight_path),
            "metrics": worker_result.get("metrics", {}),
            "images": worker_result.get("images", []),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
        }

    def continue_experiment(
        self,
        experiment_id: str,
        *,
        param_updates: dict[str, Any] | None,
        reason: str | None,
    ) -> dict[str, Any]:
        trials = self.repo.list_trials(experiment_id)
        if not trials:
            raise ServiceError("cannot continue an experiment without trials")
        config = self.repo.get_experiment(experiment_id)
        latest_trial = trials[-1]
        if not latest_trial.summary_path:
            raise ServiceError("latest trial has no summary")
        latest_summary = read_json(latest_trial.summary_path)
        target_reached = float(latest_summary["final_metrics"].get(config.goal.metric, 0.0)) >= config.goal.target
        try:
            validated = validate_continue_request(
                param_updates,
                reason,
                target_reached=target_reached,
            )
        except ProposalValidationError as exc:
            raise ServiceError(str(exc)) from exc
        params = dict(latest_trial.params)
        params.update(validated["param_updates"])
        self.repo.add_event(
            experiment_id,
            "CONTINUE_REQUESTED",
            {
                "trial_id": latest_trial.trial_id,
                "param_updates": validated["param_updates"],
                "reason": validated["reason"],
            },
            latest_trial.trial_id,
        )
        result = self.run_trial(experiment_id, params=params, reason=validated["reason"])
        result["applied_updates"] = validated["param_updates"]
        result["reason"] = validated["reason"]
        return result

    def import_run(
        self,
        experiment_id: str,
        *,
        run_dir: str,
        params: dict[str, Any] | None = None,
        pretrained: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        config = self.repo.get_experiment(experiment_id)
        run_path = Path(run_dir)
        if not run_path.exists():
            raise ServiceError(f"run_dir not found: {run_dir}")
        if not (run_path / "results.csv").exists():
            raise ServiceError(f"results.csv not found in run_dir: {run_dir}")

        config_path = run_path / TRIAL_CONFIG_FILENAME
        args_path = run_path / "args.yaml"
        args_data: dict[str, Any] = {}
        if args_path.exists():
            args_data = _parse_args_yaml(args_path.read_text(encoding="utf-8"))
        raw_params = params
        params_source = "manual" if params is not None else "latest"
        if raw_params is None and args_data:
            raw_params = _params_from_args(args_data)
            params_source = "args_yaml"
        elif raw_params is None and config_path.exists():
            raw_params = read_json(config_path)
            params_source = "config_json"
        if raw_params:
            base_params = self._latest_params(config, self.repo.list_trials(experiment_id))
            merged_params = dict(base_params)
            merged_params.update(raw_params)
            raw_params = merged_params
            if params_source == "args_yaml" and set(raw_params) != set(_params_from_args(args_data)):
                params_source = "args_yaml_partial"
        validation = self.validate_params(experiment_id, params=raw_params or self._latest_params(config, self.repo.list_trials(experiment_id)))
        if not validation["valid"]:
            raise ServiceError(f"invalid imported params: {validation['errors']}")
        trial_params = validation["normalized_params"]
        model_source = "manual"
        trial_model = pretrained or ""
        if not trial_model and args_data.get("model"):
            trial_model = str(args_data["model"])
            model_source = "args_yaml"
        if not trial_model:
            trial_model = config.pretrained_model
            model_source = "experiment_default"

        trials = self.repo.list_trials(experiment_id)
        iteration = self._next_iteration(trials)
        display_name = self._next_trial_display_name(experiment_id, trial_model, trial_params)
        trial_id = self.repo.next_trial_id()
        trial_dir = ensure_dir(Path(config.save_root) / "experiments" / experiment_id / display_name)
        dataset_analysis = analyze_dataset(config.dataset_yaml)
        previous_summary = None
        summaries = self.repo.recent_summaries(experiment_id, limit=1)
        if summaries:
            previous_summary = summaries[-1]
        summary = build_summary(
            trial_id,
            config.task_type,
            str(run_path),
            trial_params,
            previous_summary,
        ).to_dict()
        summary_path = trial_dir / SUMMARY_FILENAME
        write_json(trial_dir / TRIAL_CONFIG_FILENAME, trial_params)
        write_json(summary_path, summary)
        next_status = self._completion_status(config, summary, len(self.repo.list_trials(experiment_id)) + 1)
        trial = TrialRecord(
            trial_id=trial_id,
            display_name=display_name,
            experiment_id=experiment_id,
            iteration=iteration,
            params=trial_params,
            status=next_status,
            run_dir=str(run_path.resolve()),
            summary_path=str(summary_path),
            metrics=summary["final_metrics"],
            source="imported",
            note=(note or "").strip(),
            model=trial_model,
            model_source=model_source,
            params_source=params_source,
            dataset_analysis=dataset_analysis,
        )
        self.repo.create_trial(trial)
        self.repo.update_experiment_status(experiment_id, next_status)
        self.repo.add_event(
            experiment_id,
            "TRIAL_IMPORTED",
            {
                "trial_id": trial_id,
                "display_name": display_name,
                "run_dir": trial.run_dir,
                "summary_path": str(summary_path),
                "note": trial.note,
            },
            trial_id,
        )
        return {
            "status": next_status,
            "trial_id": trial_id,
            "display_name": display_name,
            "run_dir": trial.run_dir,
            "summary_path": str(summary_path),
            "final_metrics": summary["final_metrics"],
        }

    def register_remote_trial(
        self,
        experiment_id: str,
        *,
        remote_server_id: str,
        remote_run_dir: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        config = self.repo.get_experiment(experiment_id)
        server = self.repo.get_remote_server(remote_server_id)
        args_text = self._read_remote_text(server, self._remote_join(remote_run_dir, "args.yaml"))
        args_data = _parse_args_yaml(args_text)
        if not args_data:
            raise ServiceError("args.yaml is empty or unsupported")
        if not args_data.get("model"):
            raise ServiceError("args.yaml does not contain model")

        base_params = self._latest_params(config, self.repo.list_trials(experiment_id))
        parsed_params = _params_from_args(args_data)
        merged_params = dict(base_params)
        merged_params.update(parsed_params)
        validation = self.validate_params(experiment_id, params=merged_params)
        if not validation["valid"]:
            raise ServiceError(f"invalid remote args params: {validation['errors']}")
        trial_params = validation["normalized_params"]
        params_source = "remote_args_yaml" if set(parsed_params) >= set(SEARCH_SPACE) else "remote_args_yaml_partial"
        trial_model = str(args_data["model"])
        trials = self.repo.list_trials(experiment_id)
        display_name = self._next_trial_display_name(experiment_id, trial_model, trial_params)
        trial_id = self.repo.next_trial_id()
        iteration = self._next_iteration(trials)
        cache_dir = ensure_dir(Path(config.save_root) / "experiments" / experiment_id / display_name)
        dataset_analysis = analyze_dataset(config.dataset_yaml)
        write_json(cache_dir / TRIAL_CONFIG_FILENAME, trial_params)
        trial = TrialRecord(
            trial_id=trial_id,
            display_name=display_name,
            experiment_id=experiment_id,
            iteration=iteration,
            params=trial_params,
            status=STATE_TRAINING,
            run_dir=str(cache_dir),
            source=REMOTE_SOURCE,
            note=(note or "").strip(),
            model=trial_model,
            model_source="remote_args_yaml",
            params_source=params_source,
            remote_server_id=remote_server_id,
            remote_run_dir=remote_run_dir,
            sync_status=REMOTE_SYNC_PENDING,
            remote_training_status=REMOTE_TRAINING_UNKNOWN,
            dataset_analysis=dataset_analysis,
        )
        self.repo.create_trial(trial)
        self.repo.update_experiment_status(experiment_id, STATE_TRAINING)
        self.repo.add_event(
            experiment_id,
            "REMOTE_TRIAL_REGISTERED",
            {
                "trial_id": trial_id,
                "display_name": display_name,
                "remote_server_id": remote_server_id,
                "remote_run_dir": remote_run_dir,
                "model": trial_model,
                "params_source": params_source,
            },
            trial_id,
        )
        return {
            "status": trial.status,
            "trial_id": trial_id,
            "display_name": display_name,
            "remote_server_id": remote_server_id,
            "remote_run_dir": remote_run_dir,
            "local_run_dir": str(cache_dir),
            "model": _model_basename(trial_model),
            "params": trial_params,
            "params_source": params_source,
        }

    def import_remote_run(
        self,
        experiment_id: str,
        *,
        remote_server_id: str,
        remote_run_dir: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        registered = self.register_remote_trial(
            experiment_id,
            remote_server_id=remote_server_id,
            remote_run_dir=remote_run_dir,
            note=note,
        )
        try:
            synced = self.sync_remote_trial(registered["trial_id"])
            synced["registered"] = registered
            return synced
        except ServiceError as exc:
            return {
                "status": STATE_TRAINING,
                "trial_id": registered["trial_id"],
                "sync_status": REMOTE_SYNC_FAILED,
                "sync_error": str(exc),
                "remote_training_status": REMOTE_TRAINING_UNKNOWN,
                "registered": registered,
            }

    def sync_remote_trial(self, trial_id: str) -> dict[str, Any]:
        trial = self.repo.get_trial(trial_id)
        if trial.source != REMOTE_SOURCE:
            raise ServiceError(f"trial is not a remote SFTP trial: {trial_id}")
        config = self.repo.get_experiment(trial.experiment_id)
        server = self.repo.get_remote_server(trial.remote_server_id)
        cache_dir = ensure_dir(trial.run_dir)
        client, sftp = self._open_sftp(server)
        sync_error = ""
        try:
            self._download_remote_file(sftp, self._remote_join(trial.remote_run_dir, "args.yaml"), cache_dir / "args.yaml")
            remote_csv = self._remote_join(trial.remote_run_dir, "results.csv")
            csv_stat = sftp.stat(remote_csv)
            self._download_remote_file(sftp, remote_csv, cache_dir / "results.csv")
            self._download_top_level_pngs(sftp, trial.remote_run_dir, cache_dir)
        except OSError as exc:
            sync_error = str(exc)
            self.repo.update_trial(
                trial_id,
                sync_status=REMOTE_SYNC_FAILED,
                sync_error=sync_error,
                last_synced_at=utc_now_iso(),
            )
            raise ServiceError(f"remote sync failed: {sync_error}") from exc
        finally:
            sftp.close()
            client.close()

        args_data = _parse_args_yaml((cache_dir / "args.yaml").read_text(encoding="utf-8"))
        epoch_count = _valid_epoch_count(cache_dir / "results.csv")
        remote_size = int(getattr(csv_stat, "st_size", 0) or 0)
        remote_mtime = float(getattr(csv_stat, "st_mtime", 0) or 0)
        unchanged = (
            trial.last_remote_csv_size == remote_size
            and trial.last_remote_csv_mtime == remote_mtime
        )
        unchanged_count = trial.unchanged_sync_count + 1 if unchanged else 0
        remote_training_status = self._remote_training_status(args_data, epoch_count, unchanged_count, remote_size)
        summary_path = str(cache_dir / SUMMARY_FILENAME)
        final_metrics: dict[str, Any] = {}
        next_status = STATE_TRAINING
        if epoch_count > 0:
            try:
                previous_summary = self._previous_summary_for_trial(trial.experiment_id, trial.trial_id)
                summary = build_summary(
                    trial.trial_id,
                    config.task_type,
                    str(cache_dir),
                    trial.params,
                    previous_summary,
                ).to_dict()
                summary["remote"] = self._remote_trial_payload(trial, server)
                write_json(summary_path, summary)
                final_metrics = summary["final_metrics"]
                next_status = (
                    self._completion_status(config, summary, len(self.repo.list_trials(trial.experiment_id)))
                    if remote_training_status == REMOTE_TRAINING_COMPLETED
                    else STATE_TRAINING
                    if remote_training_status == REMOTE_TRAINING_RUNNING
                    else STATE_WAITING
                )
            except Exception as exc:
                sync_error = f"summary parse failed: {exc}"
                next_status = STATE_TRAINING
        else:
            sync_error = "results.csv has no valid epoch rows"

        self.repo.update_trial(
            trial_id,
            status=next_status,
            metrics=final_metrics if final_metrics else None,
            summary_path=summary_path if final_metrics else None,
            sync_status=REMOTE_SYNC_SYNCED if not sync_error else REMOTE_SYNC_FAILED,
            sync_error=sync_error,
            remote_training_status=remote_training_status,
            last_remote_csv_size=remote_size,
            last_remote_csv_mtime=remote_mtime,
            last_synced_epoch_count=epoch_count,
            unchanged_sync_count=unchanged_count,
            last_synced_at=utc_now_iso(),
        )
        self.repo.update_experiment_status(trial.experiment_id, next_status)
        self.repo.add_event(
            trial.experiment_id,
            "REMOTE_TRIAL_SYNCED",
            {
                "trial_id": trial_id,
                "remote_training_status": remote_training_status,
                "epoch_count": epoch_count,
                "sync_error": sync_error,
            },
            trial_id,
        )
        return {
            "status": next_status,
            "trial_id": trial_id,
            "sync_status": REMOTE_SYNC_SYNCED if not sync_error else REMOTE_SYNC_FAILED,
            "sync_error": sync_error,
            "remote_training_status": remote_training_status,
            "epoch_count": epoch_count,
            "final_metrics": final_metrics,
            "summary_path": summary_path if final_metrics else None,
        }

    def compare_experiment(self, experiment_id: str) -> dict[str, Any]:
        config = self.repo.get_experiment(experiment_id)
        rows = [self._trial_row(trial) for trial in self.repo.list_trials(experiment_id)]
        metric = config.goal.metric
        best_row = None
        best_value = None
        for row in rows:
            value = row.get(metric)
            if isinstance(value, (int, float)) and (best_value is None or value > best_value):
                best_value = float(value)
                best_row = row
        for row in rows:
            row["is_best"] = bool(best_row and row["trial_id"] == best_row["trial_id"])
        target_reached = bool(best_value is not None and best_value >= config.goal.target)
        columns = [
            {"key": "iteration", "label": "Iteration"},
            {"key": "trial_id", "label": "Trial"},
            {"key": "status", "label": "Status"},
            {"key": "model_display", "label": "Model"},
            {"key": "source", "label": "Source"},
            {"key": "server", "label": "Location"},
            {"key": "map50_95", "label": "mAP50-95"},
            {"key": "map50", "label": "mAP50"},
            {"key": "precision", "label": "Precision"},
            {"key": "recall", "label": "Recall"},
            {"key": "delta_map50_95", "label": "Delta mAP50-95"},
            {"key": "best_epoch", "label": "Best Epoch"},
            {"key": "epochs_completed", "label": "Epochs"},
            {"key": "train_time_sec", "label": "Train Time"},
            {"key": "gpu_mem_peak", "label": "GPU Mem"},
            {"key": "params", "label": "Params"},
            {"key": "note", "label": "Note"},
        ]
        return {
            "experiment_id": experiment_id,
            "goal": config.goal.__dict__,
            "target_reached": target_reached,
            "best_trial": None
            if best_row is None
            else {
                "trial_id": best_row["trial_id"],
                "iteration": best_row["iteration"],
                "metric": metric,
                "value": best_value,
            },
            "columns": columns,
            "rows": rows,
        }

    def _latest_params(
        self,
        config: ExperimentConfig,
        trials: list[TrialRecord],
    ) -> dict[str, Any]:
        base_params = build_initial_params(config.task_type, config.initial_params)
        for trial in reversed(trials):
            if trial.params:
                merged = dict(base_params)
                merged.update(trial.params)
                return merged
        return dict(base_params)

    def _next_iteration(self, trials: list[TrialRecord]) -> int:
        if not trials:
            return 1
        return max(trial.iteration for trial in trials) + 1

    def _ensure_openclaw_training_limit(
        self,
        config: ExperimentConfig,
        trials: list[TrialRecord],
    ) -> None:
        if not config.session_key:
            return
        started_training_count = sum(1 for trial in trials if trial.source == "trained")
        if started_training_count >= OPENCLAW_MAX_TRAINING_RUNS:
            raise ServiceError(
                "openclaw training run limit reached for this task: "
                f"{started_training_count}/{OPENCLAW_MAX_TRAINING_RUNS}"
            )

    def _next_named_trial_id(
        self,
        experiment_id: str,
        model: str,
        params: dict[str, Any],
    ) -> str:
        model_stem = _model_stem(model)
        imgsz = int(params.get("imgsz") or 0)
        prefix = f"{model_stem}_{imgsz}"
        trials = self.repo.list_trials(experiment_id)
        max_index = 0
        for trial in trials:
            if not trial.trial_id.startswith(f"{prefix}_"):
                continue
            try:
                max_index = max(max_index, int(trial.trial_id.rsplit("_", 1)[1]))
            except (ValueError, IndexError):
                continue
        index = max_index + 1
        while True:
            candidate = f"{prefix}_{index}"
            if not self.repo.trial_id_exists(candidate):
                return candidate
            index += 1

    def _next_trial_display_name(
        self,
        experiment_id: str,
        model: str,
        params: dict[str, Any],
    ) -> str:
        model_stem = _model_stem(model)
        imgsz = int(params.get("imgsz") or 0)
        prefix = f"{model_stem}_{imgsz}"
        trials = self.repo.list_trials(experiment_id)
        max_index = 0
        for trial in trials:
            if not trial.display_name.startswith(f"{prefix}_"):
                continue
            try:
                max_index = max(max_index, int(trial.display_name.rsplit("_", 1)[1]))
            except (ValueError, IndexError):
                continue
        index = max_index + 1
        while True:
            candidate = f"{prefix}_{index}"
            if not self.repo.trial_display_name_exists(experiment_id, candidate):
                return candidate
            index += 1

    def _validate_trial_display_name(self, display_name: str) -> str:
        normalized = str(display_name or "").strip()
        if not normalized:
            raise ServiceError("trial name cannot be empty")
        return normalized

    def _previous_summary_for_trial(
        self,
        experiment_id: str,
        trial_id: str,
    ) -> dict[str, Any] | None:
        summaries: list[dict[str, Any]] = []
        for trial in self.repo.list_trials(experiment_id):
            if trial.trial_id == trial_id:
                break
            if trial.summary_path and Path(trial.summary_path).exists():
                summaries.append(read_json(trial.summary_path))
        return summaries[-1] if summaries else None

    def _remote_join(self, remote_dir: str, filename: str) -> str:
        return posixpath.join(remote_dir.rstrip("/"), filename)

    def _open_sftp(self, server: RemoteServer) -> tuple[Any, Any]:
        try:
            import paramiko
        except ImportError as exc:
            raise ServiceError("paramiko is required for remote SFTP support") from exc

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs: dict[str, Any] = {
            "hostname": server.host,
            "port": int(server.port),
            "username": server.username,
            "timeout": 15,
        }
        if server.auth_type == "key":
            kwargs["key_filename"] = str(Path(server.private_key_path).expanduser())
        else:
            password = os.environ.get(server.password_ref)
            if password is None:
                raise ServiceError(f"password env var not found: {server.password_ref}")
            kwargs["password"] = password
        try:
            client.connect(**kwargs)
            return client, client.open_sftp()
        except Exception as exc:
            client.close()
            raise ServiceError(f"failed to connect remote server {server.remote_server_id}: {exc}") from exc

    def _read_remote_text(self, server: RemoteServer, remote_path: str) -> str:
        client, sftp = self._open_sftp(server)
        try:
            with sftp.open(remote_path, "r") as handle:
                data = handle.read()
            if isinstance(data, bytes):
                return data.decode("utf-8")
            return str(data)
        except OSError as exc:
            raise ServiceError(f"failed to read remote file {remote_path}: {exc}") from exc
        finally:
            sftp.close()
            client.close()

    def _download_remote_file(self, sftp: Any, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = local_path.with_name(f".{local_path.name}.{os.getpid()}.tmp")
        try:
            sftp.get(remote_path, str(temp_path))
            try:
                os.replace(temp_path, local_path)
            except PermissionError:
                if local_path.exists():
                    local_path.unlink()
                temp_path.rename(local_path)
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def _download_top_level_pngs(self, sftp: Any, remote_dir: str, cache_dir: Path) -> None:
        try:
            entries = sftp.listdir_attr(remote_dir)
        except OSError:
            return
        for entry in entries:
            name = getattr(entry, "filename", "")
            if not name.lower().endswith(".png"):
                continue
            try:
                self._download_remote_file(
                    sftp,
                    self._remote_join(remote_dir, name),
                    cache_dir / name,
                )
            except OSError:
                continue

    def _remote_training_status(
        self,
        args_data: dict[str, Any],
        epoch_count: int,
        unchanged_count: int,
        remote_size: int,
    ) -> str:
        try:
            planned_epochs = int(args_data.get("epochs"))
        except (TypeError, ValueError):
            planned_epochs = 0
        if planned_epochs > 0 and epoch_count >= planned_epochs:
            return REMOTE_TRAINING_COMPLETED
        if epoch_count <= 0 or remote_size <= 0:
            return REMOTE_TRAINING_UNKNOWN
        if unchanged_count >= 2:
            return REMOTE_TRAINING_MAYBE_STOPPED
        return REMOTE_TRAINING_RUNNING

    def _remote_trial_payload(self, trial: TrialRecord, server: RemoteServer) -> dict[str, Any]:
        return {
            "remote_server_id": trial.remote_server_id,
            "remote_server_name": server.name,
            "remote_run_dir": trial.remote_run_dir,
            "sync_status": trial.sync_status,
            "sync_error": trial.sync_error,
            "remote_training_status": trial.remote_training_status,
            "last_synced_at": trial.last_synced_at,
        }

    def _editable_schema(self, search_space: dict[str, Any]) -> dict[str, Any]:
        schema: dict[str, Any] = {}
        for name, spec in search_space.items():
            field = dict(spec)
            field["name"] = name
            field["required"] = True
            schema[name] = field
        return schema

    def _trial_logs(self, run_dir: str) -> dict[str, str | None]:
        run_path = Path(run_dir)
        stdout_log = run_path / "stdout.log"
        stderr_log = run_path / "stderr.log"
        return {
            "stdout": str(stdout_log) if stdout_log.exists() else None,
            "stderr": str(stderr_log) if stderr_log.exists() else None,
        }

    def _trial_row(self, trial: TrialRecord) -> dict[str, Any]:
        summary = read_json(trial.summary_path) if trial.summary_path and Path(trial.summary_path).exists() else {}
        final_metrics = summary.get("final_metrics", trial.metrics or {})
        delta = summary.get("delta_vs_prev", {})
        basic = summary.get("basic_info", {})
        resource = summary.get("resource", {})
        params = summary.get("params", trial.params)
        server_name = "local"
        if trial.remote_server_id:
            try:
                server_name = self.repo.get_remote_server(trial.remote_server_id).name
            except KeyError:
                server_name = trial.remote_server_id
        return {
            "iteration": trial.iteration,
            "trial_id": trial.trial_id,
            "display_name": trial.display_name,
            "status": trial.status,
            "source": trial.source,
            "model": trial.model,
            "model_display": _model_basename(trial.model),
            "server": server_name,
            "remote_server_id": trial.remote_server_id,
            "precision": final_metrics.get("precision"),
            "recall": final_metrics.get("recall"),
            "map50": final_metrics.get("map50"),
            "map50_95": final_metrics.get("map50_95"),
            "delta_map50_95": delta.get("map50_95"),
            "delta_recall": delta.get("recall"),
            "best_epoch": basic.get("best_epoch"),
            "epochs_completed": basic.get("epochs_completed"),
            "train_time_sec": basic.get("train_time_sec"),
            "gpu_mem_peak": resource.get("gpu_mem_peak"),
            "params": params,
            "run_dir": trial.run_dir,
            "summary_path": trial.summary_path,
            "note": trial.note,
            "reason": trial.reason,
            "remote_training_status": trial.remote_training_status,
            "last_synced_at": trial.last_synced_at,
            "logs": self._trial_logs(trial.run_dir),
            "is_best": False,
        }

    def _completion_status(
        self,
        config: ExperimentConfig,
        summary: dict[str, Any],
        total_trial_count: int,
    ) -> str:
        """Determine experiment status after a trial completes.

        *total_trial_count* must be the **actual number of trials** that exist
        for this experiment (not the iteration ordinal which may diverge after
        deletions).
        """
        metric_value = float(summary["final_metrics"].get(config.goal.metric, 0.0))
        if metric_value >= config.goal.target:
            return STATE_COMPLETED
        if total_trial_count >= int(config.stop_conditions["max_trials"]):
            return STATE_COMPLETED
        return STATE_WAITING

    def _notify_training_result(
        self,
        config: ExperimentConfig,
        trial_id: str,
        summary: dict[str, Any],
    ) -> None:
        if not config.session_key:
            return
        try:
            session_id = _resolve_session_id(config.session_key)
            compact_summary = _notification_summary(summary)
            trial = self.repo.get_trial(trial_id)
            message = _build_notify_message(config, trial.display_name, compact_summary)
            _notify_openclaw_session(session_id, message)
            self.repo.add_event(
                config.experiment_id,
                "OPENCLAW_NOTIFY_SENT",
                {
                    "trial_id": trial_id,
                    "session_key": config.session_key,
                    "session_id": session_id,
                    "compact_summary": compact_summary,
                },
                trial_id,
            )
        except ServiceError as exc:
            self.repo.add_event(
                config.experiment_id,
                "OPENCLAW_NOTIFY_FAILED",
                {
                    "trial_id": trial_id,
                    "session_key": config.session_key,
                    "error": str(exc),
                },
                trial_id,
            )

    def _cleanup_stale_empty_tasks(self) -> None:
        stale_experiments = self.repo.stale_unstarted_experiments(
            _stale_task_cutoff_iso(STALE_EMPTY_TASK_TTL_HOURS),
            STATE_READY,
        )
        for config in stale_experiments:
            experiment_dir = Path(config.save_root).resolve() / "experiments" / config.experiment_id
            self.repo.delete_events_for_experiment(config.experiment_id)
            self.repo.delete_experiment(config.experiment_id)
            if experiment_dir.exists():
                try:
                    save_root = Path(config.save_root).resolve()
                    experiments_root = (save_root / "experiments").resolve()
                    resolved_experiment_dir = experiment_dir.resolve()
                    if experiments_root in resolved_experiment_dir.parents:
                        shutil.rmtree(resolved_experiment_dir, onerror=_handle_rmtree_error)
                except Exception:
                    continue

    def get_experiment_curves(self, experiment_id: str) -> dict[str, Any]:
        trials = self.repo.list_trials(experiment_id)
        curves = {}
        trial_labels = {}
        for trial in trials:
            results_csv = Path(trial.run_dir) / "results.csv"
            if not results_csv.exists():
                continue
            
            import csv
            trial_data = []
            with open(results_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cleaned_row = {}
                    for k, v in row.items():
                        if k and v:
                            v_str = str(v).strip()
                            if v_str:
                                try:
                                    cleaned_row[str(k).strip()] = float(v_str) if '.' in v_str else int(v_str)
                                except ValueError:
                                    pass
                    if "epoch" in cleaned_row:
                        trial_data.append(cleaned_row)
            curves[trial.trial_id] = trial_data
            trial_labels[trial.trial_id] = trial.display_name

        return {"experiment_id": experiment_id, "curves": curves, "trial_labels": trial_labels}

    def get_trial_visualizations(self, trial_id: str) -> dict[str, Any]:
        trial = self.repo.get_trial(trial_id)
        run_dir = Path(trial.run_dir)
        visualizations = []
        if run_dir.exists():
            for f in sorted(run_dir.iterdir()):
                if f.is_file() and (f.name.startswith("train_batch") or f.name.startswith("val_batch") or f.name.endswith(".png")):
                    visualizations.append(f.name)
        return {"trial_id": trial_id, "visualizations": visualizations}

    def get_trial_file_path(self, trial_id: str, filename: str) -> str:
        trial = self.repo.get_trial(trial_id)
        run_dir_resolved = Path(trial.run_dir).resolve()
        file_path = (run_dir_resolved / filename).resolve()
        if not file_path.is_relative_to(run_dir_resolved):
            raise ServiceError("invalid filename")
        if not file_path.exists() or not file_path.is_file():
            raise ServiceError("file not found")
        return str(file_path)

    def get_validation_preview_file_path(self, trial_id: str, validation_id: str, filename: str) -> str:
        trial = self.repo.get_trial(trial_id)
        normalized_validation_id = _validate_preview_filename(validation_id)
        normalized_filename = _validate_preview_filename(filename)
        run_dir_resolved = Path(trial.run_dir).resolve()
        preview_dir = (run_dir_resolved / ".validation_previews" / normalized_validation_id).resolve()
        if not preview_dir.is_relative_to(run_dir_resolved):
            raise ServiceError("invalid validation id")
        file_path = (preview_dir / normalized_filename).resolve()
        if not file_path.is_relative_to(preview_dir):
            raise ServiceError("invalid filename")
        if not file_path.exists() or not file_path.is_file():
            raise ServiceError("file not found")
        return str(file_path)
