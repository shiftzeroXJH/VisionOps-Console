from __future__ import annotations

import json
import math
import os
import posixpath
import re
import subprocess
import shutil
import stat
import sys
import shlex
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import uuid4

from backend.constants import (
    EXPERIMENT_FILENAME,
    SEARCH_SPACE,
    DEFAULT_MAX_PARALLEL_TRAINING_TASKS,
    MAX_PARALLEL_TRAINING_SETTING_KEY,
    MAX_PARALLEL_TRAINING_TASKS_LIMIT,
    STATE_ANALYZING,
    STATE_CANCELLED,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_INIT,
    STATE_QUEUED,
    STATE_READY,
    STATE_RETRAINING,
    STATE_TRAINING,
    STATE_WAITING,
    PUBLIC_STATUS_COMPLETED,
    PUBLIC_STATUS_INTERRUPTED_OR_FAILED,
    PUBLIC_STATUS_NOT_STARTED,
    PUBLIC_STATUS_QUEUED,
    PUBLIC_STATUS_TRAINING,
    STOP_CONDITIONS,
    SUMMARY_FILENAME,
    TASK_BASELINES,
    TRIAL_CONFIG_FILENAME,
)
from backend.core.analyzer import build_summary
from backend.core.baseline import build_initial_params
from backend.core.constraints import validate_param_value
from backend.core.dataset import analyze_dataset, inspect_dataset
from backend.core.metrics import calculate_fitness, fitness_metric
from backend.core.trainer import (
    TrainingCancelledError,
    TrainingError,
    cancel_training_process,
    run_training,
)
from backend.db.repository import Repository, default_project_name
from backend.models import ExperimentConfig, HyperparameterTemplate, RemoteServer, TrialRecord
from backend.utils import ensure_dir, read_json, utc_now_iso, write_json
from backend.workbench import WorkbenchService


class ServiceError(RuntimeError):
    pass


class TemplateNameConflictError(ServiceError):
    pass


def public_task_status(status: str, remote_training_status: str = "", sync_status: str = "") -> str:
    """Map detailed persistence states to the four user-facing task states."""
    if remote_training_status == REMOTE_TRAINING_MAYBE_STOPPED or sync_status == REMOTE_SYNC_FAILED:
        return PUBLIC_STATUS_INTERRUPTED_OR_FAILED
    if status in {STATE_INIT, STATE_READY}:
        return PUBLIC_STATUS_NOT_STARTED
    if status == STATE_QUEUED:
        return PUBLIC_STATUS_QUEUED
    if status in {STATE_TRAINING, STATE_RETRAINING, STATE_ANALYZING}:
        return PUBLIC_STATUS_TRAINING
    if status in {STATE_COMPLETED, STATE_WAITING}:
        return PUBLIC_STATUS_COMPLETED
    if status in {STATE_CANCELLED, STATE_FAILED}:
        return PUBLIC_STATUS_INTERRUPTED_OR_FAILED
    return PUBLIC_STATUS_INTERRUPTED_OR_FAILED


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
YOLO_PYTHON_SETTING_KEY = "yolo_python"
DEFAULT_EXPORT_DIR = "exports"
PROJECT_EXPORT_DIR_SETTING_PREFIX = "project_default_export_dir:"
VALIDATION_PREVIEW_DIRNAME = ".validation_previews"
VALIDATION_CURRENT_DIRNAME = "current"
VALIDATION_RESULT_FILENAME = "validation_result.json"
ILLEGAL_WINDOWS_NAME_CHARS = set('<>:"/\\|?*')
PLATFORM_CONTROLLED_YOLO_PARAMS = frozenset(
    {
        "model", "data", "task", "mode", "project", "name", "save_dir", "exist_ok",
        "device", "cache", "seed", "deterministic", "pretrained", "plots", "save",
        "save_period", "save_conf", "save_crop", "save_frames", "save_json", "save_txt",
        "show", "visualize", "resume", "val", "verbose",
    }
)
_YOLO_PARAM_SCHEMA_CACHE: dict[str, dict[str, dict[str, str]]] = {}


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
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
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


def _continuation_weight_path(run_dir: str) -> Path:
    base = Path(run_dir)
    for candidate in (base / "weights" / "last.pt", base / "last.pt"):
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    raise ServiceError(f"last.pt not found under run_dir: {run_dir}")


def _export_filename(model_name: str) -> str:
    if model_name.lower().endswith(".onnx"):
        return model_name
    return f"{model_name}.onnx"


def _project_export_dir_setting_key(project: str) -> str:
    return f"{PROJECT_EXPORT_DIR_SETTING_PREFIX}{project}"


def _default_python_candidates() -> list[str]:
    candidates: list[str] = []
    env_python = os.environ.get("YOLO_PYTHON", "").strip()
    if env_python:
        candidates.append(env_python)
    candidates.append(sys.executable)
    return candidates


def _python_for_yolo() -> str:
    for candidate in _default_python_candidates():
        normalized = str(candidate or "").strip()
        if normalized and Path(normalized).exists():
            return normalized
    return sys.executable


def _validate_preview_filename(filename: str) -> str:
    normalized = str(filename or "").strip()
    if not normalized:
        raise ServiceError("filename is required")
    if "/" in normalized or "\\" in normalized or normalized in {".", ".."}:
        raise ServiceError("invalid filename")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", normalized):
        raise ServiceError("invalid filename")
    return normalized


def _validation_preview_root(run_dir: str) -> Path:
    run_path = Path(run_dir).resolve()
    preview_root = (run_path / VALIDATION_PREVIEW_DIRNAME).resolve()
    if not preview_root.is_relative_to(run_path):
        raise ServiceError("invalid validation preview path")
    return preview_root


def _last_json_object(path: Path) -> dict[str, Any] | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            value = json.loads(line.strip())
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


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


def _params_from_args(args: dict[str, Any], allowed_extra_params: set[str] | None = None) -> dict[str, Any]:
    return {
        key: args[key]
        for key in set(SEARCH_SPACE) | (allowed_extra_params or set())
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
        "per_class_metrics": summary.get("per_class_metrics", []),
        "metric_breakdown": summary.get("metric_breakdown", {}),
        "delta_vs_prev": summary.get("delta_vs_prev", {}),
        "metric_breakdown_delta_vs_prev": summary.get("metric_breakdown_delta_vs_prev", {}),
        "training_dynamics": summary.get("training_dynamics", {}),
        "warnings": summary.get("warnings", []),
        "resource": summary.get("resource", {}),
        "basic_info": summary.get("basic_info", {}),
        "params": summary.get("params", {}),
    }


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
            repo_path = str(Path(db_path or "yolo_state.sqlite").resolve())
        self.repo = Repository(repo_path)
        self._migrate_experiment_files()
        self._bootstrap_python_setting()
        workbench_cache = None
        if repo_path == ":memory:":
            workbench_cache = Path(".workbench_cache") / "tests" / uuid4().hex
        self.workbench = WorkbenchService(self.repo, self._python_for_yolo, cache_root=workbench_cache)

    def _migrate_experiment_files(self) -> None:
        """Remove retired goal data from persisted experiment snapshots."""
        if not hasattr(self.repo, "list_experiments"):
            return
        for config in self.repo.list_experiments():
            experiment_json = Path(config.save_root) / "experiments" / config.experiment_id / EXPERIMENT_FILENAME
            if not experiment_json.exists():
                continue
            try:
                current = read_json(experiment_json)
            except (OSError, TypeError, ValueError):
                continue
            if not isinstance(current, dict):
                continue
            if "goal" in current or current.get("status") != config.status:
                write_json(experiment_json, config.to_dict())

    def _experiment_api_payload(self, config: ExperimentConfig) -> dict[str, Any]:
        trials = self.repo.list_trials(config.experiment_id)
        latest_trial = trials[-1] if trials else None
        effective_status = self._effective_experiment_status(config.experiment_id, config.status)
        payload = config.to_dict()
        payload["status"] = public_task_status(
            effective_status,
            latest_trial.remote_training_status if latest_trial else "",
            latest_trial.sync_status if latest_trial else "",
        )
        payload["internal_status"] = effective_status
        return payload

    def _effective_experiment_status(self, experiment_id: str, stored_status: str) -> str:
        active_tasks = self.repo.list_training_tasks(("RUNNING", "QUEUED"))
        statuses = {task.status for task in active_tasks if task.experiment_id == experiment_id}
        if "RUNNING" in statuses:
            return STATE_TRAINING
        if "QUEUED" in statuses:
            return STATE_QUEUED
        return stored_status

    def _yolo_param_schema(self) -> dict[str, dict[str, str]]:
        python_executable = self._python_for_yolo()
        cached = _YOLO_PARAM_SCHEMA_CACHE.get(python_executable)
        if cached is not None:
            return cached
        src_root = str(Path(__file__).resolve().parent.parent)
        env = dict(os.environ)
        env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else f"{src_root}{os.pathsep}{env['PYTHONPATH']}"
        try:
            result = subprocess.run(
                [python_executable, "-m", "backend.core.yolo_schema_worker"],
                capture_output=True,
                check=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=15,
            )
            raw_schema = json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            raise ServiceError(f"failed to load Ultralytics parameter schema: {exc}") from exc
        if not isinstance(raw_schema, dict):
            raise ServiceError("failed to load Ultralytics parameter schema")
        schema = {
            str(name): {"type": str(spec.get("type", "json"))}
            for name, spec in raw_schema.items()
            if isinstance(name, str) and isinstance(spec, dict)
        }
        _YOLO_PARAM_SCHEMA_CACHE[python_executable] = schema
        return schema

    def _extra_param_schema(self) -> dict[str, dict[str, str]]:
        return {
            name: spec
            for name, spec in self._yolo_param_schema().items()
            if name not in SEARCH_SPACE and name not in PLATFORM_CONTROLLED_YOLO_PARAMS
        }

    def _validate_extra_param_value(self, name: str, value: Any, expected_type: str) -> Any:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid JSON value for '{name}'") from exc
        if expected_type == "boolean" and not isinstance(value, bool):
            raise ValueError(f"invalid boolean value for '{name}'")
        if expected_type == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"invalid integer value for '{name}'")
        if expected_type == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"invalid number value for '{name}'")
        if expected_type == "string" and not isinstance(value, str):
            raise ValueError(f"invalid string value for '{name}'")
        return value

    def inspect_dataset(self, dataset_root: str) -> dict[str, Any]:
        return {"yaml_candidates": inspect_dataset(dataset_root)}

    def _bootstrap_python_setting(self) -> None:
        if not hasattr(self.repo, "get_setting") or not hasattr(self.repo, "set_setting"):
            return
        configured = self.repo.get_setting(YOLO_PYTHON_SETTING_KEY, "").strip()
        if configured:
            return
        env_python = os.environ.get("YOLO_PYTHON", "").strip()
        if not env_python:
            return
        path = Path(env_python).expanduser()
        if not path.exists() or path.is_dir():
            return
        self.repo.set_setting(YOLO_PYTHON_SETTING_KEY, str(path.resolve()))

    def _python_for_yolo(self) -> str:
        configured = self.repo.get_setting(YOLO_PYTHON_SETTING_KEY, "").strip()
        if configured:
            return configured
        return _python_for_yolo()

    def get_settings(self) -> dict[str, Any]:
        configured = self.repo.get_setting(YOLO_PYTHON_SETTING_KEY, "").strip()
        effective = self._python_for_yolo()
        return {
            "yolo_python": configured,
            "effective_yolo_python": effective,
            "uses_default_python": not bool(configured),
            "max_parallel_training_tasks": self._max_parallel_training_tasks(),
        }

    def _max_parallel_training_tasks(self) -> int:
        raw = self.repo.get_setting(
            MAX_PARALLEL_TRAINING_SETTING_KEY,
            str(DEFAULT_MAX_PARALLEL_TRAINING_TASKS),
        )
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = DEFAULT_MAX_PARALLEL_TRAINING_TASKS
        return max(1, min(MAX_PARALLEL_TRAINING_TASKS_LIMIT, value))

    def update_settings(
        self,
        *,
        yolo_python: str | None = None,
        max_parallel_training_tasks: int | None = None,
    ) -> dict[str, Any]:
        if yolo_python is not None:
            normalized = str(yolo_python or "").strip()
            if normalized:
                path = Path(normalized).expanduser()
                if not path.exists():
                    raise ServiceError(f"python executable not found: {normalized}")
                if path.is_dir():
                    raise ServiceError(f"python executable is a directory: {normalized}")
                normalized = str(path.resolve())
            self.repo.set_setting(YOLO_PYTHON_SETTING_KEY, normalized)
        if max_parallel_training_tasks is not None:
            try:
                normalized_max = int(max_parallel_training_tasks)
            except (TypeError, ValueError) as exc:
                raise ServiceError("max_parallel_training_tasks must be an integer") from exc
            if not 1 <= normalized_max <= MAX_PARALLEL_TRAINING_TASKS_LIMIT:
                raise ServiceError(
                    f"max_parallel_training_tasks must be between 1 and {MAX_PARALLEL_TRAINING_TASKS_LIMIT}"
                )
            self.repo.set_setting(MAX_PARALLEL_TRAINING_SETTING_KEY, str(normalized_max))
        return self.get_settings()

    def _project_default_export_dir(self, project: str) -> str:
        configured = self.repo.get_setting(_project_export_dir_setting_key(project), "").strip()
        return configured or DEFAULT_EXPORT_DIR

    def get_project_settings(self, project: str) -> dict[str, Any]:
        normalized_project = str(project or "").strip() or default_project_name("")
        experiments = [config for config in self.repo.list_experiments() if config.project == normalized_project]
        if not experiments:
            raise ServiceError(f"project not found: {normalized_project}")
        return {
            "project": normalized_project,
            "default_export_dir": self._project_default_export_dir(normalized_project),
            "experiment_count": len(experiments),
        }

    def update_project_settings(
        self,
        project: str,
        *,
        name: str | None = None,
        default_export_dir: str | None = None,
    ) -> dict[str, Any]:
        normalized_project = str(project or "").strip() or default_project_name("")
        experiments = [config for config in self.repo.list_experiments() if config.project == normalized_project]
        if not experiments:
            raise ServiceError(f"project not found: {normalized_project}")

        next_project = normalized_project
        if name is not None:
            next_project = str(name or "").strip()
            if not next_project:
                raise ServiceError("project name cannot be empty")

        if next_project != normalized_project:
            for config in experiments:
                self.repo.update_experiment(config.experiment_id, project=next_project)
                experiment_json = Path(config.save_root) / "experiments" / config.experiment_id / EXPERIMENT_FILENAME
                if experiment_json.exists():
                    config.project = next_project
                    write_json(experiment_json, config.to_dict())
                self.repo.add_event(
                    config.experiment_id,
                    "EXPERIMENT_UPDATED",
                    {"description": config.description, "project": next_project},
                )

        export_dir = self._project_default_export_dir(normalized_project)
        if default_export_dir is not None:
            export_dir = str(default_export_dir or "").strip() or DEFAULT_EXPORT_DIR
        if next_project != normalized_project:
            self.repo.set_setting(_project_export_dir_setting_key(normalized_project), "")
        self.repo.set_setting(_project_export_dir_setting_key(next_project), export_dir)

        return {
            "project": next_project,
            "previous_project": normalized_project,
            "default_export_dir": export_dir,
            "experiment_count": len(experiments),
        }

    def delete_project(self, project: str, *, confirmation: str) -> dict[str, Any]:
        normalized_project = str(project or "").strip() or default_project_name("")
        if confirmation != "确认删除":
            raise ServiceError("confirmation text must be 确认删除")
        experiments = [config for config in self.repo.list_experiments() if config.project == normalized_project]
        if not experiments:
            raise ServiceError(f"project not found: {normalized_project}")

        results = [
            self.delete_task(config.experiment_id, keep_files=False, force=True)
            for config in experiments
        ]
        self.repo.set_setting(_project_export_dir_setting_key(normalized_project), "")
        return {
            "project": normalized_project,
            "deleted": True,
            "deleted_experiments": len(results),
            "deleted_trials": sum(int(result.get("deleted_trials", 0)) for result in results),
            "files_deleted": any(bool(result.get("files_deleted")) for result in results),
            "results": results,
        }

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
                    "remote_python": server.remote_python,
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
        remote_python: str | None = None,
        password: str | None = None,
    ) -> dict[str, Any]:
        normalized_auth_type = auth_type.strip().lower()
        if normalized_auth_type not in {"key", "password"}:
            raise ServiceError("auth_type must be 'key' or 'password'")
        if normalized_auth_type == "key" and not (private_key_path or "").strip():
            raise ServiceError("private_key_path is required for key auth")
        if normalized_auth_type == "password" and not (password or "").strip() and not (password_ref or "").strip():
            raise ServiceError("password or password_ref is required for password auth")

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
            remote_python=(remote_python or "").strip(),
            password=(password or "").strip(),
        )
        if not server.host:
            raise ServiceError("host is required")
        if not server.username:
            raise ServiceError("username is required")
        self.repo.create_remote_server(server)
        payload = dict(server.__dict__)
        payload.pop("password", None)
        return {"remote_server": payload}

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

    def update_remote_server(self, remote_server_id: str, **fields: Any) -> dict[str, Any]:
        current = self.repo.get_remote_server(remote_server_id)
        values = {key: str(value).strip() for key, value in fields.items() if value is not None}
        if not values.get("name", current.name):
            raise ServiceError("name is required")
        if not values.get("host", current.host) or not values.get("username", current.username):
            raise ServiceError("host and username are required")
        if not values.get("remote_python", current.remote_python) or not values.get("default_runs_root", current.default_runs_root):
            raise ServiceError("remote_python and default_runs_root are required")
        if "password" in values and not values["password"]:
            values.pop("password")
        if "port" in fields:
            values["port"] = int(fields["port"])
        self.repo.update_remote_server(remote_server_id, **values)
        updated = self.repo.get_remote_server(remote_server_id)
        payload = dict(updated.__dict__)
        payload.pop("password", None)
        return {"remote_server": payload}

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
        initial_params: dict[str, Any] | None = None,
        remote_configs: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
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
            task_type=task_type,
            dataset_root=str(Path(dataset_root).resolve()),
            dataset_yaml=str(Path(dataset_yaml).resolve()),
            pretrained_model=resolved_pretrained,
            save_root=str(Path(save_root).resolve()),
            status=STATE_READY,
            initial_params=build_initial_params(task_type, initial_overrides),
            search_space=SEARCH_SPACE,
            stop_conditions=STOP_CONDITIONS,
            remote_configs=remote_configs or {},
        )
        self.repo.create_experiment(config)
        write_json(experiment_dir / EXPERIMENT_FILENAME, config.to_dict())
        self.repo.add_event(experiment_id, "EXPERIMENT_CREATED", config.to_dict())
        return {
            "status": public_task_status(config.status),
            "internal_status": config.status,
            "experiment_id": experiment_id,
            "description": config.description,
            "project": config.project,
            "dataset_yaml": config.dataset_yaml,
            "initial_params": config.initial_params,
            "experiment_dir": str(experiment_dir),
        }

    def list_experiments(self) -> dict[str, Any]:
        experiments = self.repo.list_experiments()
        active_training_statuses: dict[str, set[str]] = {}
        for task in self.repo.list_training_tasks(("RUNNING", "QUEUED")):
            active_training_statuses.setdefault(task.experiment_id, set()).add(task.status)
        items: list[dict[str, Any]] = []
        for config in experiments:
            trials = self.repo.list_trials(config.experiment_id)
            latest_trial = trials[-1] if trials else None
            queue_statuses = active_training_statuses.get(config.experiment_id, set())
            effective_status = (
                STATE_TRAINING
                if "RUNNING" in queue_statuses
                else STATE_QUEUED
                if "QUEUED" in queue_statuses
                else config.status
            )
            metric = "map50_95"
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
                    "status": public_task_status(
                        effective_status,
                        latest_trial.remote_training_status if latest_trial else "",
                        latest_trial.sync_status if latest_trial else "",
                    ),
                    "internal_status": effective_status,
                    "task_type": config.task_type,
                    "dataset_root": config.dataset_root,
                    "dataset_yaml": config.dataset_yaml,
                    "pretrained_model": config.pretrained_model,
                    "default_export_dir": self._project_default_export_dir(config.project),
                    "trial_count": len(trials),
                    "best_metric": best_trial_info,
                    "latest_trial": None
                    if latest_trial is None
                    else {
                        "trial_id": latest_trial.trial_id,
                        "iteration": latest_trial.iteration,
                        "status": public_task_status(
                            latest_trial.status,
                            latest_trial.remote_training_status,
                            latest_trial.sync_status,
                        ),
                        "internal_status": latest_trial.status,
                        "metrics": latest_trial.metrics,
                        "source": latest_trial.source,
                        "model": _model_basename(latest_trial.model or config.pretrained_model),
                        "remote_training_status": latest_trial.remote_training_status,
                        "created_at": latest_trial.created_at,
                    },
                }
            )
        return {"experiments": items}

    def get_experiment_detail(self, experiment_id: str) -> dict[str, Any]:
        config = self.repo.get_experiment(experiment_id)
        trials = self.repo.list_trials(experiment_id)
        experiment_payload = self._experiment_api_payload(config)
        return {
            "experiment": experiment_payload,
            "trial_count": len(trials),
            "latest_params": self._latest_params(config, trials),
            "default_model": config.pretrained_model,
            "default_export_dir": self._project_default_export_dir(config.project),
            "search_space": config.search_space,
            "trials": [self._trial_row(trial) for trial in trials],
            "latest_trial_created_at": trials[-1].created_at if trials else "",
        }

    def update_experiment(
        self,
        experiment_id: str,
        *,
        description: str | None = None,
        project: str | None = None,
        dataset_root: str | None = None,
        dataset_yaml: str | None = None,
        pretrained_model: str | None = None,
        remote_configs: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        config = self.repo.get_experiment(experiment_id)
        if description is None and project is None and dataset_root is None and dataset_yaml is None and pretrained_model is None and remote_configs is None:
            return {"experiment": self._experiment_api_payload(config)}

        normalized_description = None
        if description is not None:
            normalized_description = description.strip()
            if not normalized_description:
                raise ServiceError("description cannot be empty")

        normalized_project = None
        if project is not None:
            normalized_project = project.strip() or default_project_name(normalized_description or config.description)
        normalized_dataset_root = str(dataset_root).strip() if dataset_root is not None else None
        normalized_dataset_yaml = str(dataset_yaml).strip() if dataset_yaml is not None else None
        normalized_pretrained = str(pretrained_model).strip() if pretrained_model is not None else None
        if normalized_dataset_root is not None and not normalized_dataset_root:
            raise ServiceError("dataset root cannot be empty")
        if normalized_dataset_yaml is not None and not normalized_dataset_yaml:
            raise ServiceError("dataset YAML cannot be empty")
        if normalized_pretrained is not None and not normalized_pretrained:
            raise ServiceError("default model cannot be empty")

        self.repo.update_experiment(
            experiment_id,
            description=normalized_description,
            project=normalized_project,
            dataset_root=normalized_dataset_root,
            dataset_yaml=normalized_dataset_yaml,
            pretrained_model=normalized_pretrained,
            remote_configs=remote_configs,
        )
        if normalized_description is not None:
            config.description = normalized_description
        if normalized_project is not None:
            config.project = normalized_project
        if normalized_dataset_root is not None:
            config.dataset_root = normalized_dataset_root
        if normalized_dataset_yaml is not None:
            config.dataset_yaml = normalized_dataset_yaml
        if normalized_pretrained is not None:
            config.pretrained_model = normalized_pretrained
        if remote_configs is not None:
            config.remote_configs = remote_configs

        experiment_json = Path(config.save_root) / "experiments" / experiment_id / EXPERIMENT_FILENAME
        if experiment_json.exists():
            write_json(experiment_json, config.to_dict())

        self.repo.add_event(
            experiment_id,
            "EXPERIMENT_UPDATED",
            {"description": config.description, "project": config.project},
        )
        return {"experiment": self._experiment_api_payload(config)}

    def remote_run_config(self, experiment_id: str, remote_server_id: str) -> dict[str, Any]:
        config = self.repo.get_experiment(experiment_id)
        server = self.repo.get_remote_server(remote_server_id)
        return {
            "remote_server_id": remote_server_id,
            "dataset_root": config.remote_configs.get(remote_server_id, {}).get("dataset_root", ""),
            "dataset_yaml": config.remote_configs.get(remote_server_id, {}).get("dataset_yaml", ""),
            "pretrained_model": config.remote_configs.get(remote_server_id, {}).get("pretrained_model", ""),
            "remote_python": server.remote_python,
            "work_root": server.default_runs_root,
        }

    def launch_remote_trial(
        self,
        experiment_id: str,
        *,
        remote_server_id: str,
        params: dict[str, Any] | None = None,
        pretrained: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        config = self.repo.get_experiment(experiment_id)
        server = self.repo.get_remote_server(remote_server_id)
        remote_cfg = config.remote_configs.get(remote_server_id, {})
        dataset_root = str(remote_cfg.get("dataset_root", "")).strip()
        dataset_yaml = str(remote_cfg.get("dataset_yaml", "")).strip()
        remote_model = str(remote_cfg.get("pretrained_model", "")).strip() or str(pretrained or "").strip()
        remote_python = server.remote_python.strip()
        work_root = server.default_runs_root.strip()
        if not remote_python or not work_root:
            raise ServiceError("remote server requires remote_python and default_runs_root")
        if not dataset_root and not dataset_yaml:
            raise ServiceError("remote dataset path is not configured for this task")
        if not remote_model:
            raise ServiceError("remote model path is not configured for this task")
        validation = self.validate_params(experiment_id, params=params or config.initial_params)
        if not validation["valid"]:
            raise ServiceError(f"invalid trial params: {validation['errors']}")

        client, sftp = self._open_sftp(server)
        trial_id = self.repo.next_trial_id()
        display_name = self._next_trial_display_name(experiment_id, remote_model, validation["normalized_params"])
        remote_dir = posixpath.join(work_root.rstrip("/"), "experiments", experiment_id, trial_id)
        local_dir = ensure_dir(Path(config.save_root) / "experiments" / experiment_id / display_name)
        try:
            dataset_yaml = self._resolve_remote_dataset_yaml(client, dataset_root, dataset_yaml)
            self._exec_remote(client, f"test -x {shlex.quote(remote_python)}")
            self._exec_remote(client, f"test -r {shlex.quote(dataset_yaml)}")
            remote_model_path = remote_model
            model_check = self._exec_remote(client, f"test -r {shlex.quote(remote_model)}", check=False)
            amp_source = Path(__file__).resolve().parent / "models" / "yolo26n.pt"
            if not amp_source.exists():
                amp_source = Path(__file__).resolve().parent / "models" / "yolo26n.pt"
            self._mkdir_remote(sftp, remote_dir)
            request = {
                "dataset_yaml": dataset_yaml,
                "pretrained_model": remote_model_path,
                "run_dir": remote_dir,
                "params": validation["normalized_params"],
                "task_type": config.task_type,
            }
            request_path = local_dir / "request.json"
            write_json(request_path, request)
            self._upload_remote_file(sftp, request_path, self._remote_join(remote_dir, "request.json"))
            worker_path = Path(__file__).resolve().parent / "core" / "remote_train_worker.py"
            self._upload_remote_file(sftp, worker_path, self._remote_join(remote_dir, "remote_train_worker.py"))
            if amp_source.exists():
                self._upload_remote_file(sftp, amp_source, self._remote_join(remote_dir, "yolo26n.pt"))
            if model_check != 0:
                local_model = _resolve_pretrained_model(pretrained or config.pretrained_model)
                fallback_remote = self._remote_join(remote_dir, "pretrained_model" + Path(local_model).suffix)
                self._upload_remote_file(sftp, Path(local_model), fallback_remote)
                request["pretrained_model"] = fallback_remote
                write_json(request_path, request)
                self._upload_remote_file(sftp, request_path, self._remote_join(remote_dir, "request.json"))
            command = (
                f"nohup {shlex.quote(remote_python)} {shlex.quote(self._remote_join(remote_dir, 'remote_train_worker.py'))} "
                f"{shlex.quote(self._remote_join(remote_dir, 'request.json'))} > {shlex.quote(self._remote_join(remote_dir, 'stdout.log'))} "
                f"2> {shlex.quote(self._remote_join(remote_dir, 'stderr.log'))} < /dev/null & echo $! > {shlex.quote(self._remote_join(remote_dir, 'pid'))}"
            )
            self._exec_remote(client, command)
        except Exception:
            try:
                self._remove_remote_tree(sftp, remote_dir)
            except Exception:
                pass
            raise
        finally:
            sftp.close()
            client.close()

        trial = TrialRecord(
            trial_id=trial_id,
            display_name=display_name,
            experiment_id=experiment_id,
            iteration=self._next_iteration(self.repo.list_trials(experiment_id)),
            params=validation["normalized_params"],
            status=STATE_TRAINING,
            run_dir=str(local_dir),
            source=REMOTE_SOURCE,
            note=(note or "").strip(),
            model=remote_model,
            model_source="remote_config",
            params_source="manual",
            remote_server_id=remote_server_id,
            remote_run_dir=remote_dir,
            sync_status=REMOTE_SYNC_PENDING,
            remote_training_status=REMOTE_TRAINING_RUNNING,
        )
        self.repo.create_trial(trial)
        self.repo.update_experiment_status(experiment_id, STATE_TRAINING)
        self.repo.add_event(experiment_id, "REMOTE_TRIAL_STARTED", {"trial_id": trial_id, "remote_run_dir": remote_dir}, trial_id)
        return {"status": public_task_status(STATE_TRAINING, REMOTE_TRAINING_RUNNING), "internal_status": STATE_TRAINING, "trial_id": trial_id, "display_name": display_name, "remote_run_dir": remote_dir}

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
            "extra_param_schema": self._extra_param_schema(),
            "protected_extra_params": sorted(PLATFORM_CONTROLLED_YOLO_PARAMS),
        }

    @staticmethod
    def _hyperparameter_template_payload(template: HyperparameterTemplate) -> dict[str, Any]:
        return {
            "template_id": template.template_id,
            "name": template.name,
            "params": template.params,
            "source_trial_id": template.source_trial_id,
            "source_task_type": template.source_task_type,
            "created_at": template.created_at,
            "updated_at": template.updated_at,
        }

    def list_hyperparameter_templates(self) -> dict[str, Any]:
        return {
            "templates": [
                self._hyperparameter_template_payload(template)
                for template in self.repo.list_hyperparameter_templates()
            ]
        }

    def save_trial_hyperparameter_template(
        self,
        trial_id: str,
        name: str,
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        normalized_name = str(name or "").strip()
        if not normalized_name:
            raise ServiceError("template name is required")
        if len(normalized_name) > 80:
            raise ServiceError("template name must not exceed 80 characters")

        trial = self.repo.get_trial(trial_id)
        config = self.repo.get_experiment(trial.experiment_id)
        existing = self.repo.get_hyperparameter_template_by_name(normalized_name)
        if existing is not None and not overwrite:
            raise TemplateNameConflictError(f"hyperparameter template already exists: {existing.name}")

        if existing is not None:
            self.repo.overwrite_hyperparameter_template(
                existing.template_id,
                name=normalized_name,
                params=dict(trial.params),
                source_trial_id=trial.trial_id,
                source_task_type=config.task_type,
            )
            saved = self.repo.get_hyperparameter_template_by_name(normalized_name)
            return {"template": self._hyperparameter_template_payload(saved), "overwritten": True}

        template = HyperparameterTemplate(
            template_id=self.repo.next_hyperparameter_template_id(),
            name=normalized_name,
            params=dict(trial.params),
            source_trial_id=trial.trial_id,
            source_task_type=config.task_type,
        )
        self.repo.create_hyperparameter_template(template)
        saved = self.repo.get_hyperparameter_template_by_name(normalized_name)
        return {"template": self._hyperparameter_template_payload(saved), "overwritten": False}

    def delete_hyperparameter_template(self, template_id: str) -> dict[str, Any]:
        deleted = self.repo.delete_hyperparameter_template(template_id)
        if not deleted:
            raise ServiceError(f"hyperparameter template not found: {template_id}")
        return {"template_id": template_id, "deleted": True}

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
        has_extra_candidates = any(
            key not in SEARCH_SPACE and key not in PLATFORM_CONTROLLED_YOLO_PARAMS
            for key in candidate
        )
        extra_schema = self._extra_param_schema() if has_extra_candidates else {}
        for key in candidate:
            if key in PLATFORM_CONTROLLED_YOLO_PARAMS:
                errors[key] = "parameter is controlled by the platform"
            elif key not in SEARCH_SPACE and key not in extra_schema:
                errors[key] = "unsupported parameter"
        for key in SEARCH_SPACE:
            if key not in candidate:
                errors[key] = "missing required parameter"
                continue
            try:
                normalized[key] = validate_param_value(key, candidate[key])
            except ValueError as exc:
                errors[key] = str(exc)
        for key, value in candidate.items():
            if key not in extra_schema:
                continue
            try:
                normalized[key] = self._validate_extra_param_value(key, value, extra_schema[key]["type"])
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
        trials = self.repo.list_trials(experiment_id)
        active_states = {STATE_TRAINING, STATE_RETRAINING, STATE_ANALYZING}
        has_active_trial = any(trial.status in active_states for trial in trials)
        if config.status in {STATE_COMPLETED, STATE_CANCELLED} or (
            config.status == STATE_WAITING and not has_active_trial
        ):
            latest_trial = trials[-1] if trials else None
            return {
                "experiment_id": experiment_id,
                "status": public_task_status(
                    config.status,
                    latest_trial.remote_training_status if latest_trial else "",
                    latest_trial.sync_status if latest_trial else "",
                ),
                "internal_status": config.status,
                "message": "task already finalized",
            }
        normalized_reason = reason or "cancelled by user"
        process_terminated = cancel_training_process(experiment_id)
        remote_errors: list[str] = []
        for trial in trials:
            if trial.status in active_states:
                if trial.source == REMOTE_SOURCE and trial.remote_server_id:
                    try:
                        self.cancel_remote_trial(trial.trial_id)
                        process_terminated = True
                    except ServiceError as exc:
                        remote_errors.append(str(exc))
                self.repo.update_trial(trial.trial_id, status=STATE_CANCELLED)
        self.repo.update_experiment_status(experiment_id, STATE_CANCELLED)
        self.repo.add_event(
            experiment_id,
            "TASK_CANCELLED",
            {"reason": normalized_reason, "process_terminated": process_terminated},
        )
        return {
            "experiment_id": experiment_id,
            "status": public_task_status(STATE_CANCELLED),
            "internal_status": STATE_CANCELLED,
            "reason": normalized_reason,
            "process_terminated": process_terminated,
            "errors": remote_errors,
        }

    def cancel_remote_trial(self, trial_id: str) -> dict[str, Any]:
        trial = self.repo.get_trial(trial_id)
        if trial.source != REMOTE_SOURCE or not trial.remote_server_id:
            raise ServiceError("trial is not a remotely managed trial")
        server = self.repo.get_remote_server(trial.remote_server_id)
        client, sftp = self._open_sftp(server)
        try:
            pid_path = self._remote_join(trial.remote_run_dir, "pid")
            with sftp.open(pid_path, "r") as handle:
                raw_pid = handle.read()
            pid = int(raw_pid.decode() if isinstance(raw_pid, bytes) else str(raw_pid).strip())
            if pid <= 1:
                raise ValueError("invalid remote pid")
            self._exec_remote(client, f"kill -- -{pid} 2>/dev/null || kill {pid} 2>/dev/null || true", check=False)
        except (OSError, TypeError, ValueError) as exc:
            raise ServiceError(f"failed to cancel remote trial: {exc}") from exc
        finally:
            sftp.close()
            client.close()
        self.repo.update_trial(trial_id, remote_training_status=REMOTE_TRAINING_MAYBE_STOPPED, sync_error="cancelled by user")
        return {"trial_id": trial_id, "cancelled": True}

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

        self.repo.delete_training_tasks_for_experiment(experiment_id)
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
        if not force and any(
            task.parent_trial_id == trial_id
            for task in self.repo.list_training_tasks(("RUNNING", "QUEUED"))
        ):
            raise ServiceError("trial is the checkpoint source of an active continuation task")

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
                preview_dir = _validation_preview_root(trial.run_dir)
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
        workbench_result = self.workbench.clear_cache()
        deleted_dirs += int(workbench_result.get("deleted_dirs", 0))
        deleted_files += int(workbench_result.get("deleted_files", 0))
        deleted_bytes += int(workbench_result.get("deleted_bytes", 0))
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
        parent_trial_id: str = "",
        training_mode: str = "fresh",
        on_trial_started: Any | None = None,
    ) -> dict[str, Any]:
        config = self.repo.get_experiment(experiment_id)
        trials = self.repo.list_trials(experiment_id)
        iteration = self._next_iteration(trials)
        prepared = self.prepare_trial_request(
            experiment_id,
            params=params,
            pretrained=pretrained,
            note=note,
            reason=reason,
        )
        trial_params = prepared["params"]
        trial_model = prepared["pretrained"]
        dataset_analysis = analyze_dataset(config.dataset_yaml)
        parent_trial = self.repo.get_trial(parent_trial_id) if parent_trial_id else None
        if parent_trial and parent_trial.experiment_id != experiment_id:
            raise ServiceError("continuation parent must belong to the same experiment")
        display_name = (
            self._next_continuation_display_name(experiment_id, parent_trial.display_name)
            if parent_trial
            else self._next_trial_display_name(experiment_id, trial_model, trial_params)
        )
        trial_id = self.repo.next_trial_id()
        trial_dir = ensure_dir(Path(config.save_root) / "experiments" / experiment_id / display_name)
        status = STATE_TRAINING if iteration == 1 else STATE_RETRAINING
        trial = TrialRecord(
            trial_id=trial_id,
            display_name=display_name,
            experiment_id=experiment_id,
            iteration=iteration,
            params=trial_params,
            status=status,
            run_dir=str(trial_dir),
            source="trained",
            note=prepared["note"],
            reason=prepared["reason"],
            model=trial_model,
            model_source="continued_trial" if parent_trial else "manual" if pretrained else "experiment_default",
            params_source="continued_trial" if parent_trial else "manual",
            dataset_analysis=dataset_analysis,
            parent_trial_id=parent_trial_id,
            training_mode="continued" if parent_trial else training_mode,
        )
        write_json(trial_dir / TRIAL_CONFIG_FILENAME, trial_params)
        self.repo.create_trial(trial)
        if on_trial_started is not None:
            on_trial_started(trial_id)
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
                "parent_trial_id": parent_trial_id,
                "training_mode": trial.training_mode,
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
                task_type=config.task_type,
                python_executable=self._python_for_yolo(),
                src_root=str(Path(__file__).resolve().parent.parent),
                process_key=experiment_id,
            )
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
            next_status = STATE_COMPLETED
            self.repo.update_trial(
                trial_id,
                status=next_status,
                metrics=summary["final_metrics"],
                summary_path=str(summary_path),
            )
            self.repo.update_experiment_status(experiment_id, next_status)
            self.repo.add_event(experiment_id, "TRIAL_COMPLETED", summary, trial_id)
            return {
                "status": public_task_status(next_status),
                "internal_status": next_status,
                "trial_id": trial_id,
                "display_name": display_name,
                "run_dir": run_dir,
                "stdout_log": training_result["stdout_log"],
                "stderr_log": training_result["stderr_log"],
                "summary_path": str(summary_path),
                "final_metrics": summary["final_metrics"],
                "parent_trial_id": parent_trial_id,
                "training_mode": trial.training_mode,
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
                "status": public_task_status(STATE_CANCELLED),
                "internal_status": STATE_CANCELLED,
                "trial_id": trial_id,
                "display_name": display_name,
                "run_dir": str(trial_dir),
            }
        except TrainingError as exc:
            self.repo.update_trial(trial_id, status=STATE_FAILED)
            self.repo.update_experiment_status(experiment_id, STATE_FAILED)
            self.repo.add_event(experiment_id, "TRIAL_FAILED", {"trial_id": trial_id, "error": str(exc)}, trial_id)
            raise ServiceError(str(exc)) from exc

    def prepare_trial_request(
        self,
        experiment_id: str,
        params: dict[str, Any] | None = None,
        *,
        pretrained: str | None = None,
        note: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        config = self.repo.get_experiment(experiment_id)
        validation = self.validate_params(experiment_id, params=params or config.initial_params)
        if not validation["valid"]:
            raise ServiceError(f"invalid trial params: {validation['errors']}")
        trial_model = _resolve_pretrained_model(pretrained or config.pretrained_model)
        _validate_pretrained_model(trial_model)
        return {
            "params": validation["normalized_params"],
            "pretrained": trial_model,
            "note": (note or "").strip(),
            "reason": (reason or "").strip(),
        }

    def get_continuation_options(self, trial_id: str) -> dict[str, Any]:
        trial = self.repo.get_trial(trial_id)
        config = self.repo.get_experiment(trial.experiment_id)
        effective_params = build_initial_params(config.task_type, config.initial_params)
        effective_params.update(trial.params)
        completed_epochs = self._trial_completed_epochs(trial)
        parent_lr0 = float(effective_params["lr0"])
        lr_rule = SEARCH_SPACE["lr0"]
        recommended_lr0 = float(
            f"{max(float(lr_rule['min']), min(float(lr_rule['max']), parent_lr0 * 0.1)):.12g}"
        )
        unavailable_reason = ""
        checkpoint = ""
        has_active_continuation = any(
            task.parent_trial_id == trial_id
            for task in self.repo.list_training_tasks(("RUNNING", "QUEUED"))
        )
        if has_active_continuation:
            unavailable_reason = "当前 Trial 已有续训任务正在运行或排队"
        elif trial.remote_server_id or trial.source == REMOTE_SOURCE:
            unavailable_reason = "远程训练记录尚未同步 last.pt"
        elif trial.status in {STATE_TRAINING, STATE_RETRAINING, STATE_ANALYZING}:
            unavailable_reason = "当前 Trial 仍在训练中"
        else:
            try:
                checkpoint = str(_continuation_weight_path(trial.run_dir))
            except ServiceError:
                unavailable_reason = "训练目录中没有 last.pt"
        return {
            "trial_id": trial.trial_id,
            "display_name": trial.display_name,
            "experiment_id": trial.experiment_id,
            "can_continue": not unavailable_reason,
            "unavailable_reason": unavailable_reason,
            "checkpoint": checkpoint,
            "completed_epochs": completed_epochs,
            "cumulative_epochs": self._cumulative_epochs(trial),
            "defaults": {
                "additional_epochs": int(effective_params["epochs"]),
                "lr0": recommended_lr0,
                "original_lr0": parent_lr0,
                "patience": int(effective_params["patience"]),
            },
        }

    def prepare_continuation_request(
        self,
        trial_id: str,
        *,
        additional_epochs: int,
        lr0: float | None,
        patience: int | None,
        note: str | None,
    ) -> dict[str, Any]:
        trial = self.repo.get_trial(trial_id)
        options = self.get_continuation_options(trial_id)
        if not options["can_continue"]:
            raise ServiceError(str(options["unavailable_reason"]))
        try:
            normalized_epochs = validate_param_value("epochs", additional_epochs)
            normalized_lr0 = validate_param_value(
                "lr0",
                options["defaults"]["lr0"] if lr0 is None else lr0,
            )
            normalized_patience = validate_param_value(
                "patience",
                options["defaults"]["patience"] if patience is None else patience,
            )
        except ValueError as exc:
            raise ServiceError(str(exc)) from exc
        config = self.repo.get_experiment(trial.experiment_id)
        params = build_initial_params(config.task_type, config.initial_params)
        params.update(trial.params)
        params.update(
            {
                "epochs": normalized_epochs,
                "lr0": normalized_lr0,
                "patience": normalized_patience,
            }
        )
        validation = self.validate_params(trial.experiment_id, params=params)
        if not validation["valid"]:
            raise ServiceError(f"invalid continuation params: {validation['errors']}")
        return {
            "experiment_id": trial.experiment_id,
            "params": validation["normalized_params"],
            "pretrained": options["checkpoint"],
            "note": (note or "").strip(),
            "reason": f"Continue from {trial.display_name}",
        }

    def get_summary(self, trial_id: str, compact: bool = False) -> dict[str, Any]:
        trial = self.repo.get_trial(trial_id)
        config = self.repo.get_experiment(trial.experiment_id)
        if not trial.summary_path:
            summary: dict[str, Any] = {
                "trial_id": trial_id,
                "final_metrics": {},
                "per_class_metrics": [],
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
        continuation = self.get_continuation_options(trial_id)
        parent_display_name = ""
        if trial.parent_trial_id:
            try:
                parent_display_name = self.repo.get_trial(trial.parent_trial_id).display_name
            except KeyError:
                parent_display_name = trial.parent_trial_id
        if compact:
            return {
                "trial_id": trial_id,
                "display_name": trial.display_name,
                "status": public_task_status(trial.status, trial.remote_training_status, trial.sync_status),
                "internal_status": trial.status,
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
                "created_at": trial.created_at,
                "logs": logs,
                "metric_context": summary.get("metric_context", {}),
                "final_metrics": summary.get("final_metrics", {}),
                "per_class_metrics": summary.get("per_class_metrics", []),
                "metric_breakdown": summary.get("metric_breakdown", {}),
                "delta_vs_prev": summary.get("delta_vs_prev", {}),
                "metric_breakdown_delta_vs_prev": summary.get("metric_breakdown_delta_vs_prev", {}),
                "training_dynamics": summary.get("training_dynamics", {}),
                "warnings": summary.get("warnings", []),
                "dataset_analysis": trial.dataset_analysis,
                "parent_trial_id": trial.parent_trial_id,
                "parent_display_name": parent_display_name,
                "training_mode": trial.training_mode,
                "cumulative_epochs": continuation["cumulative_epochs"],
                "continuation": continuation,
            }
        summary["trial"] = {
            "trial_id": trial.trial_id,
            "display_name": trial.display_name,
            "experiment_id": trial.experiment_id,
            "project": config.project,
            "default_export_dir": self._project_default_export_dir(config.project),
            "task_type": config.task_type,
            "iteration": trial.iteration,
            "status": public_task_status(trial.status, trial.remote_training_status, trial.sync_status),
            "internal_status": trial.status,
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
            "created_at": trial.created_at,
            "logs": logs,
            "imgsz": int(summary.get("params", {}).get("imgsz") or trial.params.get("imgsz") or 0),
            "dataset_yaml": config.dataset_yaml,
            "parent_trial_id": trial.parent_trial_id,
            "parent_display_name": parent_display_name,
            "training_mode": trial.training_mode,
            "cumulative_epochs": continuation["cumulative_epochs"],
        }
        summary["dataset_analysis"] = trial.dataset_analysis
        summary["continuation"] = continuation
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
        output_filename = _export_filename(normalized_model_name)
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

        python_executable = self._python_for_yolo()
        env = dict(os.environ)
        src_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else f"{src_root}{os.pathsep}{env['PYTHONPATH']}"
        process = subprocess.run(
            [python_executable, "-m", "backend.core.export_worker", str(request_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
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

        preview_root = _validation_preview_root(trial.run_dir)
        if preview_root.exists():
            try:
                shutil.rmtree(preview_root, onerror=_handle_rmtree_error)
            except OSError as exc:
                raise ServiceError(f"failed to clear previous validation preview: {exc}") from exc
        validation_id = VALIDATION_CURRENT_DIRNAME
        output_dir = ensure_dir(preview_root / validation_id)
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

        python_executable = self._python_for_yolo()
        env = dict(os.environ)
        src_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else f"{src_root}{os.pathsep}{env['PYTHONPATH']}"
        completed = subprocess.run(
            [python_executable, "-m", "backend.core.validation_worker", str(request_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
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

        result = {
            "trial_id": trial_id,
            "display_name": trial.display_name,
            "validation_id": validation_id,
            "task_type": config.task_type,
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
        write_json(output_dir / VALIDATION_RESULT_FILENAME, result)
        return result

    def get_validation_preview(self, trial_id: str) -> dict[str, Any]:
        trial = self.repo.get_trial(trial_id)
        config = self.repo.get_experiment(trial.experiment_id)
        preview_root = _validation_preview_root(trial.run_dir)
        current_result = self._read_validation_preview_result(
            trial,
            config,
            preview_root / VALIDATION_CURRENT_DIRNAME,
        )
        if current_result is not None:
            return {"trial_id": trial_id, "result": current_result}

        if not preview_root.exists():
            return {"trial_id": trial_id, "result": None}

        legacy_dirs = sorted(
            (
                path for path in preview_root.iterdir()
                if path.is_dir() and path.name.startswith("val_")
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for preview_dir in legacy_dirs:
            result = self._read_validation_preview_result(trial, config, preview_dir)
            if result is not None:
                return {"trial_id": trial_id, "result": result}
        return {"trial_id": trial_id, "result": None}

    def _read_validation_preview_result(
        self,
        trial: TrialRecord,
        config: ExperimentConfig,
        preview_dir: Path,
    ) -> dict[str, Any] | None:
        preview_root = _validation_preview_root(trial.run_dir)
        resolved_dir = preview_dir.resolve()
        if not resolved_dir.is_relative_to(preview_root) or not resolved_dir.is_dir():
            return None
        try:
            validation_id = _validate_preview_filename(resolved_dir.name)
        except ServiceError:
            return None

        result_path = resolved_dir / VALIDATION_RESULT_FILENAME
        if result_path.exists():
            try:
                stored = read_json(result_path)
            except (OSError, ValueError):
                stored = None
            if isinstance(stored, dict):
                stored["trial_id"] = trial.trial_id
                stored["display_name"] = trial.display_name
                stored["validation_id"] = validation_id
                stored.setdefault("task_type", config.task_type)
                return stored

        request_path = resolved_dir / ".validation_request.json"
        stdout_path = resolved_dir / "stdout.log"
        try:
            request = read_json(request_path)
        except (OSError, ValueError):
            return None
        worker_result = _last_json_object(stdout_path)
        if not isinstance(request, dict) or worker_result is None:
            return None
        return {
            "trial_id": trial.trial_id,
            "display_name": trial.display_name,
            "validation_id": validation_id,
            "task_type": str(request.get("task_type") or config.task_type),
            "split": "val",
            "conf": request.get("conf"),
            "image_limit": request.get("image_limit"),
            "imgsz": request.get("imgsz"),
            "batch": request.get("batch"),
            "weight_path": request.get("model_path", ""),
            "metrics": worker_result.get("metrics", {}),
            "images": worker_result.get("images", []),
            "stdout_log": str(stdout_path),
            "stderr_log": str(resolved_dir / "stderr.log"),
        }

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
            raw_params = _params_from_args(args_data, set(self._extra_param_schema()))
            params_source = "args_yaml"
        elif raw_params is None and config_path.exists():
            raw_params = read_json(config_path)
            params_source = "config_json"
        if raw_params:
            base_params = self._latest_params(config, self.repo.list_trials(experiment_id))
            merged_params = dict(base_params)
            merged_params.update(raw_params)
            raw_params = merged_params
            if params_source == "args_yaml" and set(raw_params) != set(_params_from_args(args_data, set(self._extra_param_schema()))):
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
        next_status = STATE_COMPLETED
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
            "status": public_task_status(next_status),
            "internal_status": next_status,
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
        parsed_params = _params_from_args(args_data, set(self._extra_param_schema()))
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
            "status": public_task_status(trial.status, trial.remote_training_status),
            "internal_status": trial.status,
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
                "status": PUBLIC_STATUS_INTERRUPTED_OR_FAILED,
                "internal_status": STATE_TRAINING,
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
        csv_stat: Any = None
        remote_status_data: dict[str, Any] = {}
        try:
            try:
                self._download_remote_file(sftp, self._remote_join(trial.remote_run_dir, "status.json"), cache_dir / "status.json")
                remote_status_data = read_json(cache_dir / "status.json")
            except (OSError, ValueError, TypeError):
                remote_status_data = {}
            for filename in ("stdout.log", "stderr.log", "args.yaml"):
                try:
                    self._download_remote_file(sftp, self._remote_join(trial.remote_run_dir, filename), cache_dir / filename)
                except OSError:
                    pass
            remote_csv = self._remote_join(trial.remote_run_dir, "results.csv")
            try:
                csv_stat = sftp.stat(remote_csv)
                self._download_remote_file(sftp, remote_csv, cache_dir / "results.csv")
            except OSError:
                if remote_status_data.get("status") not in {"running", "completed"}:
                    raise
            self._download_top_level_pngs(sftp, trial.remote_run_dir, cache_dir)
            if remote_status_data.get("status") == "completed":
                for filename in ("weights/best.pt", "weights/last.pt"):
                    try:
                        self._download_remote_file(sftp, self._remote_join(trial.remote_run_dir, filename), cache_dir / filename)
                    except OSError:
                        pass
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

        args_data = _parse_args_yaml((cache_dir / "args.yaml").read_text(encoding="utf-8")) if (cache_dir / "args.yaml").exists() else dict(trial.params)
        epoch_count = _valid_epoch_count(cache_dir / "results.csv") if (cache_dir / "results.csv").exists() else 0
        remote_size = int(getattr(csv_stat, "st_size", 0) or 0)
        remote_mtime = float(getattr(csv_stat, "st_mtime", 0) or 0)
        unchanged = (
            trial.last_remote_csv_size == remote_size
            and trial.last_remote_csv_mtime == remote_mtime
        )
        unchanged_count = trial.unchanged_sync_count + 1 if unchanged else 0
        remote_status = str(remote_status_data.get("status") or "").strip().lower()
        # A managed worker writes status.json and keeps it at ``running`` for
        # the entire YOLO process lifetime. Treat that value as authoritative;
        # results.csv can contain a short/partial file (or args.yaml can report
        # an unexpected epoch value) while training is still in progress.
        if remote_status == "running":
            remote_training_status = REMOTE_TRAINING_RUNNING
        else:
            remote_training_status = self._remote_training_status(args_data, epoch_count, unchanged_count, remote_size)
        if remote_status == "completed":
            remote_training_status = REMOTE_TRAINING_COMPLETED
        elif remote_status == "failed":
            remote_training_status = REMOTE_TRAINING_MAYBE_STOPPED
            sync_error = str(remote_status_data.get("error") or "remote training failed")
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
                    STATE_COMPLETED
                    if remote_training_status == REMOTE_TRAINING_COMPLETED
                    else STATE_TRAINING
                    if remote_training_status == REMOTE_TRAINING_RUNNING
                    else STATE_WAITING
                )
            except Exception as exc:
                sync_error = f"summary parse failed: {exc}"
                next_status = STATE_TRAINING
        elif not sync_error and remote_training_status not in {REMOTE_TRAINING_RUNNING, REMOTE_TRAINING_UNKNOWN}:
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
            "status": public_task_status(
                next_status,
                remote_training_status,
                REMOTE_SYNC_FAILED if sync_error else REMOTE_SYNC_SYNCED,
            ),
            "internal_status": next_status,
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
        metric = "map50_95"
        best_row = None
        best_value = None
        for row in rows:
            value = row.get(metric)
            if isinstance(value, (int, float)) and (best_value is None or value > best_value):
                best_value = float(value)
                best_row = row
        for row in rows:
            row["is_best"] = bool(best_row and row["trial_id"] == best_row["trial_id"])
        columns = [
            {"key": "iteration", "label": "Iteration"},
            {"key": "trial_id", "label": "Trial"},
            {"key": "status", "label": "Status"},
            {"key": "model_display", "label": "Model"},
            {"key": "source", "label": "Source"},
            {"key": "server", "label": "Location"},
            {"key": "map50_95", "label": "mAP50-95"},
            {"key": "fitness", "label": "Fitness"},
            {"key": "map50", "label": "mAP50"},
            {"key": "precision", "label": "Precision"},
            {"key": "recall", "label": "Recall"},
            {"key": "delta_map50_95", "label": "Delta mAP50-95"},
            {"key": "best_epoch", "label": "Best Epoch"},
            {"key": "epochs_completed", "label": "Epochs"},
            {"key": "cumulative_epochs", "label": "Cumulative Epochs"},
            {"key": "training_mode", "label": "Training Mode"},
            {"key": "train_time_sec", "label": "Train Time"},
            {"key": "gpu_mem_peak", "label": "GPU Mem"},
            {"key": "params", "label": "Params"},
            {"key": "note", "label": "Note"},
        ]
        return {
            "experiment_id": experiment_id,
            "best_trial": None
            if best_row is None
            else {
                "trial_id": best_row["trial_id"],
                "iteration": best_row["iteration"],
                "metric": metric,
                "value": best_value,
            },
            "fitness_metric": fitness_metric(config.task_type),
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

    def _next_continuation_display_name(self, experiment_id: str, parent_display_name: str) -> str:
        prefix = f"{parent_display_name}_cont"
        index = 1
        while True:
            candidate = f"{prefix}_{index}"
            if not self.repo.trial_display_name_exists(experiment_id, candidate):
                return candidate
            index += 1

    def _trial_completed_epochs(self, trial: TrialRecord) -> int:
        if trial.summary_path and Path(trial.summary_path).exists():
            try:
                value = read_json(trial.summary_path).get("basic_info", {}).get("epochs_completed")
                if isinstance(value, (int, float)):
                    return max(0, int(value))
            except (OSError, ValueError, TypeError):
                pass
        results_csv = Path(trial.run_dir) / "results.csv"
        if results_csv.exists():
            try:
                with results_csv.open("r", encoding="utf-8", errors="replace") as handle:
                    return max(0, sum(1 for line in handle if line.strip()) - 1)
            except OSError:
                pass
        return 0

    def _cumulative_epochs(self, trial: TrialRecord, seen: set[str] | None = None) -> int:
        visited = set(seen or ())
        if trial.trial_id in visited:
            return self._trial_completed_epochs(trial)
        visited.add(trial.trial_id)
        total = self._trial_completed_epochs(trial)
        if trial.parent_trial_id:
            try:
                total += self._cumulative_epochs(self.repo.get_trial(trial.parent_trial_id), visited)
            except KeyError:
                pass
        return total

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

    def _exec_remote(self, client: Any, command: str, *, check: bool = True) -> str:
        _stdin, stdout, stderr = client.exec_command(command)
        exit_status = stdout.channel.recv_exit_status()
        output = stdout.read()
        error = stderr.read()
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        if isinstance(error, bytes):
            error = error.decode("utf-8", errors="replace")
        if check and exit_status != 0:
            raise ServiceError(str(error or output or f"remote command failed with code {exit_status}").strip())
        return str(output or "")

    def _resolve_remote_dataset_yaml(self, client: Any, dataset_root: str, dataset_yaml: str) -> str:
        if dataset_yaml:
            return dataset_yaml
        command = (
            "find " + shlex.quote(dataset_root) + " -maxdepth 2 -type f "
            "\\( -name data.yaml -o -name dataset.yaml -o -name detect.yaml \\) -print"
        )
        candidates = [line.strip() for line in self._exec_remote(client, command).splitlines() if line.strip()]
        if len(candidates) != 1:
            raise ServiceError(f"remote dataset YAML candidates: {len(candidates)}")
        return candidates[0]

    def _mkdir_remote(self, sftp: Any, remote_dir: str) -> None:
        parts = [part for part in remote_dir.split("/") if part]
        current = "/" if remote_dir.startswith("/") else ""
        for part in parts:
            current = posixpath.join(current, part)
            try:
                sftp.stat(current)
            except OSError:
                sftp.mkdir(current)

    def _upload_remote_file(self, sftp: Any, local_path: Path, remote_path: str) -> None:
        sftp.put(str(local_path), remote_path)

    def _remove_remote_tree(self, sftp: Any, remote_dir: str) -> None:
        try:
            entries = sftp.listdir_attr(remote_dir)
        except OSError:
            return
        for entry in entries:
            path = self._remote_join(remote_dir, getattr(entry, "filename", ""))
            mode = int(getattr(entry, "st_mode", 0) or 0)
            if stat.S_ISDIR(mode):
                self._remove_remote_tree(sftp, path)
            else:
                try:
                    sftp.remove(path)
                except OSError:
                    pass
        try:
            sftp.rmdir(remote_dir)
        except OSError:
            pass

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
            password = server.password or os.environ.get(server.password_ref)
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
        metric_context = summary.get("metric_context", {})
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
            "status": public_task_status(trial.status, trial.remote_training_status, trial.sync_status),
            "internal_status": trial.status,
            "source": trial.source,
            "model": trial.model,
            "model_display": _model_basename(trial.model),
            "server": server_name,
            "remote_server_id": trial.remote_server_id,
            "precision": final_metrics.get("precision"),
            "recall": final_metrics.get("recall"),
            "map50": final_metrics.get("map50"),
            "map50_95": final_metrics.get("map50_95"),
            "fitness": metric_context.get("selection_fitness"),
            "fitness_metric": metric_context.get("selection_metric"),
            "delta_map50_95": delta.get("map50_95"),
            "delta_recall": delta.get("recall"),
            "best_epoch": basic.get("best_epoch"),
            "epochs_completed": basic.get("epochs_completed"),
            "cumulative_epochs": self._cumulative_epochs(trial),
            "train_time_sec": basic.get("train_time_sec"),
            "gpu_mem_peak": resource.get("gpu_mem_peak"),
            "params": params,
            "run_dir": trial.run_dir,
            "summary_path": trial.summary_path,
            "note": trial.note,
            "reason": trial.reason,
            "parent_trial_id": trial.parent_trial_id,
            "training_mode": trial.training_mode,
            "remote_training_status": trial.remote_training_status,
            "last_synced_at": trial.last_synced_at,
            "created_at": trial.created_at,
            "logs": self._trial_logs(trial.run_dir),
            "is_best": False,
        }

    def get_experiment_curves(self, experiment_id: str) -> dict[str, Any]:
        config = self.repo.get_experiment(experiment_id)
        trials = self.repo.list_trials(experiment_id)
        curves = {}
        trial_labels = {}
        for trial in trials:
            results_csv = Path(trial.run_dir) / "results.csv"
            if not results_csv.exists():
                continue
            
            import csv
            trial_data = []
            epoch_offset = 0
            if trial.parent_trial_id:
                try:
                    epoch_offset = self._cumulative_epochs(self.repo.get_trial(trial.parent_trial_id))
                except KeyError:
                    epoch_offset = 0
            with open(results_csv, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cleaned_row = {}
                    for k, v in row.items():
                        if not k or v is None:
                            continue
                        v_str = str(v).strip()
                        if not v_str:
                            continue
                        try:
                            value = float(v_str)
                        except ValueError:
                            continue
                        if math.isfinite(value):
                            cleaned_row[str(k).strip()] = int(value) if value.is_integer() else value
                    if "epoch" in cleaned_row:
                        cleaned_row["epoch"] = int(cleaned_row["epoch"]) + epoch_offset
                        fitness_value = calculate_fitness(cleaned_row, config.task_type)
                        if fitness_value is None:
                            cleaned_row.pop("fitness", None)
                        else:
                            cleaned_row["fitness"] = round(fitness_value, 6)
                        trial_data.append(cleaned_row)
            curves[trial.trial_id] = trial_data
            trial_labels[trial.trial_id] = trial.display_name

        return {
            "experiment_id": experiment_id,
            "curves": curves,
            "trial_labels": trial_labels,
            "fitness_metric": fitness_metric(config.task_type),
        }

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
        preview_dir = (_validation_preview_root(trial.run_dir) / normalized_validation_id).resolve()
        if not preview_dir.is_relative_to(run_dir_resolved):
            raise ServiceError("invalid validation id")
        file_path = (preview_dir / normalized_filename).resolve()
        if not file_path.is_relative_to(preview_dir):
            raise ServiceError("invalid filename")
        if not file_path.exists() or not file_path.is_file():
            raise ServiceError("file not found")
        return str(file_path)
