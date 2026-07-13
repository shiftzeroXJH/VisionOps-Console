from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GoalConfig:
    metric: str
    target: float


@dataclass
class ExperimentConfig:
    experiment_id: str
    description: str
    project: str
    task_type: str
    dataset_root: str
    dataset_yaml: str
    pretrained_model: str
    save_root: str
    goal: GoalConfig
    status: str
    initial_params: dict[str, Any]
    search_space: dict[str, list[Any]]
    stop_conditions: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["goal"] = asdict(self.goal)
        return data


@dataclass
class TrialRecord:
    trial_id: str
    display_name: str
    experiment_id: str
    iteration: int
    params: dict[str, Any]
    status: str
    run_dir: str
    summary_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    source: str = "trained"
    note: str = ""
    reason: str = ""
    model: str = ""
    model_source: str = "experiment_default"
    params_source: str = "manual"
    remote_server_id: str = ""
    remote_run_dir: str = ""
    sync_status: str = ""
    sync_error: str = ""
    remote_training_status: str = ""
    last_remote_csv_size: int | None = None
    last_remote_csv_mtime: float | None = None
    last_synced_epoch_count: int = 0
    unchanged_sync_count: int = 0
    last_synced_at: str = ""
    dataset_analysis: dict[str, Any] = field(default_factory=dict)


@dataclass
class RemoteServer:
    remote_server_id: str
    name: str
    host: str
    port: int
    username: str
    auth_type: str
    private_key_path: str = ""
    password_ref: str = ""
    default_runs_root: str = ""


@dataclass
class Summary:
    trial_id: str
    basic_info: dict[str, Any]
    metric_context: dict[str, Any]
    final_metrics: dict[str, Any]
    per_class_metrics: list[dict[str, Any]]
    metric_breakdown: dict[str, Any]
    delta_vs_prev: dict[str, Any]
    metric_breakdown_delta_vs_prev: dict[str, Any]
    training_dynamics: dict[str, Any]
    warnings: list[str]
    resource: dict[str, Any]
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
