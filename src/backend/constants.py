from __future__ import annotations

from typing import Any

YOLO_BASELINE: dict[str, Any] = {
    "imgsz": 640,
    "batch": 16,
    "workers": 2,
    "epochs": 100,
    "patience": 10,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.1,
    "momentum": 0.937,
    "warmup_epochs": 3.0,
    "cos_lr": True,
    "weight_decay": 0.0005,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
    "degrees": 2.0,
    "translate": 0.03,
    "scale": 0.10,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "hsv_h": 0.0,
    "hsv_s": 0.10,
    "hsv_v": 0.15,
}

TASK_BASELINES: dict[str, dict[str, Any]] = {
    "detection": dict(YOLO_BASELINE),
    "segment": dict(YOLO_BASELINE),
    "obb": dict(YOLO_BASELINE),
}

SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "imgsz": {"type": "int", "min": 160, "max": 2560, "step": 32},
    "batch": {"type": "choice", "values": [2, 4, 8, 16, 32]},
    "workers": {"type": "choice", "values": [0, 1, 2, 4]},
    "epochs": {"type": "int", "min": 1, "max": 1000},
    "patience": {"type": "int", "min": 0, "max": 500},
    "optimizer": {"type": "choice", "values": ["SGD", "Adam", "AdamW", "auto"]},
    "lr0": {"type": "float", "min": 0.00001, "max": 0.1},
    "lrf": {"type": "float", "min": 0.001, "max": 1.0},
    "momentum": {"type": "float", "min": 0.0, "max": 0.999},
    "warmup_epochs": {"type": "float", "min": 0.0, "max": 20.0},
    "cos_lr": {"type": "choice", "values": [True, False]},
    "weight_decay": {"type": "float", "min": 0.0, "max": 0.01},
    "mosaic": {"type": "float", "min": 0.0, "max": 1.0},
    "mixup": {"type": "float", "min": 0.0, "max": 1.0},
    "copy_paste": {"type": "float", "min": 0.0, "max": 1.0},
    "erasing": {"type": "float", "min": 0.0, "max": 1.0},
    "degrees": {"type": "float", "min": 0.0, "max": 45.0},
    "translate": {"type": "float", "min": 0.0, "max": 0.5},
    "scale": {"type": "float", "min": 0.0, "max": 1.0},
    "shear": {"type": "float", "min": 0.0, "max": 10.0},
    "perspective": {"type": "float", "min": 0.0, "max": 0.001},
    "flipud": {"type": "float", "min": 0.0, "max": 1.0},
    "fliplr": {"type": "float", "min": 0.0, "max": 1.0},
    "hsv_h": {"type": "float", "min": 0.0, "max": 0.1},
    "hsv_s": {"type": "float", "min": 0.0, "max": 1.0},
    "hsv_v": {"type": "float", "min": 0.0, "max": 1.0},
}

STOP_CONDITIONS = {
    "max_trials": 8,
    "max_no_improve": 3,
    "min_delta": 0.003,
}

ALLOWED_TASK_TYPES = tuple(TASK_BASELINES.keys())

PER_CLASS_METRICS_FILENAME = "per_class_metrics.json"

STATE_INIT = "INIT"
STATE_READY = "READY"
STATE_QUEUED = "QUEUED"
STATE_TRAINING = "TRAINING"
STATE_ANALYZING = "ANALYZING"
STATE_WAITING = "WAITING_USER_CONFIRM"
STATE_RETRAINING = "RETRAINING"
STATE_COMPLETED = "COMPLETED"
STATE_FAILED = "FAILED"
STATE_CANCELLED = "CANCELLED"

PUBLIC_STATUS_NOT_STARTED = "NOT_STARTED"
PUBLIC_STATUS_QUEUED = "QUEUED"
PUBLIC_STATUS_TRAINING = "TRAINING"
PUBLIC_STATUS_COMPLETED = "COMPLETED"
PUBLIC_STATUS_INTERRUPTED_OR_FAILED = "INTERRUPTED_OR_FAILED"

MAX_PARALLEL_TRAINING_SETTING_KEY = "max_parallel_training_tasks"
DEFAULT_MAX_PARALLEL_TRAINING_TASKS = 1
MAX_PARALLEL_TRAINING_TASKS_LIMIT = 64

SUMMARY_FILENAME = "summary.json"
EXPERIMENT_FILENAME = "experiment.json"
TRIAL_CONFIG_FILENAME = "config.json"
