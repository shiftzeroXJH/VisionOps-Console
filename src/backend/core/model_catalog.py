"""Official model selections and process-safe weight resolution (also used remotely)."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path

FAMILIES = {"v8": "yolov8", "11": "yolo11", "26": "yolo26"}
TASKS = {"detection": ("目标检测", ""), "segment": ("实例分割", "-seg"),
         "obb": ("旋转框", "-obb"), "semantic": ("语义分割", "-sem")}
SCALES = tuple("nsmlx")


def model_catalog() -> dict:
    return {"tasks": [{"value": task, "label": label} for task, (label, _) in TASKS.items()],
            "models": [{"task_type": task, "model_family": family, "model_scale": scale,
                        "filename": f"{prefix}{scale}{suffix}.pt"}
                       for task, (_, suffix) in TASKS.items() for family, prefix in FAMILIES.items()
                       if task != "semantic" or family == "26" for scale in SCALES]}


OFFICIAL_NAMES = frozenset(row["filename"] for row in model_catalog()["models"])


def select_model(task: str, family: str, scale: str) -> str:
    for row in model_catalog()["models"]:
        if (row["task_type"], row["model_family"], row["model_scale"]) == (task, family, scale):
            return row["filename"]
    raise ValueError(f"Unsupported model selection: {task}/{family}/{scale}")


@contextmanager
def weight_lock(path: Path):
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        if os.name == "nt":
            import msvcrt
            while True:
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def resolve_training_model(value: str, models_dir: str | Path) -> str:
    if value not in OFFICIAL_NAMES:
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Custom model not found: {value}")
        return str(path.resolve())
    root = Path(models_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / value
    with weight_lock(root / (value + ".lock")):
        if not target.is_file():
            from ultralytics.utils.downloads import attempt_download_asset
            staging = root / (value + ".download")
            staging.mkdir(exist_ok=True)
            (staging / value).unlink(missing_ok=True)
            downloaded = Path(attempt_download_asset(str(staging / value)))
            if not downloaded.is_file() or downloaded.stat().st_size < 1024:
                raise RuntimeError(f"Model download failed: {value}")
            os.replace(downloaded, target)
            staging.rmdir()
        if target.stat().st_size < 1024:
            raise RuntimeError(f"Model file is truncated: {target}")
    return str(target)


def configure_amp_weights(run_dir: str | Path, models_dir: str | Path) -> None:
    import ultralytics.utils as utils

    for directory in (Path(run_dir).resolve(), Path(models_dir).resolve()):
        weight = directory / "yolo26n.pt"
        if weight.is_file() and weight.stat().st_size >= 1024:
            # check_amp imports this value at call time. Keep the override worker-local;
            # updating persistent SETTINGS would affect unrelated concurrent training.
            utils.WEIGHTS_DIR = directory
            return


def check_semantic_runtime(task: str) -> None:
    if task not in {"sem", "semantic"}:
        return
    try:
        from ultralytics.models.yolo.semantic import SemanticSegmentationTrainer
    except ImportError as exc:
        raise RuntimeError("Semantic segmentation requires Ultralytics >= 8.4.155 in this Python environment") from exc


def check_model_task(model, task: str, params: dict) -> None:
    expected = {"detection": "detect", "sem": "semantic"}.get(task, task)
    if expected == "semantic" and "semantic" not in model.task_map:
        raise RuntimeError("Semantic segmentation requires Ultralytics >= 8.4.155 in the training environment")
    if model.task != expected:
        raise ValueError(f"Model task {model.task} does not match experiment task {expected}")
    if expected == "semantic" and params.get("single_cls"):
        raise ValueError("single_cls=True is not supported for semantic segmentation")
