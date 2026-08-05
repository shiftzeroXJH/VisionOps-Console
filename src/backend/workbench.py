from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable
from uuid import uuid4

from PIL import Image

from backend.utils import ensure_dir, read_json, write_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODEL_EXTENSIONS = {".pt", ".onnx"}
WORKBENCH_TASK_TYPES = {"detection", "segment", "obb"}
CACHE_TTL = timedelta(hours=24)


class WorkbenchError(RuntimeError):
    pass


def _normalize_roi(raw: Any, image_width: int, image_height: int) -> dict[str, float] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise WorkbenchError("ROI must be an object")
    try:
        cx = float(raw["cx"])
        cy = float(raw["cy"])
        width = float(raw["width"])
        height = float(raw["height"])
        angle = float(raw.get("angle", 0))
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkbenchError("ROI requires numeric cx, cy, width, height and angle") from exc
    if not all(math.isfinite(value) for value in (cx, cy, width, height, angle)):
        raise WorkbenchError("ROI values must be finite")
    if not (0 <= cx <= image_width and 0 <= cy <= image_height):
        raise WorkbenchError("ROI center must be inside the image")
    max_extent = math.hypot(image_width, image_height)
    if width < 2 or height < 2 or width > max_extent or height > max_extent:
        raise WorkbenchError("ROI size is outside the image bounds")
    if angle < -45 or angle > 45:
        normalized_angle = (angle + 45) % 90 - 45
        quarter_turns = int(round((angle - normalized_angle) / 90))
        if quarter_turns % 2:
            width, height = height, width
        angle = normalized_angle
    return {"cx": cx, "cy": cy, "width": width, "height": height, "angle": angle}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _checkpoint_paths(run_dir: str) -> list[dict[str, Any]]:
    base = Path(run_dir)
    candidates: dict[str, Path] = {}
    for directory in (base / "weights", base):
        if not directory.is_dir():
            continue
        for path in directory.glob("*.pt"):
            name = path.name.lower()
            if name in {"best.pt", "last.pt"} or re.fullmatch(r"epoch\d+\.pt", name):
                candidates.setdefault(name, path.resolve())

    def order(item: tuple[str, Path]) -> tuple[int, int]:
        name = item[0]
        if name == "best.pt":
            return (0, 0)
        if name == "last.pt":
            return (1, 0)
        match = re.fullmatch(r"epoch(\d+)\.pt", name)
        return (2, -int(match.group(1)) if match else 0)

    checkpoints: list[dict[str, Any]] = []
    for name, path in sorted(candidates.items(), key=order):
        match = re.fullmatch(r"epoch(\d+)\.pt", name)
        checkpoints.append(
            {
                "name": name,
                "label": name if match is None else f"Epoch {match.group(1)}",
                "epoch": int(match.group(1)) if match else None,
                "path": path,
            }
        )
    return checkpoints


class WorkbenchService:
    def __init__(self, repo: Any, python_getter: Callable[[], str], cache_root: str | Path | None = None) -> None:
        self.repo = repo
        self.python_getter = python_getter
        self.cache_root = Path(cache_root or os.environ.get("YOLO_WORKBENCH_CACHE", ".workbench_cache")).resolve()
        ensure_dir(self.cache_root / "sessions")
        ensure_dir(self.cache_root / "evaluations")
        self.cleanup_expired()

    def list_models(self) -> dict[str, Any]:
        self.cleanup_expired()
        models: list[dict[str, Any]] = []
        for experiment in self.repo.list_experiments():
            if experiment.task_type not in WORKBENCH_TASK_TYPES:
                continue
            trials = sorted(
                self.repo.list_trials(experiment.experiment_id),
                key=lambda item: (str(item.created_at or ""), int(item.iteration), item.trial_id),
                reverse=True,
            )
            for trial in trials:
                checkpoints = _checkpoint_paths(trial.run_dir)
                if not checkpoints:
                    continue
                models.append(
                    {
                        "trial_id": trial.trial_id,
                        "trial_name": trial.display_name,
                        "experiment_id": experiment.experiment_id,
                        "experiment_name": experiment.description,
                        "project": experiment.project,
                        "task_type": experiment.task_type,
                        "created_at": trial.created_at,
                        "path": str(checkpoints[0]["path"]),
                        "default_checkpoint": checkpoints[0]["name"],
                        "checkpoints": [
                            {**item, "path": str(item["path"])}
                            for item in checkpoints
                        ],
                    }
                )
        return {"models": models, "effective_yolo_python": self.python_getter()}

    def create_session(self) -> dict[str, Any]:
        self.cleanup_expired()
        session_id = f"session_{uuid4().hex}"
        session_dir = ensure_dir(self.cache_root / "sessions" / session_id)
        ensure_dir(session_dir / "images")
        manifest = {
            "session_id": session_id,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "images": [],
            "model": None,
            "task_type": None,
            "classes": [],
        }
        write_json(session_dir / "manifest.json", manifest)
        return manifest

    def add_images(self, session_id: str, uploads: list[tuple[str, BinaryIO]]) -> dict[str, Any]:
        manifest, session_dir = self._session(session_id)
        added: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []
        for original_name, stream in uploads:
            suffix = Path(original_name).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                rejected.append({"name": original_name, "error": "unsupported image type"})
                continue
            image_id = f"img_{uuid4().hex}"
            target = session_dir / "images" / f"{image_id}{suffix}"
            with target.open("wb") as handle:
                shutil.copyfileobj(stream, handle, length=1024 * 1024)
            try:
                with Image.open(target) as image:
                    image.verify()
                with Image.open(target) as image:
                    width, height = image.size
            except Exception as exc:
                target.unlink(missing_ok=True)
                rejected.append({"name": original_name, "error": f"invalid image: {exc}"})
                continue
            record = {
                "image_id": image_id,
                "name": Path(original_name).name,
                "filename": target.name,
                "width": width,
                "height": height,
                "status": "pending",
                "detections": [],
                "error": "",
                "roi": None,
                "rotation": 0,
                "revision": 0,
            }
            manifest["images"].append(record)
            added.append(record)
        manifest["updated_at"] = _utc_now()
        write_json(session_dir / "manifest.json", manifest)
        if not added and rejected:
            raise WorkbenchError("; ".join(f"{item['name']}: {item['error']}" for item in rejected[:8]))
        return {"session_id": session_id, "images": added, "rejected": rejected}

    def get_session(self, session_id: str) -> dict[str, Any]:
        manifest, _ = self._session(session_id)
        return manifest

    def delete_session(self, session_id: str) -> dict[str, Any]:
        _, session_dir = self._session(session_id)
        shutil.rmtree(session_dir)
        return {"session_id": session_id, "deleted": True}

    def delete_images(self, session_id: str, image_ids: list[str]) -> dict[str, Any]:
        manifest, session_dir = self._session(session_id)
        requested = {str(image_id) for image_id in image_ids if str(image_id)}
        if not requested:
            raise WorkbenchError("image_ids is required")
        existing = {item["image_id"]: item for item in manifest["images"]}
        unknown = requested - existing.keys()
        if unknown:
            raise WorkbenchError("one or more images were not found")
        for image_id in requested:
            (session_dir / "images" / existing[image_id]["filename"]).unlink(missing_ok=True)
        manifest["images"] = [item for item in manifest["images"] if item["image_id"] not in requested]
        manifest["updated_at"] = _utc_now()
        write_json(session_dir / "manifest.json", manifest)
        return {**manifest, "deleted_image_ids": sorted(requested)}

    def set_image_roi(self, session_id: str, image_id: str, raw_roi: Any) -> dict[str, Any]:
        manifest, session_dir = self._session(session_id)
        item = next((record for record in manifest["images"] if record["image_id"] == image_id), None)
        if item is None:
            raise WorkbenchError("image not found")
        item["roi"] = _normalize_roi(raw_roi, int(item["width"]), int(item["height"]))
        item["detections"] = []
        item["status"] = "pending"
        item["error"] = ""
        manifest["updated_at"] = _utc_now()
        write_json(session_dir / "manifest.json", manifest)
        return manifest

    def rotate_image(self, session_id: str, image_id: str, direction: str) -> dict[str, Any]:
        manifest, session_dir = self._session(session_id)
        item = next((record for record in manifest["images"] if record["image_id"] == image_id), None)
        if item is None:
            raise WorkbenchError("image not found")
        if direction not in {"clockwise", "counterclockwise"}:
            raise WorkbenchError("direction must be clockwise or counterclockwise")
        path = session_dir / "images" / item["filename"]
        temporary = path.with_name(f"{path.stem}.rotating{path.suffix}")
        try:
            with Image.open(path) as source:
                source_format = source.format
                transpose = Image.Transpose.ROTATE_270 if direction == "clockwise" else Image.Transpose.ROTATE_90
                rotated = source.transpose(transpose)
                rotated.save(temporary, format=source_format)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        with Image.open(path) as rotated:
            item["width"], item["height"] = rotated.size
        delta = 90 if direction == "clockwise" else -90
        item["rotation"] = (int(item.get("rotation", 0)) + delta) % 360
        item["revision"] = int(item.get("revision", 0)) + 1
        item["roi"] = None
        item["detections"] = []
        item["status"] = "pending"
        item["error"] = ""
        manifest["updated_at"] = _utc_now()
        write_json(session_dir / "manifest.json", manifest)
        return manifest

    def infer(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        manifest, session_dir = self._session(session_id)
        model_path = self._resolve_model(payload)
        conf = self._number(payload.get("conf", 0.25), "conf", 0.001, 1.0)
        raw_imgsz = payload.get("imgsz")
        imgsz = None if raw_imgsz is None else self._integer(raw_imgsz, "imgsz", 32, 4096)
        requested = payload.get("image_ids")
        image_ids = set(requested or [item["image_id"] for item in manifest["images"]])
        selected = [item for item in manifest["images"] if item["image_id"] in image_ids]
        if not selected:
            raise WorkbenchError("no images selected")
        roi_overrides = payload.get("rois") or {}
        if not isinstance(roi_overrides, dict):
            raise WorkbenchError("rois must be an object")
        for item in selected:
            if item["image_id"] in roi_overrides:
                item["roi"] = _normalize_roi(roi_overrides[item["image_id"]], int(item["width"]), int(item["height"]))
            item["status"] = "running"
            item["error"] = ""
        manifest["updated_at"] = _utc_now()
        write_json(session_dir / "manifest.json", manifest)

        request = {
            "mode": "infer",
            "model_path": str(model_path),
            "task_type": payload.get("task_type"),
            "conf": conf,
            "imgsz": imgsz,
            "images": [
                {
                    "image_id": item["image_id"],
                    "path": str(session_dir / "images" / item["filename"]),
                    "roi": item.get("roi"),
                }
                for item in selected
            ],
        }
        result = self._run_worker(request, session_dir / "inference_request.json")
        by_id = {item["image_id"]: item for item in result.get("images", [])}
        for item in selected:
            worker_item = by_id.get(item["image_id"], {})
            item["status"] = worker_item.get("status", "failed")
            item["detections"] = worker_item.get("detections", [])
            item["error"] = worker_item.get("error", "inference returned no result")
        manifest["model"] = {"path": str(model_path), "source": payload.get("model_source", "local")}
        manifest["task_type"] = result.get("task_type") or payload.get("task_type")
        manifest["classes"] = result.get("classes", [])
        manifest["conf"] = conf
        manifest["imgsz"] = imgsz
        manifest["updated_at"] = _utc_now()
        write_json(session_dir / "manifest.json", manifest)
        return manifest

    def inspect_dataset(self, dataset_path: str) -> dict[str, Any]:
        path = self._absolute_existing_path(dataset_path, "dataset")
        request = {"mode": "inspect_dataset", "dataset_path": str(path)}
        request_path = self.cache_root / f"dataset_inspect_{uuid4().hex}.json"
        try:
            return self._run_worker(request, request_path)
        finally:
            request_path.unlink(missing_ok=True)

    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.cleanup_expired()
        model_path = self._resolve_model(payload)
        dataset_path = self._absolute_existing_path(str(payload.get("dataset_path", "")), "dataset")
        display_conf = self._number(payload.get("conf", 0.25), "conf", 0.001, 1.0)
        imgsz = self._integer(payload.get("imgsz", 640), "imgsz", 32, 4096)
        batch = self._integer(payload.get("batch", 8), "batch", 1, 256)
        evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
        evaluation_dir = ensure_dir(self.cache_root / "evaluations" / evaluation_id)
        request = {
            "mode": "evaluate",
            "evaluation_id": evaluation_id,
            "model_path": str(model_path),
            "task_type": payload.get("task_type"),
            "dataset_path": str(dataset_path),
            "display_conf": display_conf,
            "imgsz": imgsz,
            "batch": batch,
        }
        result = self._run_worker(request, evaluation_dir / "evaluation_request.json")
        result.update(
            {
                "evaluation_id": evaluation_id,
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "model_path": str(model_path),
                "model_source": str(payload.get("model_source", "local")),
                "trial_id": str(payload.get("trial_id", "") or ""),
                "checkpoint_name": str(payload.get("checkpoint_name", "") or ""),
                "dataset_path": str(dataset_path),
                "conf": display_conf,
                "imgsz": imgsz,
                "batch": batch,
            }
        )
        write_json(evaluation_dir / "manifest.json", result)
        predictions_dir = Path(str(result.get("predictions_dir", ""))).resolve()
        dataset_root = Path(str(result.get("dataset", {}).get("dataset_root", ""))).resolve()
        expected_dir = (dataset_root / "predictions_xml" / evaluation_id).resolve()
        if predictions_dir != expected_dir or not predictions_dir.is_dir():
            raise WorkbenchError("evaluation returned an invalid predictions directory")
        write_json(predictions_dir / "manifest.json", result)
        return result

    def list_evaluations(self, dataset_path: str) -> dict[str, Any]:
        dataset_root = self._evaluation_dataset_root(dataset_path)
        predictions_root = dataset_root / "predictions_xml"
        evaluations: list[dict[str, Any]] = []
        if predictions_root.is_dir():
            for candidate in predictions_root.iterdir():
                if not candidate.is_dir() or not self._valid_evaluation_id(candidate.name):
                    continue
                manifest_path = candidate / "manifest.json"
                if not manifest_path.is_file():
                    continue
                try:
                    manifest = read_json(manifest_path)
                    if not isinstance(manifest, dict):
                        continue
                    if manifest.get("evaluation_id") != candidate.name:
                        continue
                    evaluations.append(
                        {
                            "evaluation_id": candidate.name,
                            "created_at": manifest.get("created_at"),
                            "model_path": manifest.get("model_path"),
                            "model_source": manifest.get("model_source"),
                            "trial_id": manifest.get("trial_id"),
                            "checkpoint_name": manifest.get("checkpoint_name"),
                            "task_type": manifest.get("task_type"),
                            "conf": manifest.get("conf"),
                            "imgsz": manifest.get("imgsz"),
                            "batch": manifest.get("batch"),
                            "image_count": len(manifest.get("images") or []),
                            "metrics": manifest.get("metrics") or {},
                        }
                    )
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        evaluations.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return {"dataset_root": str(dataset_root), "evaluations": evaluations}

    def get_evaluation(self, dataset_path: str, evaluation_id: str) -> dict[str, Any]:
        if not self._valid_evaluation_id(evaluation_id):
            raise WorkbenchError("invalid evaluation id")
        dataset_root = self._evaluation_dataset_root(dataset_path)
        manifest_path = dataset_root / "predictions_xml" / evaluation_id / "manifest.json"
        if not manifest_path.is_file():
            raise WorkbenchError("evaluation not found")
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise WorkbenchError("evaluation manifest is invalid") from exc
        if not isinstance(manifest, dict):
            raise WorkbenchError("evaluation manifest is invalid")
        if manifest.get("evaluation_id") != evaluation_id:
            raise WorkbenchError("evaluation manifest id does not match its directory")
        evaluation_dir = ensure_dir(self._evaluation_dir(evaluation_id))
        write_json(evaluation_dir / "manifest.json", manifest)
        return manifest

    def image_path(self, session_id: str, image_id: str) -> Path:
        manifest, session_dir = self._session(session_id)
        record = next((item for item in manifest["images"] if item["image_id"] == image_id), None)
        if record is None:
            raise WorkbenchError("image not found")
        path = (session_dir / "images" / record["filename"]).resolve()
        if not path.is_relative_to(session_dir) or not path.is_file():
            raise WorkbenchError("image not found")
        return path

    def evaluation_image_path(self, evaluation_id: str, image_id: str) -> Path:
        manifest_path = self._evaluation_dir(evaluation_id) / "manifest.json"
        if not manifest_path.is_file():
            raise WorkbenchError("evaluation not found")
        manifest = read_json(manifest_path)
        record = next((item for item in manifest.get("images", []) if item.get("image_id") == image_id), None)
        if record is None:
            raise WorkbenchError("image not found")
        path = Path(record["source_path"]).resolve()
        if not path.is_file():
            raise WorkbenchError("source image no longer exists")
        return path

    def clear_cache(self) -> dict[str, Any]:
        files = 0
        size = 0
        if self.cache_root.exists():
            for path in self.cache_root.rglob("*"):
                if path.is_file():
                    files += 1
                    try:
                        size += path.stat().st_size
                    except OSError:
                        pass
            shutil.rmtree(self.cache_root)
        ensure_dir(self.cache_root / "sessions")
        ensure_dir(self.cache_root / "evaluations")
        return {"deleted_dirs": 1 if files else 0, "deleted_files": files, "deleted_bytes": size}

    def cleanup_expired(self) -> None:
        cutoff = datetime.now(timezone.utc) - CACHE_TTL
        for kind in ("sessions", "evaluations"):
            root = ensure_dir(self.cache_root / kind)
            for candidate in root.iterdir():
                if not candidate.is_dir():
                    continue
                try:
                    manifest = candidate / "manifest.json"
                    modified_path = manifest if manifest.is_file() else candidate
                    modified = datetime.fromtimestamp(modified_path.stat().st_mtime, timezone.utc)
                    if modified < cutoff:
                        shutil.rmtree(candidate)
                except OSError:
                    continue

    def _resolve_model(self, payload: dict[str, Any]) -> Path:
        source = str(payload.get("model_source", "local"))
        if source == "platform":
            trial_id = str(payload.get("trial_id", "")).strip()
            if not trial_id:
                raise WorkbenchError("trial_id is required")
            trial = self.repo.get_trial(trial_id)
            checkpoints = _checkpoint_paths(trial.run_dir)
            if not checkpoints:
                raise WorkbenchError("trial checkpoint not found")
            requested_checkpoint = str(payload.get("checkpoint_name", "") or checkpoints[0]["name"]).lower()
            checkpoint = next((item for item in checkpoints if item["name"] == requested_checkpoint), None)
            if checkpoint is None:
                raise WorkbenchError(f"checkpoint not found for trial: {requested_checkpoint}")
            model_path = checkpoint["path"]
        else:
            model_path = self._absolute_existing_path(str(payload.get("model_path", "")), "model")
        if model_path.suffix.lower() not in MODEL_EXTENSIONS:
            raise WorkbenchError("model must be a .pt or .onnx file")
        return model_path

    def _run_worker(self, request: dict[str, Any], request_path: Path) -> dict[str, Any]:
        write_json(request_path, request)
        env = dict(os.environ)
        src_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = src_root if not env.get("PYTHONPATH") else f"{src_root}{os.pathsep}{env['PYTHONPATH']}"
        completed = subprocess.run(
            [self.python_getter(), "-m", "backend.core.workbench_worker", str(request_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "workbench worker failed").strip()
            raise WorkbenchError(message.splitlines()[-1])
        try:
            return json.loads((completed.stdout or "").strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise WorkbenchError("workbench worker returned invalid JSON") from exc

    def _session(self, session_id: str) -> tuple[dict[str, Any], Path]:
        if not session_id.startswith("session_") or not session_id[8:].isalnum():
            raise WorkbenchError("invalid session id")
        session_dir = (self.cache_root / "sessions" / session_id).resolve()
        if not session_dir.is_relative_to(self.cache_root) or not session_dir.is_dir():
            raise WorkbenchError("session not found")
        manifest_path = session_dir / "manifest.json"
        if not manifest_path.is_file():
            raise WorkbenchError("session not found")
        return read_json(manifest_path), session_dir

    def _evaluation_dir(self, evaluation_id: str) -> Path:
        if not self._valid_evaluation_id(evaluation_id):
            raise WorkbenchError("invalid evaluation id")
        path = (self.cache_root / "evaluations" / evaluation_id).resolve()
        if not path.is_relative_to(self.cache_root):
            raise WorkbenchError("invalid evaluation id")
        return path

    def _evaluation_dataset_root(self, dataset_path: str) -> Path:
        path = self._absolute_existing_path(dataset_path, "dataset")
        inspection = self.inspect_dataset(str(path))
        root = Path(str(inspection.get("dataset_root", ""))).resolve()
        if not root.is_dir():
            raise WorkbenchError("dataset root not found")
        return root

    @staticmethod
    def _valid_evaluation_id(evaluation_id: str) -> bool:
        return evaluation_id.startswith("eval_") and all(char.isalnum() or char == "_" for char in evaluation_id)

    @staticmethod
    def _absolute_existing_path(raw: str, label: str) -> Path:
        value = str(raw or "").strip()
        if not value:
            raise WorkbenchError(f"{label} path is required")
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise WorkbenchError(f"{label} path must be absolute")
        path = path.resolve()
        if not path.exists():
            raise WorkbenchError(f"{label} path not found: {path}")
        return path

    @staticmethod
    def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
        number = float(value)
        if number < minimum or number > maximum:
            raise WorkbenchError(f"{name} must be between {minimum} and {maximum}")
        return number

    @staticmethod
    def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
        number = int(value)
        if number < minimum or number > maximum:
            raise WorkbenchError(f"{name} must be between {minimum} and {maximum}")
        return number
