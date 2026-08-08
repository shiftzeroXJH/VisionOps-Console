from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from backend.core.workbench_worker import (
    _map_roi_detection,
    _serialize_boxes,
    _yolo_labels,
)
from backend.workbench import WorkbenchService


def _box(xyxy: list[float], class_id: int = 0, confidence: float = 0.8) -> SimpleNamespace:
    return SimpleNamespace(xyxy=[xyxy], cls=[class_id], conf=[confidence])


def test_serialize_segment_result_keeps_mask_polygon() -> None:
    result = SimpleNamespace(
        boxes=[_box([10, 20, 80, 90])],
        masks=SimpleNamespace(xy=[[[10, 20], [80, 20], [70, 90], [15, 80]]]),
    )

    detections = _serialize_boxes(result, {0: "part"}, "segment")

    assert detections[0]["class_name"] == "part"
    assert detections[0]["polygon"] == [[10.0, 20.0], [80.0, 20.0], [70.0, 90.0], [15.0, 80.0]]


def test_serialize_obb_result_uses_four_corner_geometry() -> None:
    result = SimpleNamespace(
        obb=SimpleNamespace(
            cls=[1],
            conf=[0.91],
            xyxyxyxy=[[[20, 40], [80, 20], [100, 60], [40, 80]]],
            xyxy=[[20, 20, 100, 80]],
        )
    )

    detections = _serialize_boxes(result, {1: "rotated"}, "obb")

    assert detections[0]["class_id"] == 1
    assert detections[0]["polygon"] == [[20.0, 40.0], [80.0, 20.0], [100.0, 60.0], [40.0, 80.0]]
    assert detections[0]["x1"] == 20.0
    assert detections[0]["y2"] == 80.0


def test_yolo_geometry_labels_are_scaled_to_image_pixels(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "val"
    label_dir = tmp_path / "labels" / "val"
    image_path = image_dir / "sample.png"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (200, 100)).save(image_path)
    (label_dir / "sample.txt").write_text("0 0.1 0.2 0.4 0.2 0.5 0.4 0.1 0.4\n", encoding="utf-8")

    labels = _yolo_labels(image_path, image_dir, label_dir, {0: "part"}, "obb")

    assert labels == [
        {
            "class_id": 0,
            "class_name": "part",
            "x1": 20.0,
            "y1": 20.0,
            "x2": 100.0,
            "y2": 40.0,
            "polygon": [[20.0, 20.0], [80.0, 20.0], [100.0, 40.0], [20.0, 40.0]],
        }
    ]


def test_roi_mapping_transforms_existing_polygon() -> None:
    detection = {
        "class_id": 0,
        "class_name": "part",
        "confidence": 0.8,
        "x1": 10.0,
        "y1": 10.0,
        "x2": 90.0,
        "y2": 90.0,
        "polygon": [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]],
    }

    mapped = _map_roi_detection(
        detection,
        {"cx": 100.0, "cy": 100.0, "width": 100.0, "height": 100.0, "angle": 0.0},
        (100, 100),
    )

    assert mapped["polygon"] == [[60.0, 60.0], [140.0, 60.0], [140.0, 140.0], [60.0, 140.0]]


def test_workbench_lists_all_supported_yolo_tasks(tmp_path: Path) -> None:
    experiments = [SimpleNamespace(experiment_id=task, task_type=task, description=task, project=task) for task in ("detection", "segment", "obb", "pose")]
    trials = {}
    for experiment in experiments:
        run_dir = tmp_path / experiment.task_type / "weights"
        run_dir.mkdir(parents=True)
        (run_dir / "best.pt").write_text("fake", encoding="utf-8")
        trials[experiment.experiment_id] = [SimpleNamespace(
            trial_id=f"trial-{experiment.task_type}",
            display_name=experiment.task_type,
            created_at="2026-01-01T00:00:00+00:00",
            iteration=1,
            run_dir=str(run_dir.parent),
        )]

    repo = SimpleNamespace(
        list_experiments=lambda: experiments,
        list_trials=lambda experiment_id: trials[experiment_id],
    )
    service = WorkbenchService(repo, lambda: "python", cache_root=tmp_path / "cache")

    models = service.list_models()["models"]

    assert {model["task_type"] for model in models} == {"detection", "segment", "obb"}


def test_evaluation_manifest_persists_with_predictions_and_can_be_reloaded(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    image_path = dataset_root / "sample.jpg"
    image_path.write_bytes(b"image-placeholder")
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model-placeholder")
    service = WorkbenchService(SimpleNamespace(), lambda: "python", cache_root=tmp_path / "cache")
    service.inspect_dataset = lambda dataset_path: {"dataset_root": str(dataset_root)}  # type: ignore[method-assign]

    def run_worker(request: dict, request_path: Path) -> dict:
        predictions_dir = dataset_root / "predictions_xml" / request["evaluation_id"]
        predictions_dir.mkdir(parents=True)
        (predictions_dir / "sample.xml").write_text("<annotation />", encoding="utf-8")
        return {
            "task_type": "detection",
            "dataset": {"dataset_root": str(dataset_root), "dataset_type": "voc"},
            "classes": [{"class_id": 0, "class_name": "part"}],
            "metrics": {"map50": 0.75},
            "per_class_metrics": [],
            "images": [{
                "image_id": "eval_img_000000",
                "name": image_path.name,
                "source_path": str(image_path),
                "width": 100,
                "height": 80,
                "labels": [],
                "detections": [],
                "xml_path": str(predictions_dir / "sample.xml"),
            }],
            "predictions_dir": str(predictions_dir),
        }

    service._run_worker = run_worker  # type: ignore[method-assign]
    result = service.evaluate({
        "model_source": "local",
        "model_path": str(model_path),
        "dataset_path": str(dataset_root),
        "conf": 0.25,
        "imgsz": 640,
        "batch": 8,
    })
    persistent_manifest = Path(result["predictions_dir"]) / "manifest.json"

    assert persistent_manifest.is_file()
    service.clear_cache()
    assert persistent_manifest.is_file()
    assert service.list_evaluations(str(dataset_root))["evaluations"][0]["evaluation_id"] == result["evaluation_id"]

    loaded = service.get_evaluation(str(dataset_root), result["evaluation_id"])

    assert loaded["metrics"]["map50"] == 0.75
    assert service.evaluation_image_path(result["evaluation_id"], "eval_img_000000") == image_path
