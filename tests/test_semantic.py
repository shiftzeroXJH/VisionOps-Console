from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.core.model_catalog import model_catalog, select_model, resolve_training_model, check_model_task
from backend.core.analyzer import build_summary
from backend.core.train_worker import _extract_per_class_metrics
from backend.core.validation_worker import _extract_metrics


def make_dataset(root, polygons=False, binary=False):
    import numpy as np
    import yaml
    from PIL import Image
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / ("labels" if polygons else "masks") / split).mkdir(parents=True)
        for i in range(2):
            image = np.zeros((64, 96, 3), dtype=np.uint8)
            image[:, :, 1] = 60
            image[16:48, 24:72] = (230, 70, 100)
            Image.fromarray(image).save(root / "images" / split / f"sample{i}.png")
            if polygons:
                (root / "labels" / split / f"sample{i}.txt").write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n")
            else:
                mask = np.zeros((64, 96), dtype=np.uint8)
                mask[16:48, 24:72] = 1
                mask[0, :] = 255
                Image.fromarray(mask).save(root / "masks" / split / f"sample{i}.png")
    data = {"path": root.as_posix(), "train": "images/train", "val": "images/val",
            "names": {0: "object"} if binary else {0: "background" if not polygons else "object", 1: "object" if not polygons else "second"}}
    if not polygons:
        data["masks_dir"] = "masks"
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_catalog_matrix():
    catalog = model_catalog()
    assert len(catalog["models"]) == 50
    for row in catalog["models"]:
        assert select_model(row["task_type"], row["model_family"], row["model_scale"]) == row["filename"]
    for selection in [("semantic", "11", "n"), ("semantic", "v8", "s"), ("detection", "26", "z")]:
        with pytest.raises(ValueError):
            select_model(*selection)


def test_weight_cache_serializes_downloads(tmp_path, monkeypatch):
    import ultralytics.utils.downloads as downloads
    calls = []
    def download(path):
        calls.append(path)
        Path(path).write_bytes(b"x" * 2048)
        return path
    monkeypatch.setattr(downloads, "attempt_download_asset", download)
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: resolve_training_model("yolo26n-sem.pt", tmp_path), range(4)))
    assert len(set(outputs)) == 1
    assert len(calls) == 1
    custom = tmp_path / "custom.pt"
    custom.write_bytes(b"custom")
    assert resolve_training_model(str(custom), tmp_path) == str(custom.resolve())
    with pytest.raises(FileNotFoundError):
        resolve_training_model(str(tmp_path / "missing.pt"), tmp_path)


def test_download_failure_does_not_publish(tmp_path, monkeypatch):
    import ultralytics.utils.downloads as downloads
    monkeypatch.setattr(downloads, "attempt_download_asset", lambda path: path)
    with pytest.raises(RuntimeError, match="download failed"):
        resolve_training_model("yolo26n-sem.pt", tmp_path)
    assert not (tmp_path / "yolo26n-sem.pt").exists()


def test_semantic_summary_and_metrics(tmp_path):
    (tmp_path / "results.csv").write_text("epoch,time,metrics/mIoU,metrics/pixel_acc,train/seg_loss\n1,2,0.8,0.9,1\n2,4,0.6,0.95,0.5\n")
    summary = build_summary("trial", "semantic", str(tmp_path), {"epochs": 2}).to_dict()
    assert summary["basic_info"]["best_epoch"] == 1
    assert summary["final_metrics"] == {"miou": 0.8, "pixel_accuracy": 0.9}
    assert summary["delta_vs_prev"]["miou"] is None
    fake = SimpleNamespace(miou=0.8, pixel_accuracy=0.9, names={0: "object"},
                           per_class_iou=[0.8], per_class_pixel_accuracy=[0.9], ap_class_index=[0])
    assert _extract_metrics(fake, "semantic") == summary["final_metrics"]
    assert _extract_per_class_metrics(fake, "semantic")[0]["iou"] == 0.8
    with pytest.raises(ValueError, match="does not match"):
        check_model_task(SimpleNamespace(task="detect", task_map={"semantic": {}}), "semantic", {})


@pytest.mark.parametrize("polygons,binary", [(False, False), (False, True), (True, False), (True, True)])
def test_semantic_datasets(tmp_path, polygons, binary):
    from backend.core.semantic import analyze_semantic_dataset, label_masks
    path = make_dataset(tmp_path / "dataset_\u4e2d\u6587", polygons, binary)
    analysis = analyze_semantic_dataset(path)
    assert analysis["totals"]["total_pixels"] > 0
    assert analysis["splits"]["val"]["image_count"] == 2
    assert "total_instances" not in analysis["totals"]
    masks, names = label_masks(path)
    assert len(masks) == 2
    assert len(names) == (2 if binary or not polygons else 3)
    assert all(mask.shape == (64, 96) for mask in masks.values())


def test_semantic_roi_raster(tmp_path):
    import numpy as np
    from PIL import Image
    from backend.core.semantic import serialize_mask
    result = serialize_mask(np.ones((10, 20), dtype=np.uint8), {0: "object"}, tmp_path / "mask.png",
                            (100, 80), {"cx": 50, "cy": 40, "width": 20, "height": 10, "angle": 90}, (20, 10))
    values = np.asarray(Image.open(result["mask_path"]))
    assert values.shape == (80, 100)
    assert (values == 1).sum() == 200
    assert len(result["layers"]) == 1
    assert result["layers"][0]["class_id"] == 1


@pytest.mark.parametrize("fault", ["missing", "size", "class"])
def test_invalid_semantic_masks_are_rejected(tmp_path, fault):
    import numpy as np
    from PIL import Image
    from backend.core.semantic import analyze_semantic_dataset
    path = make_dataset(tmp_path / "invalid")
    mask = path.parent / "masks" / "val" / "sample0.png"
    if fault == "missing":
        mask.unlink()
    else:
        Image.fromarray(np.full((32, 32) if fault == "size" else (64, 96), 7 if fault == "class" else 0, dtype=np.uint8)).save(mask)
    with pytest.raises(ValueError):
        analyze_semantic_dataset(path)


def test_semantic_label_mapping(tmp_path):
    import numpy as np
    import yaml
    from PIL import Image
    from backend.core.semantic import analyze_semantic_dataset
    path = make_dataset(tmp_path / "mapping")
    data = yaml.safe_load(path.read_text())
    data["label_mapping"] = {7: 1, 255: "ignore_label"}
    path.write_text(yaml.safe_dump(data))
    mask = path.parent / "masks" / "val" / "sample0.png"
    Image.fromarray(np.full((64, 96), 7, dtype=np.uint8)).save(mask)
    result = analyze_semantic_dataset(path)
    foreground = next(row for row in result["classes"] if row["class_id"] == 1)
    assert foreground["val_pixels"] == 64 * 96 + 32 * 48


def test_remote_semantic_metric_parity():
    from backend.core.remote_train_worker import _extract_per_class_metrics as remote_metrics
    fake = SimpleNamespace(names={0: "object"}, per_class_iou=[0.3],
                           per_class_pixel_accuracy=[0.4], ap_class_index=[0])
    assert remote_metrics(fake, "semantic") == _extract_per_class_metrics(fake, "semantic")


def test_missing_semantic_metrics_stay_missing(tmp_path):
    (tmp_path / "results.csv").write_text("epoch,time,train/ce_loss\n1,1,0.5\n")
    result = build_summary("empty", "semantic", str(tmp_path), {"epochs": 1})
    assert result.final_metrics == {"miou": None, "pixel_accuracy": None}
    assert result.metric_context["selection_fitness"] is None


def test_creation_contract(tmp_path):
    from backend.service import OrchestratorService, ServiceError
    service = OrchestratorService(str(tmp_path / "state.sqlite"))
    data = tmp_path / "data.yaml"
    data.write_text("train: images/train\nval: images/val\nnames: [object]\n")
    args = dict(description="semantic", task_type="sem", dataset_root=str(tmp_path), dataset_yaml=str(data), save_root=str(tmp_path / "runs"))
    created = service.create_experiment(**args, model_family="26", model_scale="n")
    config = service.repo.get_experiment(created["experiment_id"])
    assert config.task_type == "semantic"
    assert config.pretrained_model == "yolo26n-sem.pt"
    with pytest.raises(ServiceError, match="cannot be supplied together"):
        service.create_experiment(**args, model_family="26", model_scale="n", pretrained="custom.pt")
    old = service.create_experiment(**args, pretrained="yolo26s-sem.pt")
    assert old["experiment_id"]


@pytest.mark.skipif(os.environ.get("RUN_SEMANTIC_SMOKE") != "1", reason="explicit synthetic training smoke test")
def test_semantic_training_export_inference(tmp_path):
    import numpy as np
    from ultralytics import YOLO
    from backend.core.train_worker import _train
    from backend.core.workbench_worker import run_inference, run_evaluation
    from backend.core.validation_worker import run_validation_preview
    path = make_dataset(tmp_path / "synthetic")
    model = YOLO("yolo26n-sem.yaml")
    initial = tmp_path / "initial.pt"
    model.save(initial)
    run_dir = tmp_path / "run"
    _train({"pretrained_model": str(initial), "dataset_yaml": str(path), "run_dir": str(run_dir),
            "task_type": "semantic", "params": {"epochs": 1, "patience": 1, "imgsz": 160, "batch": 2,
            "workers": 0, "device": "cpu", "amp": False, "plots": False, "mosaic": 0, "cache": False}})
    best = run_dir / "weights" / "best.pt"
    assert best.is_file()
    summary = build_summary("smoke", "semantic", str(run_dir), {"epochs": 1})
    assert summary.final_metrics["miou"] is not None
    model = YOLO(best)
    exported = model.export(format="onnx", imgsz=64, opset=13, simplify=True, dynamic=False, device="cpu", nms=False)
    image_path = path.parent / "images" / "val" / "sample0.png"
    request = {"model_path": str(exported), "conf": 0.25, "imgsz": 64,
               "images": [{"image_id": "synthetic", "path": str(image_path)}]}
    result = run_inference(request)
    assert result["images"][0]["status"] == "completed", result
    assert result["images"][0]["semantic"]["width"] == 96
    pt = model.predict(str(image_path), imgsz=64, verbose=False)[0].semantic_mask.data.cpu().numpy()
    from PIL import Image
    onnx_mask = np.asarray(Image.open(result["images"][0]["semantic"]["mask_path"]))
    assert (pt == onnx_mask).mean() > 0.99
    evaluation = run_evaluation({"model_path": str(best), "task_type": "semantic", "dataset_path": str(path),
                                 "imgsz": 64, "batch": 2, "evaluation_id": "eval_semantic_smoke", "display_conf": 0.25})
    assert "miou" in evaluation["metrics"]
    assert evaluation["images"][0]["semantic_labels"]
    preview = run_validation_preview({"model_path": str(best), "task_type": "semantic", "dataset_yaml": str(path),
                                      "imgsz": 64, "batch": 2, "image_limit": 1, "conf": 0.25, "output_dir": str(tmp_path / "preview")})
    assert "miou" in preview["metrics"]
    from backend.service import OrchestratorService
    service = OrchestratorService(str(tmp_path / "smoke.sqlite"))
    experiment = service.create_experiment(description="smoke", task_type="semantic", dataset_root=str(path.parent),
        dataset_yaml=str(path), save_root=str(tmp_path / "platform"), model_family="26", model_scale="n")
    imported = service.import_run(experiment["experiment_id"], run_dir=str(run_dir))
    comparison = service.compare_experiment(experiment["experiment_id"])
    assert comparison["best_trial"]["metric"] == "miou"
    assert comparison["rows"][0]["miou"] is not None
    assert service.list_experiments()["experiments"][0]["best_metric"]["metric"] == "miou"
    continuation = service.prepare_continuation_request(imported["trial_id"], additional_epochs=1,
                                                        lr0=None, patience=1, note="synthetic continuation")
    assert Path(continuation["pretrained"]).is_file()
    from backend.core.remote_train_worker import _extract_per_class_metrics as remote_metrics
    assert remote_metrics(model.val(data=str(path), imgsz=64, batch=2, workers=0, plots=False, verbose=False), "semantic")
