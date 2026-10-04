from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import backend.workbench as workbench
from backend.workbench import WorkbenchError, WorkbenchService


def legacy_result(tmp_path: Path, count: int = 3):
    dataset = tmp_path / "数据集"
    dataset.mkdir()
    images = []
    for index in range(count):
        path = dataset / f"合成_{index}.png"
        Image.new("RGB", (320, 80), "blue").save(path)
        images.append({"image_id": f"eval_img_{index:06d}", "name": path.name, "source_path": str(path),
                       "width": 320, "height": 80, "labels": [{"class_id": 0, "x1": 10}],
                       "detections": [{"class_id": 0, "confidence": .75}]})
    result = {"evaluation_id": "eval_legacy", "images": images, "metrics": {"map50": .8},
              "classes": [{"class_id": 0, "class_name": "object"}], "per_class_metrics": [],
              "predictions_dir": str(dataset / "predictions_xml" / "eval_legacy")}
    directory = Path(result["predictions_dir"])
    directory.mkdir(parents=True)
    workbench.write_json(directory / "manifest.json", result)
    return dataset, result, WorkbenchService(SimpleNamespace(), lambda: "not-used", tmp_path / "cache")


def test_legacy_summary_detail_and_clear_cache(tmp_path):
    dataset, original, service = legacy_result(tmp_path)
    service.inspect_dataset = lambda _: pytest.fail("history must not run inspection")
    assert service.list_evaluations(str(dataset))["evaluations"][0]["image_count"] == 3
    summary = service.get_evaluation(str(dataset), "eval_legacy", "summary")
    assert summary["image_count"] == 3
    assert "labels" not in summary["images"][0]
    assert "source_path" not in summary["images"][0]
    assert service.evaluation_image_detail("eval_legacy", "eval_img_000000") == original["images"][0]
    assert service.get_evaluation(str(dataset), "eval_legacy") == original
    service.clear_cache()
    assert Path(original["predictions_dir"]).is_dir()
    assert service.get_evaluation(str(dataset), "eval_legacy", "summary") == summary


def test_concurrent_thumbnail_requests_reuse_index_and_thumbnail(tmp_path, monkeypatch):
    dataset, _, service = legacy_result(tmp_path)
    service.get_evaluation(str(dataset), "eval_legacy", "summary")
    original_read = workbench.read_json
    reads = []

    def read(path):
        reads.append(Path(path).name)
        return original_read(path)

    monkeypatch.setattr(workbench, "read_json", read)
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda _: service.evaluation_thumbnail_path("eval_legacy", "eval_img_000000"), range(16)))
    assert len(set(paths)) == 1
    with Image.open(paths[0]) as image:
        assert image.size == (160, 40)
    assert reads.count("image_index.json") == 1
    assert "manifest.json" not in reads
    timestamp = paths[0].stat().st_mtime_ns
    service.evaluation_thumbnail_path("eval_legacy", "eval_img_000000")
    assert paths[0].stat().st_mtime_ns == timestamp


def test_index_and_derived_views_refresh_when_manifest_changes(tmp_path):
    dataset, original, service = legacy_result(tmp_path)
    service.get_evaluation(str(dataset), "eval_legacy", "summary")
    service.evaluation_image_path("eval_legacy", "eval_img_000000")
    local = service._evaluation_dir("eval_legacy") / "manifest.json"
    original["images"][0]["detections"] = []
    workbench.write_json(local, original)
    assert service.evaluation_image_detail("eval_legacy", "eval_img_000000")["detections"] == []
    original["metrics"]["map50"] = .9
    workbench.write_json(Path(original["predictions_dir"]) / "manifest.json", original)
    assert service.get_evaluation(str(dataset), "eval_legacy", "summary")["metrics"]["map50"] == .9


def test_warm_summary_does_not_read_full_manifest(tmp_path, monkeypatch):
    dataset, _, service = legacy_result(tmp_path)
    service.get_evaluation(str(dataset), "eval_legacy", "summary")
    original_read = workbench.read_json

    def read(path):
        assert Path(path).name != "manifest.json"
        return original_read(path)

    monkeypatch.setattr(workbench, "read_json", read)
    service.get_evaluation(str(dataset), "eval_legacy", "summary")
    service.list_evaluations(str(dataset))


def test_evaluation_index_cache_is_bounded(tmp_path):
    dataset, original, service = legacy_result(tmp_path, 1)
    for index in range(10):
        evaluation_id = f"eval_{index}"
        directory = dataset / "predictions_xml" / evaluation_id
        directory.mkdir()
        workbench.write_json(directory / "manifest.json", {**original, "evaluation_id": evaluation_id})
        service.get_evaluation(str(dataset), evaluation_id, "summary")
        service.evaluation_image_path(evaluation_id, "eval_img_000000")
    assert len(service._evaluation_indexes) == 8
    assert "eval_0" not in service._evaluation_indexes


def test_missing_image_and_invalid_ids(tmp_path):
    dataset, original, service = legacy_result(tmp_path)
    service.get_evaluation(str(dataset), "eval_legacy", "summary")
    for method in (service.evaluation_image_detail, service.evaluation_image_path, service.evaluation_thumbnail_path):
        with pytest.raises(WorkbenchError):
            method("eval_legacy", "../../manifest")
        with pytest.raises(WorkbenchError):
            method("../eval_legacy", "eval_img_000000")
    Path(original["images"][0]["source_path"]).unlink()
    with pytest.raises(WorkbenchError, match="no longer exists"):
        service.evaluation_thumbnail_path("eval_legacy", "eval_img_000000")

