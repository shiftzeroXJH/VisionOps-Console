from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend.core.evaluation_validator import capture_validator, serialize_validation_prediction


@pytest.mark.parametrize("task", ["detection", "segment", "obb"])
def test_capture_preserves_metric_inputs_and_original_coordinates(task, monkeypatch):
    emitted, metric_confidences = [], []
    validator_type = capture_validator(task, .25, lambda *args: emitted.append(args))
    base = validator_type.__bases__[0]
    monkeypatch.setattr(base, "update_metrics", lambda self, preds, batch: metric_confidences.append(preds[0]["conf"].clone()))
    validator = object.__new__(validator_type)
    bboxes = torch.tensor([[20., 30., 60., 50.]] * 3)
    if task == "obb":
        bboxes = torch.tensor([[40., 40., 40., 20., 0.]] * 3)
    prediction = {"bboxes": bboxes, "cls": torch.zeros(3), "conf": torch.tensor([.1, .25, .9])}
    if task == "segment":
        masks = torch.zeros((3, 80, 80))
        masks[:, 30:50, 20:60] = 1
        prediction["masks"] = masks
    empty = {key: value[:0].clone() for key, value in prediction.items()}
    before = {key: value.clone() for key, value in prediction.items()}
    batch = {"im_file": ["合成.png", "空图.png"], "ori_shape": [(40, 80), (40, 80)],
             "img": torch.zeros((2, 3, 80, 80)), "ratio_pad": [((1., 1.), (0., 20.))] * 2}
    validator.update_metrics([prediction, empty], batch)
    assert metric_confidences[0].tolist() == pytest.approx([.1, .25, .9])
    assert all(torch.equal(prediction[key], value) for key, value in before.items())
    assert emitted[0][0] == str(Path("合成.png").resolve())
    records = serialize_validation_prediction(emitted[0][1], emitted[0][2], {0: "object"}, task)
    assert len(records) == 2
    assert [row["confidence"] for row in records] == pytest.approx([.25, .9])
    assert records[0]["x1"] == pytest.approx(20)
    assert records[0]["y1"] == pytest.approx(10)
    if task in {"segment", "obb"}:
        assert len(records[0]["polygon"]) >= 4
        assert all(0 <= y <= 40 for x, y in records[0]["polygon"])
    assert serialize_validation_prediction(emitted[1][1], emitted[1][2], {0: "object"}, task) == []


def test_semantic_capture_emits_class_ids_without_changing_metrics(monkeypatch):
    emitted, metric_inputs = [], []
    validator_type = capture_validator("semantic", 0, lambda *args: emitted.append(args))
    monkeypatch.setattr(validator_type.__bases__[0], "update_metrics", lambda self, pred, batch: metric_inputs.append(pred.clone()))
    validator = object.__new__(validator_type)
    validator.image_shapes = {"synthetic.png": (40, 80)}
    pred = torch.tensor([[[0, 1], [1, 255]]])
    validator.update_metrics(pred, {"im_file": ["synthetic.png"]})
    assert torch.equal(metric_inputs[0], pred)
    assert np.array_equal(emitted[0][1], pred[0].numpy())
    assert emitted[0][2] == (40, 80)


def test_capture_reports_incompatible_prediction_shape(monkeypatch):
    validator_type = capture_validator("detection", .25, lambda *args: None)
    monkeypatch.setattr(validator_type.__bases__[0], "update_metrics", lambda *args: None)
    validator = object.__new__(validator_type)
    with pytest.raises(RuntimeError, match="capture incompatible"):
        validator.update_metrics([{}], {"im_file": ["synthetic.png"], "ori_shape": [(40, 80)]})


@pytest.mark.parametrize("task,config", [("detection", "yolo11n.yaml"), ("segment", "yolo11n-seg.yaml"),
                                         ("obb", "yolo11n-obb.yaml"), ("semantic", "yolo26n-sem.yaml")])
def test_real_validation_metrics_match_and_worker_never_predicts(tmp_path, monkeypatch, task, config):
    from PIL import Image
    from ultralytics import YOLO
    from ultralytics.utils import YAML
    from ultralytics.nn.tasks import yaml_model_load
    import backend.core.workbench_worker as worker
    root = tmp_path / "synthetic"
    image_dir = root / "images" / "val"
    label_dir = root / "labels" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index in range(2):
        Image.new("RGB", (80, 40), (index * 80, 50, 100)).save(image_dir / f"sample{index}.png")
        label = "0 .5 .5 .5 .5" if task == "detection" else "0 .25 .25 .75 .25 .75 .75 .25 .75"
        (label_dir / f"sample{index}.txt").write_text(label + "\n")
    yaml = root / "data.yaml"
    yaml.write_text(f"path: {root.as_posix()}\ntrain: images/val\nval: images/val\nnames: [object]\n")
    configuration = yaml_model_load(config)
    configuration["nc"] = 1
    model_yaml = tmp_path / config
    YAML.save(model_yaml, configuration)
    model = YOLO(str(model_yaml), task="detect" if task == "detection" else task)
    model.model.names = {0: "object"}
    args = dict(data=str(yaml), split="val", imgsz=64, batch=2, workers=0, device="cpu",
                plots=False, save=False, verbose=False, project=str(tmp_path / "runs"), name="native")
    if task != "semantic":
        args["conf"] = .001
    native = model.val(**args)
    expected = worker._extract_metrics(native, task)
    original_val = model.val

    def val(**kwargs):
        return original_val(**{**kwargs, "device": "cpu", "project": str(tmp_path / "runs"), "name": "capture"})

    monkeypatch.setattr(model, "val", val)
    monkeypatch.setattr(model, "predict", lambda **kwargs: pytest.fail("evaluation must never predict twice"))
    monkeypatch.setattr(worker, "_load_model", lambda *args: (model, task))
    result = worker.run_evaluation({"model_path": "synthetic.pt", "task_type": task, "dataset_path": str(yaml),
                                    "imgsz": 64, "batch": 2, "evaluation_id": "eval_real", "display_conf": .25})
    assert result["metrics"] == pytest.approx(expected, abs=1e-6)
    assert len(result["images"]) == 2
    for record in result["images"]:
        assert (record["width"], record["height"]) == (80, 40)
        if task == "semantic":
            with Image.open(record["semantic"]["mask_path"]) as mask:
                assert mask.size == (80, 40)
        else:
            assert Path(record["xml_path"]).is_file()


def test_voc_adapter_maps_reordered_predictions_and_keeps_same_stem_exports(tmp_path, monkeypatch):
    from PIL import Image
    import backend.core.workbench_worker as worker
    root = tmp_path / "合成数据"
    root.mkdir()
    for suffix in ("png", "jpg"):
        Image.new("RGB", (80, 40), "blue").save(root / f"same.{suffix}")
    (root / "same.xml").write_text("<annotation><object><name>object</name><bndbox>"
                                  "<xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>30</ymax>"
                                  "</bndbox></object></annotation>")

    def val(**kwargs):
        validator_type = kwargs["validator"]
        monkeypatch.setattr(validator_type.__bases__[0], "update_metrics", lambda *args: None)
        validator = object.__new__(validator_type)
        adapter = root / ".workbench_adapter_eval_adapter" / "images" / "val"
        paths = list(reversed(sorted(adapter.iterdir())))
        predictions = [{"bboxes": torch.tensor([[10., 10., 30., 30.]]), "cls": torch.zeros(1),
                        "conf": torch.tensor([.8 + index * .1])} for index in range(2)]
        validator.update_metrics(predictions, {"im_file": [str(path) for path in paths],
            "ori_shape": [(40, 80)] * 2, "img": torch.zeros((2, 3, 40, 80)),
            "ratio_pad": [((1., 1.), (0., 0.))] * 2})
        return SimpleNamespace()

    model = SimpleNamespace(names={0: "object"}, val=val,
                            predict=lambda **kwargs: pytest.fail("adapter must not predict twice"))
    monkeypatch.setattr(worker, "_load_model", lambda *args: (model, "detection"))
    monkeypatch.setattr(worker, "_extract_metrics", lambda *args: {})
    monkeypatch.setattr(worker, "_extract_per_class_metrics", lambda *args: [])
    result = worker.run_evaluation({"model_path": "synthetic.pt", "dataset_path": str(root),
                                    "imgsz": 64, "batch": 2, "evaluation_id": "eval_adapter", "display_conf": .25})
    assert [item["name"] for item in result["images"]] == ["same.jpg", "same.png"]
    assert [item["detections"][0]["confidence"] for item in result["images"]] == pytest.approx([.9, .8])
    assert len({item["xml_path"] for item in result["images"]}) == 2
    assert not (root / ".workbench_adapter_eval_adapter").exists()
