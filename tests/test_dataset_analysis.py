from __future__ import annotations

from pathlib import Path
import subprocess
import sqlite3

from backend.core.dataset import analyze_dataset
from backend.constants import SEARCH_SPACE, STATE_COMPLETED, STATE_READY, STATE_WAITING, STOP_CONDITIONS
from backend.db.repository import Repository
from backend.service import OrchestratorService
from backend.models import ExperimentConfig, TrialRecord


def _write_file(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_results(run_dir: Path, map50_95: float = 0.42) -> None:
    _write_file(
        run_dir / "results.csv",
        "\n".join(
            [
                "epoch,time,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),train/box_loss,val/box_loss,gpu_mem",
                f"1,1.5,0.50,0.40,0.55,{map50_95 - 0.02:.4f},1.2,1.4,2048",
                f"2,1.4,0.60,0.50,0.65,{map50_95:.4f},0.9,1.1,2048",
            ]
        ),
    )


def _create_experiment(service: OrchestratorService, tmp_path: Path, dataset_root: Path) -> str:
    result = service.create_experiment(
        description="dataset analysis experiment",
        task_type="detection",
        dataset_root=str(dataset_root),
        dataset_yaml=None,
        pretrained="missing-ok.pt",
        save_root=str(tmp_path / "runs"),
        initial_params={"imgsz": 224, "batch": 8, "epochs": 2, "workers": 0},
    )
    return result["experiment_id"]


def test_analyze_dataset_counts_detection_with_dict_names(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_file(
        dataset_root / "data.yaml",
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: cat",
                "  1: dog",
            ]
        ),
    )
    _write_file(dataset_root / "images/train" / "a.jpg")
    _write_file(dataset_root / "images/train" / "b.jpg")
    _write_file(dataset_root / "images/val" / "c.jpg")
    _write_file(dataset_root / "labels/train" / "a.txt", "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.2 0.2\n")
    _write_file(dataset_root / "labels/train" / "b.txt", "1 0.6 0.6 0.2 0.2\n")
    _write_file(dataset_root / "labels/val" / "c.txt", "0 0.5 0.5 0.2 0.2\n")

    result = analyze_dataset(str(dataset_root / "data.yaml"))

    assert result["totals"]["train_instances"] == 3
    assert result["totals"]["val_instances"] == 1
    assert result["totals"]["class_count"] == 2
    assert result["splits"]["train"]["image_count"] == 2
    assert result["splits"]["val"]["label_file_count"] == 1
    assert result["classes"] == [
        {
            "class_id": 0,
            "class_name": "cat",
            "train_instances": 1,
            "val_instances": 1,
            "total_instances": 2,
            "total_ratio": 0.5,
        },
        {
            "class_id": 1,
            "class_name": "dog",
            "train_instances": 2,
            "val_instances": 0,
            "total_instances": 2,
            "total_ratio": 0.5,
        },
    ]


def test_analyze_dataset_counts_obb_with_list_names(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_obb"
    _write_file(
        dataset_root / "data.yaml",
        "\n".join(
            [
                "train: images/train",
                "val: images/val",
                "names: ['board', 'chip']",
            ]
        ),
    )
    _write_file(dataset_root / "images/train" / "a.jpg")
    _write_file(dataset_root / "images/val" / "b.jpg")
    _write_file(dataset_root / "labels/train" / "a.txt", "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n1 0.2 0.2 0.3 0.2 0.3 0.3 0.2 0.3\n")
    _write_file(dataset_root / "labels/val" / "b.txt", "1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")

    result = analyze_dataset(str(dataset_root / "data.yaml"))

    assert result["classes"][0]["class_name"] == "board"
    assert result["classes"][1]["class_name"] == "chip"
    assert result["classes"][1]["train_instances"] == 1
    assert result["classes"][1]["val_instances"] == 1
    assert result["totals"]["total_instances"] == 3


def test_analyze_dataset_reports_missing_split_and_unknown_class(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_missing"
    _write_file(
        dataset_root / "data.yaml",
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: only",
            ]
        ),
    )
    _write_file(dataset_root / "images/train" / "a.jpg")
    _write_file(dataset_root / "labels/train" / "a.txt", "2 0.5 0.5 0.2 0.2\n")

    result = analyze_dataset(str(dataset_root / "data.yaml"))

    assert "val_image_dir_missing" in result["warnings"]
    assert "unknown_class_id:2" in result["warnings"]
    assert result["classes"][0]["class_name"] == "class_2"


def test_get_summary_returns_trial_dataset_snapshot(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_service"
    _write_file(
        dataset_root / "data.yaml",
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: broken",
                "  1: normal",
            ]
        ),
    )
    _write_file(dataset_root / "images/train" / "a.jpg")
    _write_file(dataset_root / "images/val" / "b.jpg")
    _write_file(dataset_root / "labels/train" / "a.txt", "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n")
    _write_file(dataset_root / "labels/val" / "b.txt", "1 0.5 0.5 0.2 0.2\n")

    service = OrchestratorService(db_path=":memory:")
    experiment_id = _create_experiment(service, tmp_path, dataset_root)
    run_dir = tmp_path / "imported-run"
    _write_results(run_dir)

    imported = service.import_run(experiment_id, run_dir=str(run_dir), note="snapshot")
    summary = service.get_summary(imported["trial_id"])

    assert summary["dataset_analysis"]["totals"]["train_instances"] == 2
    assert summary["dataset_analysis"]["totals"]["val_instances"] == 1
    assert summary["dataset_analysis"]["classes"][1]["class_name"] == "normal"


def test_imported_trials_default_display_name_is_scoped_per_experiment(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_names"
    _write_file(
        dataset_root / "data.yaml",
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: part",
            ]
        ),
    )
    _write_file(dataset_root / "images/train" / "a.jpg")
    _write_file(dataset_root / "images/val" / "b.jpg")
    _write_file(dataset_root / "labels/train" / "a.txt", "0 0.5 0.5 0.2 0.2\n")
    _write_file(dataset_root / "labels/val" / "b.txt", "0 0.5 0.5 0.2 0.2\n")

    service = OrchestratorService(db_path=":memory:")
    experiment_a = _create_experiment(service, tmp_path, dataset_root)
    experiment_b = _create_experiment(service, tmp_path, dataset_root)

    run_dir_a1 = tmp_path / "run_a1"
    run_dir_a2 = tmp_path / "run_a2"
    run_dir_b1 = tmp_path / "run_b1"
    _write_results(run_dir_a1)
    _write_results(run_dir_a2, map50_95=0.44)
    _write_results(run_dir_b1, map50_95=0.46)

    trial_a1 = service.import_run(experiment_a, run_dir=str(run_dir_a1))
    trial_a2 = service.import_run(experiment_a, run_dir=str(run_dir_a2))
    trial_b1 = service.import_run(experiment_b, run_dir=str(run_dir_b1))

    summary_a1 = service.get_summary(trial_a1["trial_id"])
    summary_a2 = service.get_summary(trial_a2["trial_id"])
    summary_b1 = service.get_summary(trial_b1["trial_id"])

    assert summary_a1["trial"]["display_name"] == "missing_ok_224_1"
    assert summary_a2["trial"]["display_name"] == "missing_ok_224_2"
    assert summary_b1["trial"]["display_name"] == "missing_ok_224_1"


def test_trial_can_be_renamed_within_experiment(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_rename"
    _write_file(
        dataset_root / "data.yaml",
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: part",
            ]
        ),
    )
    _write_file(dataset_root / "images/train" / "a.jpg")
    _write_file(dataset_root / "images/val" / "b.jpg")
    _write_file(dataset_root / "labels/train" / "a.txt", "0 0.5 0.5 0.2 0.2\n")
    _write_file(dataset_root / "labels/val" / "b.txt", "0 0.5 0.5 0.2 0.2\n")

    service = OrchestratorService(db_path=":memory:")
    experiment_id = _create_experiment(service, tmp_path, dataset_root)

    run_dir_1 = tmp_path / "rename_1"
    run_dir_2 = tmp_path / "rename_2"
    _write_results(run_dir_1)
    _write_results(run_dir_2, map50_95=0.45)

    trial_1 = service.import_run(experiment_id, run_dir=str(run_dir_1))
    trial_2 = service.import_run(experiment_id, run_dir=str(run_dir_2))

    renamed = service.rename_trial(trial_1["trial_id"], "AOI_base_try")
    assert renamed["display_name"] == "AOI_base_try"

    summary = service.get_summary(trial_1["trial_id"])
    assert summary["trial"]["display_name"] == "AOI_base_try"

    try:
        service.rename_trial(trial_2["trial_id"], "AOI_base_try")
    except RuntimeError as exc:
        assert "already exists in experiment" in str(exc)
    else:
        raise AssertionError("expected duplicate trial display name to be rejected")


def test_create_experiment_uses_explicit_project(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_project"
    _write_file(
        dataset_root / "data.yaml",
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: part",
            ]
        ),
    )
    service = OrchestratorService(db_path=":memory:")

    created = service.create_experiment(
        description="莫仕低倍整体检测",
        project="莫仕",
        task_type="detection",
        dataset_root=str(dataset_root),
        dataset_yaml=None,
        pretrained="missing-ok.pt",
        save_root=str(tmp_path / "runs"),
        initial_params={"imgsz": 224, "batch": 8, "epochs": 2, "workers": 0},
    )

    detail = service.get_experiment_detail(created["experiment_id"])
    listed = service.list_experiments()["experiments"][0]

    assert created["project"] == "莫仕"
    assert created["status"] == "NOT_STARTED"
    assert created["internal_status"] == "READY"
    assert detail["experiment"]["project"] == "莫仕"
    assert detail["experiment"]["status"] == "NOT_STARTED"
    assert "goal" not in detail["experiment"]
    assert listed["project"] == "莫仕"


def test_create_experiment_defaults_project_from_description(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_project_default"
    _write_file(
        dataset_root / "data.yaml",
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: part",
            ]
        ),
    )
    service = OrchestratorService(db_path=":memory:")

    created = service.create_experiment(
        description="金具 224 baseline",
        task_type="detection",
        dataset_root=str(dataset_root),
        dataset_yaml=None,
        pretrained="missing-ok.pt",
        save_root=str(tmp_path / "runs"),
        initial_params={"imgsz": 224, "batch": 8, "epochs": 2, "workers": 0},
    )

    assert created["project"] == "金具"
    assert service.get_experiment_detail(created["experiment_id"])["experiment"]["project"] == "金具"


def test_update_experiment_project(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_project_update"
    _write_file(
        dataset_root / "data.yaml",
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: part",
            ]
        ),
    )
    service = OrchestratorService(db_path=":memory:")
    experiment_id = _create_experiment(service, tmp_path, dataset_root)

    updated = service.update_experiment(experiment_id, project="新项")

    assert updated["experiment"]["project"] == "新项"
    assert service.list_experiments()["experiments"][0]["project"] == "新项"


def test_validate_trial_preview_rejects_missing_weight(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_validation_missing_weight"
    _write_file(
        dataset_root / "data.yaml",
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: part",
            ]
        ),
    )
    service = OrchestratorService(db_path=":memory:")
    experiment_id = _create_experiment(service, tmp_path, dataset_root)
    trial = TrialRecord(
        trial_id="trial_missing_weight",
        display_name="missing_weight",
        experiment_id=experiment_id,
        iteration=1,
        params={"imgsz": 224, "batch": 8},
        status="COMPLETED",
        run_dir=str(tmp_path / "runs" / "missing_weight"),
    )
    Path(trial.run_dir).mkdir(parents=True)
    service.repo.create_trial(trial)

    try:
        service.validate_trial_preview(trial.trial_id)
    except RuntimeError as exc:
        assert "no validation weight file found" in str(exc)
    else:
        raise AssertionError("expected missing weights to be rejected")


def test_validate_trial_preview_validates_request_params(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_validation_params"
    _write_file(
        dataset_root / "data.yaml",
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: part",
            ]
        ),
    )
    service = OrchestratorService(db_path=":memory:")
    experiment_id = _create_experiment(service, tmp_path, dataset_root)
    run_dir = tmp_path / "runs" / "validation_params"
    _write_file(run_dir / "weights" / "best.pt", "fake")
    service.repo.create_trial(
        TrialRecord(
            trial_id="trial_bad_params",
            display_name="bad_params",
            experiment_id=experiment_id,
            iteration=1,
            params={"imgsz": 224, "batch": 8},
            status="COMPLETED",
            run_dir=str(run_dir),
        )
    )

    for kwargs, message in [
        ({"image_limit": 0}, "image_limit must be between"),
        ({"image_limit": 501}, "image_limit must be between"),
        ({"conf": 0.0}, "conf must be between"),
        ({"conf": 1.5}, "conf must be between"),
    ]:
        try:
            service.validate_trial_preview("trial_bad_params", **kwargs)
        except RuntimeError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"expected invalid params to be rejected: {kwargs}")


def test_validation_preview_file_path_is_sandboxed(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_validation_files"
    _write_file(
        dataset_root / "data.yaml",
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: part",
            ]
        ),
    )
    service = OrchestratorService(db_path=":memory:")
    experiment_id = _create_experiment(service, tmp_path, dataset_root)
    run_dir = tmp_path / "runs" / "validation_files"
    preview_file = run_dir / ".validation_previews" / "val_20260603_010203_000001" / "0001_a.jpg"
    _write_file(preview_file, "fake-image")
    service.repo.create_trial(
        TrialRecord(
            trial_id="trial_preview_file",
            display_name="preview_file",
            experiment_id=experiment_id,
            iteration=1,
            params={"imgsz": 224, "batch": 8},
            status="COMPLETED",
            run_dir=str(run_dir),
        )
    )

    assert service.get_validation_preview_file_path(
        "trial_preview_file",
        "val_20260603_010203_000001",
        "0001_a.jpg",
    ) == str(preview_file.resolve())

    for validation_id, filename in [
        ("..", "0001_a.jpg"),
        ("val_20260603_010203_000001", "../secret.jpg"),
        ("val_20260603_010203_000001", "missing.jpg"),
    ]:
        try:
            service.get_validation_preview_file_path("trial_preview_file", validation_id, filename)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected preview file path to be rejected")


def test_validate_trial_preview_returns_result_without_db_event(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset_validation_success"
    _write_file(
        dataset_root / "data.yaml",
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: part",
            ]
        ),
    )
    service = OrchestratorService(db_path=":memory:")
    experiment_id = _create_experiment(service, tmp_path, dataset_root)
    run_dir = tmp_path / "runs" / "validation_success"
    _write_file(run_dir / "weights" / "best.pt", "fake")
    service.repo.create_trial(
        TrialRecord(
            trial_id="trial_validation_success",
            display_name="validation_success",
            experiment_id=experiment_id,
            iteration=1,
            params={"imgsz": 224, "batch": 8},
            status="COMPLETED",
            run_dir=str(run_dir),
        )
    )

    captured_run_kwargs = {}

    def fake_run(*args, **kwargs):
        del args
        captured_run_kwargs.update(kwargs)
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"metrics":{"map50_95":0.42,"map50":0.8,"precision":0.7,"recall":0.6},"images":[{"filename":"0001_a.jpg"}]}\n',
            stderr="",
        )

    monkeypatch.setattr("backend.service.subprocess.run", fake_run)

    result = service.validate_trial_preview("trial_validation_success", image_limit=3, conf=0.2)

    assert result["trial_id"] == "trial_validation_success"
    assert result["split"] == "val"
    assert result["image_limit"] == 3
    assert result["conf"] == 0.2
    assert result["metrics"]["map50_95"] == 0.42
    assert result["images"][0]["filename"] == "0001_a.jpg"
    assert result["validation_id"] == "current"
    assert service.repo.latest_event(experiment_id, "TRIAL_VALIDATION_PREVIEW") is None
    assert captured_run_kwargs["encoding"] == "utf-8"
    assert captured_run_kwargs["errors"] == "replace"


def test_clear_validation_preview_cache_deletes_only_preview_dirs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_clear_cache"
    _write_file(
        dataset_root / "data.yaml",
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: part",
            ]
        ),
    )
    service = OrchestratorService(db_path=":memory:")
    experiment_id = _create_experiment(service, tmp_path, dataset_root)
    run_dir = tmp_path / "runs" / "clear_cache"
    preview_file = run_dir / ".validation_previews" / "val_20260603_010203_000001" / "0001_a_label.jpg"
    keep_file = run_dir / "results.csv"
    _write_file(preview_file, "preview-cache")
    _write_file(keep_file, "epoch\n1\n")
    service.repo.create_trial(
        TrialRecord(
            trial_id="trial_clear_cache",
            display_name="clear_cache",
            experiment_id=experiment_id,
            iteration=1,
            params={"imgsz": 224, "batch": 8},
            status="COMPLETED",
            run_dir=str(run_dir),
        )
    )

    result = service.clear_validation_preview_cache()

    assert result["deleted_dirs"] == 1
    assert result["deleted_files"] == 1
    assert result["deleted_bytes"] == len("preview-cache")
    assert not (run_dir / ".validation_previews").exists()
    assert keep_file.exists()


def test_default_db_migrates_old_openclaw_filename(tmp_path: Path, monkeypatch) -> None:
    old_db = tmp_path / "openclaw_yolo_state.sqlite"
    old_db.write_bytes(b"legacy db")
    monkeypatch.chdir(tmp_path)

    class FakeRepository:
        def __init__(self, db_path: str) -> None:
            self.db_path = db_path

    monkeypatch.setattr("backend.service.Repository", FakeRepository)

    service = OrchestratorService()

    new_db = tmp_path / "yolo_state.sqlite"
    assert new_db.read_bytes() == b"legacy db"
    assert service.repo.db_path == str(new_db.resolve())


def test_repository_rebuilds_legacy_experiment_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE experiments (
                experiment_id TEXT PRIMARY KEY,
                description TEXT NOT NULL DEFAULT '',
                project TEXT NOT NULL DEFAULT '',
                session_key TEXT NOT NULL DEFAULT '',
                task_type TEXT NOT NULL,
                dataset_root TEXT NOT NULL,
                dataset_yaml TEXT NOT NULL,
                pretrained_model TEXT NOT NULL,
                save_root TEXT NOT NULL,
                goal_config TEXT NOT NULL,
                status TEXT NOT NULL,
                auto_iterate INTEGER NOT NULL,
                confirm_timeout INTEGER NOT NULL,
                initial_params TEXT NOT NULL,
                search_space TEXT NOT NULL,
                stop_conditions TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )

    repo = Repository(db_path)
    columns = {row[1] for row in sqlite3.connect(db_path).execute("PRAGMA table_info(experiments)").fetchall()}
    assert "auto_iterate" not in columns
    assert "confirm_timeout" not in columns
    assert "session_key" not in columns
    assert "goal_config" not in columns
    assert list(tmp_path.glob("legacy.sqlite.schema-backup-*"))

    repo.create_experiment(
        ExperimentConfig(
            experiment_id="exp_legacy_schema",
            description="legacy schema",
            project="le",
            task_type="detection",
            dataset_root=str(tmp_path),
            dataset_yaml=str(tmp_path / "data.yaml"),
            pretrained_model="yolo11n.pt",
            save_root=str(tmp_path / "runs"),
            status=STATE_READY,
            initial_params={"imgsz": 640},
            search_space=SEARCH_SPACE,
            stop_conditions=STOP_CONDITIONS,
        )
    )
    assert repo.get_experiment("exp_legacy_schema").description == "legacy schema"


def test_repository_migrates_goal_column_and_waiting_status(tmp_path: Path) -> None:
    db_path = tmp_path / "goal.sqlite"
    repo = Repository(db_path)
    repo.create_experiment(
        ExperimentConfig(
            experiment_id="exp_goal_migration",
            description="goal migration",
            project="go",
            task_type="detection",
            dataset_root=str(tmp_path),
            dataset_yaml=str(tmp_path / "data.yaml"),
            pretrained_model="yolo11n.pt",
            save_root=str(tmp_path / "runs"),
            status="WAITING_USER_CONFIRM",
            initial_params={},
            search_space=SEARCH_SPACE,
            stop_conditions=STOP_CONDITIONS,
        )
    )
    repo.create_trial(
        TrialRecord(
            trial_id="trial_goal_migration",
            display_name="trial_goal_migration",
            experiment_id="exp_goal_migration",
            iteration=1,
            params={},
            status="WAITING_USER_CONFIRM",
            run_dir=str(tmp_path / "run"),
        )
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE experiments ADD COLUMN goal_config TEXT NOT NULL DEFAULT '{}'")

    migrated = Repository(db_path)
    columns = {row[1] for row in sqlite3.connect(db_path).execute("PRAGMA table_info(experiments)").fetchall()}
    assert "goal_config" not in columns
    assert migrated.get_experiment("exp_goal_migration").status == "COMPLETED"
    assert migrated.get_trial("trial_goal_migration").status == "COMPLETED"


def test_run_trial_uses_yolo_python_for_training_worker(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset_train_python"
    _write_file(
        dataset_root / "data.yaml",
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: part",
            ]
        ),
    )
    service = OrchestratorService(db_path=":memory:")
    experiment_id = _create_experiment(service, tmp_path, dataset_root)
    captured_kwargs = {}

    def fake_run_training(**kwargs):
        captured_kwargs.update(kwargs)
        run_dir = Path(kwargs["run_dir"])
        _write_results(run_dir, map50_95=0.55)
        return {
            "run_dir": str(run_dir),
            "stdout_log": str(run_dir / "stdout.log"),
            "stderr_log": str(run_dir / "stderr.log"),
        }

    monkeypatch.setattr("backend.service._python_for_yolo", lambda: "D:/fake/yolo/python.exe")
    monkeypatch.setattr("backend.service.run_training", fake_run_training)

    result = service.run_trial(experiment_id)

    assert result["status"] == "COMPLETED"
    assert result["internal_status"] == "COMPLETED"
    assert captured_kwargs["python_executable"] == "D:/fake/yolo/python.exe"
    assert captured_kwargs["src_root"].endswith("src")


def test_cancel_task_does_not_relabel_completed_waiting_trial(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_cancel_completed"
    _write_file(dataset_root / "data.yaml", "train: images/train\nval: images/val\nnames: [part]\n")
    service = OrchestratorService(db_path=":memory:")
    experiment_id = _create_experiment(service, tmp_path, dataset_root)
    service.repo.update_experiment_status(experiment_id, STATE_WAITING)
    service.repo.create_trial(
        TrialRecord(
            trial_id="trial_cancel_completed",
            display_name="trial_cancel_completed",
            experiment_id=experiment_id,
            iteration=1,
            params={},
            status=STATE_WAITING,
            run_dir=str(tmp_path / "run"),
            metrics={"map50_95": 0.42},
        )
    )

    result = service.cancel_task(experiment_id)

    assert result["status"] == "COMPLETED"
    assert service.repo.get_experiment(experiment_id).status == STATE_WAITING
    assert service.repo.get_trial("trial_cancel_completed").status == STATE_WAITING


def test_successful_training_wins_cancel_completion_race(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset_cancel_race"
    _write_file(dataset_root / "data.yaml", "train: images/train\nval: images/val\nnames: [part]\n")
    service = OrchestratorService(db_path=":memory:")
    experiment_id = _create_experiment(service, tmp_path, dataset_root)

    def fake_run_training(**kwargs):
        service.repo.update_experiment_status(experiment_id, "CANCELLED")
        run_dir = Path(kwargs["run_dir"])
        _write_results(run_dir, map50_95=0.55)
        return {
            "run_dir": str(run_dir),
            "stdout_log": str(run_dir / "stdout.log"),
            "stderr_log": str(run_dir / "stderr.log"),
        }

    monkeypatch.setattr("backend.service.run_training", fake_run_training)

    result = service.run_trial(experiment_id)

    assert result["status"] == "COMPLETED"
    assert service.repo.list_trials(experiment_id)[0].status == STATE_COMPLETED


def test_run_trial_rejects_missing_dataset_yaml_before_creating_trial_dir(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_missing_yaml"
    _write_file(dataset_root / "data.yaml", "train: images/train\nval: images/val\nnames: [part]\n")
    service = OrchestratorService(db_path=":memory:")
    experiment_id = _create_experiment(service, tmp_path, dataset_root)
    (dataset_root / "data.yaml").unlink()

    try:
        service.run_trial(experiment_id)
    except FileNotFoundError as exc:
        assert "dataset yaml not found" in str(exc)
    else:
        raise AssertionError("expected missing dataset yaml to be rejected")

    experiment_run_dir = tmp_path / "runs" / "experiments" / experiment_id
    assert sorted(path.name for path in experiment_run_dir.iterdir()) == ["experiment.json"]
    assert "goal" not in service.get_experiment_detail(experiment_id)["experiment"]
    assert service.repo.list_trials(experiment_id) == []
