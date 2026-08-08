from pathlib import Path
import json

import pytest

from backend.core.analyzer import build_summary
from backend.core.metrics import calculate_fitness
from backend.core.trainer import _TRAINING_HANDLES, _TRAINING_HANDLES_LOCK, _TrainingHandle, _unregister_training_process
from backend.constants import (
    STATE_ANALYZING,
    STATE_CANCELLED,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_INIT,
    STATE_QUEUED,
    STATE_READY,
    STATE_RETRAINING,
    STATE_TRAINING,
)
from backend.models import ExperimentConfig, TrialRecord
from backend.service import OrchestratorService, public_task_status


HEADER = ",".join(
    [
        "epoch",
        "time",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(M)",
        "metrics/recall(M)",
        "metrics/mAP50(M)",
        "metrics/mAP50-95(M)",
    ]
)


def _write_results(path: Path, rows: list[str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.csv").write_text(
        "\n".join([HEADER, *rows]) + "\n",
        encoding="utf-8",
    )


def test_segment_summary_uses_ultralytics_combined_fitness(tmp_path: Path) -> None:
    _write_results(
        tmp_path,
        [
            "1,10,0.8,0.8,0.8,0.60,0.8,0.8,0.8,0.80",
            "2,20,0.8,0.8,0.8,0.75,0.8,0.8,0.8,0.66",
            "3,30,0.8,0.8,0.8,0.70,0.8,0.8,0.8,0.70",
        ],
    )

    summary = build_summary("trial", "segment", str(tmp_path), {"epochs": 3})

    assert summary.basic_info["best_epoch"] == 2
    assert summary.metric_context["selection_metric"] == "mAP50-95(B) + mAP50-95(M)"
    assert summary.metric_context["selection_fitness"] == 1.41
    assert summary.final_metrics["map50_95"] == 0.66
    assert summary.metric_breakdown["box"]["map50_95"] == 0.75
    assert summary.basic_info["train_time_sec"] == 30.0
    assert summary.resource["avg_epoch_time"] == 10.0


def test_detection_summary_uses_box_fitness_only(tmp_path: Path) -> None:
    _write_results(
        tmp_path,
        [
            "1,10,0.8,0.8,0.8,0.60,0.8,0.8,0.8,0.95",
            "2,20,0.8,0.8,0.8,0.75,0.8,0.8,0.8,0.65",
        ],
    )

    summary = build_summary("trial", "detection", str(tmp_path), {"epochs": 2})

    assert summary.basic_info["best_epoch"] == 2
    assert summary.metric_context["selection_metric"] == "mAP50-95(B)"
    assert summary.metric_context["selection_fitness"] == 0.75
    assert summary.final_metrics["map50_95"] == 0.75


def test_fitness_profiles_and_missing_metrics() -> None:
    segment_row = {
        "metrics/mAP50-95(B)": "0.70",
        "metrics/mAP50-95(M)": "0.60",
    }
    detection_row = {
        "metrics/mAP50-95(B)": "0.70",
        "metrics/mAP50-95(M)": "0.99",
    }
    obb_row = {
        "metrics/mAP50-95(O)": "0.65",
        "metrics/mAP50-95(B)": "0.90",
    }

    assert calculate_fitness(segment_row, "segment") == pytest.approx(1.3)
    assert calculate_fitness(detection_row, "detection") == 0.7
    assert calculate_fitness(obb_row, "obb") == 0.65
    assert calculate_fitness({"metrics/mAP50-95(B)": "0.70"}, "segment") is None


def test_segment_summary_keeps_first_epoch_on_fitness_tie(tmp_path: Path) -> None:
    _write_results(
        tmp_path,
        [
            "1,10,0.8,0.8,0.8,0.70,0.8,0.8,0.8,0.70",
            "2,20,0.8,0.8,0.8,0.75,0.8,0.8,0.8,0.65",
        ],
    )

    summary = build_summary("trial", "segment", str(tmp_path), {"epochs": 2})

    assert summary.basic_info["best_epoch"] == 1
    assert summary.metric_context["selection_fitness"] == 1.4


def test_trial_row_exposes_fitness_without_changing_best_metric(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "final_metrics": {"map50_95": 0.75},
                "metric_context": {
                    "selection_fitness": 1.41,
                    "selection_metric": "mAP50-95(B) + mAP50-95(M)",
                },
            }
        ),
        encoding="utf-8",
    )
    trial = TrialRecord(
        trial_id="trial",
        display_name="trial",
        experiment_id="experiment",
        iteration=1,
        params={},
        status="COMPLETED",
        run_dir=str(tmp_path),
        summary_path=str(summary_path),
    )

    row = OrchestratorService(db_path=":memory:")._trial_row(trial)

    assert row["fitness"] == 1.41
    assert row["fitness_metric"] == "mAP50-95(B) + mAP50-95(M)"
    assert row["is_best"] is False


@pytest.mark.parametrize(
    ("internal_status", "expected"),
    [
        (STATE_INIT, "NOT_STARTED"),
        (STATE_READY, "NOT_STARTED"),
        (STATE_QUEUED, "QUEUED"),
        (STATE_TRAINING, "TRAINING"),
        (STATE_RETRAINING, "TRAINING"),
        (STATE_ANALYZING, "TRAINING"),
        (STATE_COMPLETED, "COMPLETED"),
        (STATE_CANCELLED, "INTERRUPTED_OR_FAILED"),
        (STATE_FAILED, "INTERRUPTED_OR_FAILED"),
    ],
)
def test_public_task_status_maps_internal_states(internal_status: str, expected: str) -> None:
    assert public_task_status(internal_status) == expected


def test_public_task_status_maps_remote_uncertainty_to_interrupted() -> None:
    assert public_task_status("WAITING_USER_CONFIRM", "maybe_stopped") == "INTERRUPTED_OR_FAILED"


def test_clean_training_exit_wins_cancel_race() -> None:
    class FinishedProcess:
        returncode = 0

    process_key = "test-clean-training-exit"
    with _TRAINING_HANDLES_LOCK:
        _TRAINING_HANDLES[process_key] = _TrainingHandle(
            process=FinishedProcess(),
            cancel_requested=True,
        )
    try:
        assert _unregister_training_process(process_key) is False
    finally:
        with _TRAINING_HANDLES_LOCK:
            _TRAINING_HANDLES.pop(process_key, None)


def test_experiment_curves_include_fitness_and_metric_formula(tmp_path: Path) -> None:
    service = OrchestratorService(db_path=":memory:")
    experiment = ExperimentConfig(
        experiment_id="experiment",
        description="experiment",
        project="project",
        task_type="segment",
        dataset_root=str(tmp_path),
        dataset_yaml=str(tmp_path / "data.yaml"),
        pretrained_model="model.pt",
        save_root=str(tmp_path),
        status=STATE_READY,
        initial_params={},
        search_space={},
        stop_conditions={},
    )
    service.repo.create_experiment(experiment)
    run_dir = tmp_path / "run"
    _write_results(run_dir, [
        "1,10,0.8,0.8,0.8,0.70,0.8,0.8,0.8,0.60",
    ])
    service.repo.create_trial(
        TrialRecord(
            trial_id="trial",
            display_name="trial",
            experiment_id="experiment",
            iteration=1,
            params={},
            status="COMPLETED",
            run_dir=str(run_dir),
        )
    )

    curves = service.get_experiment_curves("experiment")

    assert curves["fitness_metric"] == "mAP50-95(B) + mAP50-95(M)"
    assert curves["curves"]["trial"][0]["fitness"] == pytest.approx(1.3)


def test_comparison_exposes_fitness_but_keeps_map_best(tmp_path: Path) -> None:
    service = OrchestratorService(db_path=":memory:")
    experiment = ExperimentConfig(
        experiment_id="experiment",
        description="experiment",
        project="project",
        task_type="segment",
        dataset_root=str(tmp_path),
        dataset_yaml=str(tmp_path / "data.yaml"),
        pretrained_model="model.pt",
        save_root=str(tmp_path),
        status=STATE_READY,
        initial_params={},
        search_space={},
        stop_conditions={},
    )
    service.repo.create_experiment(experiment)

    trials = [
        ("trial-1", 0.80, 1.50),
        ("trial-2", 0.90, 1.40),
    ]
    for trial_id, map50_95, fitness in trials:
        summary_path = tmp_path / f"{trial_id}.json"
        summary_path.write_text(
            json.dumps(
                {
                    "final_metrics": {"map50_95": map50_95},
                    "metric_context": {
                        "selection_fitness": fitness,
                        "selection_metric": "mAP50-95(B) + mAP50-95(M)",
                    },
                }
            ),
            encoding="utf-8",
        )
        service.repo.create_trial(
            TrialRecord(
                trial_id=trial_id,
                display_name=trial_id,
                experiment_id="experiment",
                iteration=int(trial_id[-1]),
                params={},
                status="COMPLETED",
                run_dir=str(tmp_path),
                summary_path=str(summary_path),
            )
        )

    comparison = service.compare_experiment("experiment")
    rows = {row["trial_id"]: row for row in comparison["rows"]}

    assert "fitness" in {column["key"] for column in comparison["columns"]}
    assert comparison["fitness_metric"] == "mAP50-95(B) + mAP50-95(M)"
    assert rows["trial-1"]["fitness"] == 1.5
    assert rows["trial-1"]["is_best"] is False
    assert rows["trial-2"]["is_best"] is True
