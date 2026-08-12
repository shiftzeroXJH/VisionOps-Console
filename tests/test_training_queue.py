from __future__ import annotations

from threading import Event, Lock
from time import monotonic, sleep
from typing import Any

import pytest

from backend.constants import (
    MAX_PARALLEL_TRAINING_SETTING_KEY,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_READY,
)
from backend.core.baseline import build_initial_params
from backend.models import ExperimentConfig, TrainingTask, TrialRecord
from backend.service import OrchestratorService
from backend.training_queue import (
    QUEUE_STATUS_FAILED,
    QUEUE_STATUS_QUEUED,
    QUEUE_STATUS_RUNNING,
    TrainingCapacityError,
    TrainingQueue,
)
from backend.utils import utc_now_iso


def _create_experiment(service: OrchestratorService, tmp_path, experiment_id: str) -> None:
    service.repo.create_experiment(
        ExperimentConfig(
            experiment_id=experiment_id,
            description=experiment_id,
            project="project",
            task_type="detection",
            dataset_root=str(tmp_path),
            dataset_yaml=str(tmp_path / "data.yaml"),
            pretrained_model="missing-ok.pt",
            save_root=str(tmp_path / "runs"),
            status=STATE_READY,
            initial_params=build_initial_params("detection", {}),
            search_space={},
            stop_conditions={},
        )
    )


def _wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


def test_queue_limits_parallelism_and_starts_next_automatically(tmp_path) -> None:
    service = OrchestratorService(db_path=tmp_path / "queue.sqlite")
    _create_experiment(service, tmp_path, "exp_a")
    _create_experiment(service, tmp_path, "exp_b")
    entered = {"exp_a": Event(), "exp_b": Event()}
    release = {"exp_a": Event(), "exp_b": Event()}
    concurrency_lock = Lock()
    current = 0
    maximum = 0

    def fake_run(experiment_id: str, **kwargs: Any) -> dict[str, Any]:
        nonlocal current, maximum
        kwargs["on_trial_started"](f"trial_{experiment_id}")
        with concurrency_lock:
            current += 1
            maximum = max(maximum, current)
        entered[experiment_id].set()
        assert release[experiment_id].wait(3)
        with concurrency_lock:
            current -= 1
        service.repo.update_experiment_status(experiment_id, STATE_COMPLETED)
        return {"internal_status": STATE_COMPLETED, "trial_id": f"trial_{experiment_id}"}

    queue = TrainingQueue(service, run_trial=fake_run)
    queue.start()
    first = queue.submit("exp_a", params=None, pretrained=None, note=None, reason=None, enqueue_if_busy=False)
    assert first["disposition"] == "started"
    assert entered["exp_a"].wait(1)

    with pytest.raises(TrainingCapacityError):
        queue.submit("exp_b", params=None, pretrained=None, note=None, reason=None, enqueue_if_busy=False)
    second = queue.submit("exp_b", params=None, pretrained=None, note=None, reason=None, enqueue_if_busy=True)
    assert second["disposition"] == "queued"

    release["exp_a"].set()
    assert entered["exp_b"].wait(2)
    snapshot = queue.list_tasks()
    assert snapshot["running"][0]["experiment_id"] == "exp_b"
    assert snapshot["queued_count"] == 0
    assert maximum == 1
    release["exp_b"].set()
    _wait_until(lambda: queue.list_tasks()["running_count"] == 0)


def test_same_experiment_is_serial_but_other_experiment_can_use_free_slot(tmp_path) -> None:
    service = OrchestratorService(db_path=tmp_path / "same-experiment.sqlite")
    service.repo.set_setting(MAX_PARALLEL_TRAINING_SETTING_KEY, "2")
    _create_experiment(service, tmp_path, "exp_a")
    _create_experiment(service, tmp_path, "exp_b")
    releases = [Event(), Event(), Event()]
    entered: list[str] = []
    entered_lock = Lock()

    def fake_run(experiment_id: str, **kwargs: Any) -> dict[str, Any]:
        with entered_lock:
            index = len(entered)
            entered.append(experiment_id)
        kwargs["on_trial_started"](f"trial_{index}")
        assert releases[index].wait(3)
        service.repo.update_experiment_status(experiment_id, STATE_COMPLETED)
        return {"internal_status": STATE_COMPLETED, "trial_id": f"trial_{index}"}

    queue = TrainingQueue(service, run_trial=fake_run)
    queue.start()
    queue.submit("exp_a", params=None, pretrained=None, note=None, reason=None, enqueue_if_busy=False)
    _wait_until(lambda: len(entered) == 1)
    second_a = queue.submit("exp_a", params=None, pretrained=None, note=None, reason=None, enqueue_if_busy=False)
    queue.submit("exp_b", params=None, pretrained=None, note=None, reason=None, enqueue_if_busy=False)
    _wait_until(lambda: len(entered) == 2)

    assert second_a["disposition"] == "queued"
    assert entered == ["exp_a", "exp_b"]
    releases[0].set()
    _wait_until(lambda: len(entered) == 3)
    assert entered[2] == "exp_a"
    releases[1].set()
    releases[2].set()
    _wait_until(lambda: queue.list_tasks()["running_count"] == 0)


def test_reorder_and_cancel_only_affect_waiting_tasks(tmp_path) -> None:
    service = OrchestratorService(db_path=tmp_path / "reorder.sqlite")
    for experiment_id in ("exp_a", "exp_b", "exp_c"):
        _create_experiment(service, tmp_path, experiment_id)
    queue = TrainingQueue(service)
    ids = []
    for experiment_id in ("exp_a", "exp_b", "exp_c"):
        result = queue.submit(
            experiment_id,
            params=None,
            pretrained=None,
            note=None,
            reason=None,
            enqueue_if_busy=True,
        )
        ids.append(result["training_task"]["queue_id"])

    queue.reorder(ids[2], 1)
    assert [item["experiment_id"] for item in queue.list_tasks()["queued"]] == ["exp_c", "exp_a", "exp_b"]
    queue.cancel(ids[0])
    assert [item["experiment_id"] for item in queue.list_tasks()["queued"]] == ["exp_c", "exp_b"]
    assert service.repo.get_experiment("exp_a").status == STATE_READY


def test_start_recovers_running_task_as_failed_and_keeps_waiting_task(tmp_path) -> None:
    service = OrchestratorService(db_path=tmp_path / "recovery.sqlite")
    _create_experiment(service, tmp_path, "exp_running")
    _create_experiment(service, tmp_path, "exp_waiting")
    service.repo.create_trial(
        TrialRecord(
            trial_id="trial_running",
            display_name="trial_running",
            experiment_id="exp_running",
            iteration=1,
            params={},
            status="TRAINING",
            run_dir=str(tmp_path / "run"),
        )
    )
    service.repo.create_training_task(
        TrainingTask(
            queue_id="queue_running",
            experiment_id="exp_running",
            params={},
            pretrained="missing-ok.pt",
            note="",
            reason="",
            status=QUEUE_STATUS_RUNNING,
            position=1,
            trial_id="trial_running",
            created_at=utc_now_iso(),
        )
    )
    service.repo.create_training_task(
        TrainingTask(
            queue_id="queue_waiting",
            experiment_id="exp_waiting",
            params={},
            pretrained="missing-ok.pt",
            note="",
            reason="",
            status=QUEUE_STATUS_QUEUED,
            position=2,
            created_at=utc_now_iso(),
        )
    )
    release = Event()

    def fake_run(experiment_id: str, **kwargs: Any) -> dict[str, Any]:
        assert experiment_id == "exp_waiting"
        assert release.wait(3)
        service.repo.update_experiment_status(experiment_id, STATE_COMPLETED)
        return {"internal_status": STATE_COMPLETED}

    queue = TrainingQueue(service, run_trial=fake_run)
    queue.start()
    _wait_until(lambda: service.repo.get_training_task("queue_waiting").status == QUEUE_STATUS_RUNNING)

    assert service.repo.get_training_task("queue_running").status == QUEUE_STATUS_FAILED
    assert service.repo.get_trial("trial_running").status == STATE_FAILED
    assert service.repo.get_experiment("exp_running").status == STATE_FAILED
    release.set()
    _wait_until(lambda: queue.list_tasks()["running_count"] == 0)


def test_increasing_parallel_setting_dispatches_waiting_task(tmp_path) -> None:
    service = OrchestratorService(db_path=tmp_path / "settings.sqlite")
    _create_experiment(service, tmp_path, "exp_a")
    _create_experiment(service, tmp_path, "exp_b")
    entered = {"exp_a": Event(), "exp_b": Event()}
    release = {"exp_a": Event(), "exp_b": Event()}

    def fake_run(experiment_id: str, **kwargs: Any) -> dict[str, Any]:
        entered[experiment_id].set()
        assert release[experiment_id].wait(3)
        service.repo.update_experiment_status(experiment_id, STATE_COMPLETED)
        return {"internal_status": STATE_COMPLETED}

    queue = TrainingQueue(service, run_trial=fake_run)
    queue.start()
    assert service.get_settings()["max_parallel_training_tasks"] == 1
    queue.submit("exp_a", params=None, pretrained=None, note=None, reason=None, enqueue_if_busy=False)
    assert entered["exp_a"].wait(1)
    queue.submit("exp_b", params=None, pretrained=None, note=None, reason=None, enqueue_if_busy=True)
    assert not entered["exp_b"].is_set()

    service.update_settings(max_parallel_training_tasks=2)
    queue.notify_settings_changed()
    assert entered["exp_b"].wait(1)
    assert queue.list_tasks()["running_count"] == 2

    release["exp_a"].set()
    release["exp_b"].set()
    _wait_until(lambda: queue.list_tasks()["running_count"] == 0)


def test_training_task_continuation_fields_round_trip(tmp_path) -> None:
    service = OrchestratorService(db_path=tmp_path / "continuation-fields.sqlite")
    _create_experiment(service, tmp_path, "exp_continue")
    service.repo.create_training_task(
        TrainingTask(
            queue_id="queue_continue",
            experiment_id="exp_continue",
            params={"epochs": 50, "lr0": 0.0001},
            pretrained=str(tmp_path / "last.pt"),
            note="",
            reason="Continue from parent",
            status=QUEUE_STATUS_QUEUED,
            position=1,
            parent_trial_id="trial_parent",
            training_mode="continued",
        )
    )

    task = service.repo.get_training_task("queue_continue")
    assert task.parent_trial_id == "trial_parent"
    assert task.training_mode == "continued"
