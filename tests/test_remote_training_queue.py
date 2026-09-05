from __future__ import annotations

import json
from threading import Event
from time import monotonic, sleep

import pytest

from backend.constants import SEARCH_SPACE, STATE_READY, STOP_CONDITIONS, TASK_BASELINES
from backend.models import ExperimentConfig, RemoteServer, TrainingTask, TrialRecord
from backend.service import OrchestratorService
from backend.training_queue import TrainingQueue


def wait_for(predicate):
    deadline = monotonic() + 4
    while monotonic() < deadline:
        if predicate():
            return
        sleep(.01)
    raise AssertionError("queue did not reach expected state")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    service = OrchestratorService(db_path=tmp_path / "queues.sqlite")
    for experiment_id in ("a", "b", "c"):
        service.repo.create_experiment(ExperimentConfig(
            experiment_id, experiment_id, "project", "obb", str(tmp_path), str(tmp_path / "data.yaml"),
            "model.pt", str(tmp_path / "runs"), STATE_READY, dict(TASK_BASELINES["obb"]), SEARCH_SPACE, STOP_CONDITIONS,
        ))
    for server_id in ("s1", "s2"):
        service.repo.create_remote_server(RemoteServer(server_id, server_id, "host", 22, "user", "password", password="secret"))
    launches, states = [], {}

    def prepare(experiment_id, **kwargs):
        return {"params": dict(kwargs.get("params") or {"epochs": 5}), "pretrained": "model.pt", "note": "", "path": "/data"}

    def launch(experiment_id, **kwargs):
        trial_id = kwargs["trial_id"]
        try:
            service.repo.get_trial(trial_id)
        except KeyError:
            service.repo.create_trial(TrialRecord(
                trial_id, trial_id, experiment_id, 1, kwargs["params"], "TRAINING", str(tmp_path / trial_id),
                source="remote_sftp", remote_server_id=kwargs["remote_server_id"], remote_run_dir=f"/runs/{trial_id}",
            ))
        kwargs["on_trial_prepared"](trial_id)
        kwargs["on_launch_attempt"]()
        states[trial_id] = "running"
        launches.append((experiment_id, kwargs["remote_server_id"], trial_id))
        return {"trial_id": trial_id}

    monkeypatch.setattr(service, "prepare_remote_trial_request", prepare, raising=False)
    monkeypatch.setattr(service, "launch_remote_trial", launch)
    monkeypatch.setattr(service, "check_remote_trial_status", lambda trial_id: {"state": states.get(trial_id, "unknown"), "error": "test failure"}, raising=False)
    queue = TrainingQueue(service)
    yield service, queue, launches, states
    queue.stop()


def submit(queue, experiment="a", server="s1", key=None):
    return queue.submit_remote(experiment, remote_server_id=server, idempotency_key=key)["training_task"]


def test_independent_targets_serial_experiments_and_capacity_changes(setup):
    service, queue, launches, states = setup
    queue.start()
    first = submit(queue)
    same = submit(queue)
    other = submit(queue, "b")
    remote2 = submit(queue, "a", "s2")
    wait_for(lambda: len(launches) == 2)
    assert queue.list_tasks()["running_count"] == 0  # legacy local counts unchanged
    assert queue.list_tasks()["total_running_count"] == 2
    service.repo.update_remote_server("s1", max_parallel_training_tasks=2)
    queue.notify_settings_changed()
    wait_for(lambda: len(launches) == 3)
    assert service.repo.get_training_task(same["queue_id"]).status == "QUEUED"
    service.repo.update_remote_server("s1", max_parallel_training_tasks=1)
    queue.notify_settings_changed()
    assert queue.list_tasks()["groups"][1 if queue.list_tasks()["groups"][1]["target_id"] == "s1" else 2]["running_count"] == 2
    states[first["trial_id"]] = "completed"
    wait_for(lambda: service.repo.get_training_task(first["queue_id"]).phase == "running")
    queue.recheck(first["queue_id"])
    assert service.repo.get_training_task(same["queue_id"]).status == "QUEUED"
    states[other["trial_id"]] = "failed"
    wait_for(lambda: service.repo.get_training_task(other["queue_id"]).phase == "running")
    queue.recheck(other["queue_id"])
    wait_for(lambda: len(launches) == 4)
    assert service.repo.get_experiment("a").status == "TRAINING"
    assert service.repo.get_training_task(remote2["queue_id"]).status == "RUNNING"


def test_unknown_blocks_only_its_server_and_recheck_releases(setup):
    service, queue, launches, states = setup
    queue.start()
    first = submit(queue)
    wait_for(lambda: service.repo.get_training_task(first["queue_id"]).phase == "running")
    states[first["trial_id"]] = "unknown"
    queue.recheck(first["queue_id"])
    waiting = submit(queue, "b")
    service.repo.update_remote_server("s1", max_parallel_training_tasks=2)
    queue.notify_settings_changed()
    other = submit(queue, "c", "s2")
    wait_for(lambda: len(launches) == 2)
    assert service.repo.get_training_task(waiting["queue_id"]).status == "QUEUED"
    group = next(g for g in queue.list_tasks()["groups"] if g["target_id"] == "s1")
    assert group["blocked"] and "状态待确认" in group["queued"][0]["waiting_reason"]
    states[first["trial_id"]] = "completed"
    queue.recheck(first["queue_id"])
    wait_for(lambda: len(launches) == 3)
    assert service.repo.get_training_task(other["queue_id"]).status == "RUNNING"


def test_target_scoped_order_cancel_dedup_and_persisted_snapshot(setup):
    service, queue, _, _ = setup
    a = submit(queue, key="same-key")
    assert submit(queue, key="same-key")["queue_id"] == a["queue_id"]
    b = submit(queue, "b")
    c = submit(queue, "c", "s2")
    original = service.repo.get_training_task(a["queue_id"]).request_snapshot
    assert "secret" not in json.dumps(original)
    queue.reorder(b["queue_id"], 1)
    groups = {g["target_id"]: g for g in queue.list_tasks()["groups"]}
    assert [t["queue_id"] for t in groups["s1"]["queued"]] == [b["queue_id"], a["queue_id"]]
    assert groups["s2"]["queued"][0]["queue_id"] == c["queue_id"]
    queue.cancel(b["queue_id"])
    assert service.repo.get_training_task(b["queue_id"]).status == "CANCELLED"
    assert service.repo.get_training_task(a["queue_id"]).request_snapshot == original


def test_ambiguous_launch_is_not_retried_on_restart(setup, monkeypatch):
    service, queue, launches, states = setup
    def ambiguous(experiment_id, **kwargs):
        kwargs["on_launch_attempt"]()
        launches.append(kwargs["trial_id"])
        raise OSError("SSH response lost")
    monkeypatch.setattr(service, "launch_remote_trial", ambiguous)
    queue.start()
    first = submit(queue)
    wait_for(lambda: service.repo.get_training_task(first["queue_id"]).phase == "unknown")
    waiting = submit(queue, "b")
    queue.stop()
    restored = TrainingQueue(service)
    try:
        restored.start()
        assert len(launches) == 1
        assert service.repo.get_training_task(waiting["queue_id"]).status == "QUEUED"
        assert restored.list_tasks()["total_running_count"] == 1
    finally:
        restored.stop()


def test_restart_before_launch_reuses_reserved_trial_id(setup):
    service, queue, launches, _ = setup
    task = submit(queue)
    service.repo.update_training_task(task["queue_id"], status="RUNNING", phase="preparing")
    queue.start()
    wait_for(lambda: len(launches) == 1)
    assert launches[0][2] == task["trial_id"]
    assert len(service.repo.list_training_tasks()) == 1
    assert len(service.repo.list_trials("a")) == 1


def test_preparation_failure_advances_and_stop_keeps_waiting(setup, monkeypatch):
    service, queue, launches, states = setup
    original = service.launch_remote_trial
    def failing(experiment_id, **kwargs):
        if experiment_id == "a":
            raise ValueError("dataset missing")
        return original(experiment_id, **kwargs)
    monkeypatch.setattr(service, "launch_remote_trial", failing)
    first = submit(queue)
    second = submit(queue, "b")
    queue.start()
    wait_for(lambda: len(launches) == 1)
    assert service.repo.get_training_task(first["queue_id"]).status == "FAILED"
    third = submit(queue, "c")
    queue.stop()
    states[second["trial_id"]] = "completed"
    wait_for(lambda: service.repo.get_training_task(second["queue_id"]).phase == "running")
    queue.recheck(second["queue_id"])
    assert service.repo.get_training_task(third["queue_id"]).status == "QUEUED"


def test_legacy_active_remote_trial_adopted_once(setup):
    service, queue, launches, states = setup
    service.repo.create_trial(TrialRecord("legacy", "legacy", "a", 1, {}, "TRAINING", "/local",
                                         source="remote_sftp", remote_server_id="s1", remote_run_dir="/remote"))
    queue.start()
    submit(queue, "b")
    assert len(launches) == 0
    assert len([t for t in service.repo.list_training_tasks() if t.trial_id == "legacy"]) == 1
    queue.stop()
    restored = TrainingQueue(service)
    try:
        restored.start()
        assert len([t for t in service.repo.list_training_tasks() if t.trial_id == "legacy"]) == 1
    finally:
        restored.stop()


def test_background_monitor_advances_without_any_page_requests(setup, monkeypatch):
    service, queue, launches, states = setup
    monkeypatch.setattr("backend.training_queue.REMOTE_POLL_INTERVAL_SECONDS", .02)
    first = submit(queue)
    second = submit(queue, "b")
    queue.start()
    wait_for(lambda: len(launches) == 1)
    states[first["trial_id"]] = "completed"
    wait_for(lambda: len(launches) == 2)
    assert service.repo.get_training_task(first["queue_id"]).status == "COMPLETED"
    assert service.repo.get_training_task(second["queue_id"]).status == "RUNNING"


def test_local_and_remote_do_not_share_slots(setup, monkeypatch):
    service, queue, launches, _ = setup
    release = Event()
    monkeypatch.setattr(service, "prepare_trial_request", lambda *args, **kwargs: {"params": {}, "pretrained": "model.pt", "note": "", "reason": ""})
    def local_run(*args, **kwargs):
        release.wait(3)
        return {"internal_status": "COMPLETED"}
    queue._run_trial_callback = local_run
    queue.start()
    try:
        queue.submit("a", params=None, pretrained=None, note=None, reason=None, enqueue_if_busy=True)
        submit(queue, "a", "s1")
        submit(queue, "a", "s2")
        wait_for(lambda: len(launches) == 2)
        result = queue.list_tasks()
        assert result["running_count"] == 1 and result["total_running_count"] == 3
    finally:
        release.set()
        wait_for(lambda: queue.list_tasks()["running_count"] == 0)


def test_cancel_and_atomic_claim_have_single_winner(setup):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    service, queue, _, _ = setup
    task = submit(queue)
    gate = Barrier(2)
    def claim():
        gate.wait()
        return service.repo.claim_training_task(task["queue_id"], 1)
    def cancel():
        gate.wait()
        return service.repo.cancel_queued_training_task(task["queue_id"])
    with ThreadPoolExecutor(2) as pool:
        claimed, cancelled = pool.submit(claim), pool.submit(cancel)
        assert sum((claimed.result(), cancelled.result())) == 1
    assert service.repo.get_training_task(task["queue_id"]).status in {"RUNNING", "CANCELLED"}


def test_legacy_schema_migration_preserves_tasks_and_defaults(tmp_path):
    import sqlite3
    from backend.db.repository import Repository
    db_path = tmp_path / "legacy.sqlite"
    repo = Repository(db_path)
    repo.create_remote_server(RemoteServer("server", "server", "host", 22, "user", "password"))
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE remote_servers DROP COLUMN max_parallel_training_tasks")
        for column in ("source", "remote_server_id", "request_snapshot_json", "phase"):
            conn.execute(f"ALTER TABLE training_tasks DROP COLUMN {column}")
    migrated = Repository(db_path)
    assert migrated.get_remote_server("server").max_parallel_training_tasks == 1
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(training_tasks)")}
    assert {"source", "remote_server_id", "request_snapshot_json", "phase"} <= columns


def test_queue_api_remote_submission_groups_and_server_settings(setup, monkeypatch, tmp_path):
    monkeypatch.setenv("YOLO_DB_PATH", str(tmp_path / "api.sqlite"))
    from backend import api as api_module
    from fastapi.testclient import TestClient
    service, queue, launches, states = setup
    monkeypatch.setattr(api_module, "service", service)
    monkeypatch.setattr(api_module, "training_queue", queue)
    # No lifespan here: both requests remain queued and must be deduplicated.
    client = TestClient(api_module.app)
    body = {"remote_server_id": "s1", "idempotency_key": "api-submit"}
    first = client.post("/api/experiments/a/trials/remote-run", json=body)
    assert first.status_code == 200 and first.json()["disposition"] == "queued"
    repeated = client.post("/api/experiments/a/trials/remote-run", json=body)
    queue_id = first.json()["training_task"]["queue_id"]
    assert repeated.json()["training_task"]["queue_id"] == queue_id
    groups = client.get("/api/training-tasks").json()
    assert groups["total_queued_count"] == 1 and groups["queued_count"] == 0
    assert groups["groups"][0]["source"] == "local"
    assert len(launches) == 0
    assert client.post(f"/api/training-tasks/{queue_id}/recheck").status_code == 200
    assert client.post(f"/api/training-tasks/{queue_id}/cancel").status_code == 200
    assert client.patch("/api/remote-servers/s1", json={"max_parallel_training_tasks": 0}).status_code == 400
    # Existing settings require execution paths; preserve them for this valid edit.
    service.repo.update_remote_server("s1", remote_python="/python", default_runs_root="/runs")
    edited = client.patch("/api/remote-servers/s1", json={"max_parallel_training_tasks": 2})
    assert edited.status_code == 200
    assert edited.json()["remote_server"]["max_parallel_training_tasks"] == 2
    assert "password" not in edited.json()["remote_server"]
