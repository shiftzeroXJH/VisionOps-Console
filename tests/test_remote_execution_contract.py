from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from backend.constants import SEARCH_SPACE, STATE_READY, STOP_CONDITIONS, TASK_BASELINES
from backend.models import ExperimentConfig, RemoteServer
from backend.service import OrchestratorService, ServiceError
from backend.core import remote_train_worker as worker
from test_remote_launch import _Client, _Sftp


@pytest.fixture
def setup_remote(tmp_path, monkeypatch):
    service = OrchestratorService(db_path=tmp_path / "state.sqlite")
    config = ExperimentConfig(
        experiment_id="exp_contract", description="test", project="test", task_type="obb",
        dataset_root=str(tmp_path), dataset_yaml=str(tmp_path / "data.yaml"),
        pretrained_model="local.pt", save_root=str(tmp_path / "runs"), status=STATE_READY,
        initial_params=dict(TASK_BASELINES["obb"]), search_space=SEARCH_SPACE,
        stop_conditions=STOP_CONDITIONS,
        remote_configs={"remote_001": {"dataset_yaml": "/data/old.yaml", "pretrained_model": "/remote/remote-model.pt"}},
    )
    service.repo.create_experiment(config)
    service.repo.create_remote_server(RemoteServer(
        remote_server_id="remote_001", name="test", host="host", port=22, username="user",
        auth_type="password", password="secret", remote_python="/old/python", default_runs_root="/old/runs"))
    client, sftp = _Client(), _Sftp()
    monkeypatch.setattr(service, "_open_sftp", lambda _: (client, sftp))
    monkeypatch.setattr(service, "_analyze_remote_dataset", lambda *a: {})
    return service, config, client, sftp


def test_snapshot_is_offline_and_isolated(setup_remote, monkeypatch):
    service, config, client, sftp = setup_remote
    monkeypatch.setattr(service, "_open_sftp", lambda _: pytest.fail("prepare connected"))
    snapshot = service.prepare_remote_trial_request(config.experiment_id, "remote_001", note="queued")
    assert "secret" not in json.dumps(snapshot)
    assert snapshot["note"] == "queued" and isinstance(snapshot["pretrained"], str)
    config.remote_configs = {}
    config.save_root = "changed"
    monkeypatch.setattr(service.repo, "get_experiment", lambda _: pytest.fail("launch read changed config"))
    server = service.repo.get_remote_server("remote_001")
    server.remote_python = "/changed/python"
    server.default_runs_root = "/changed/runs"
    server.password = "latest"
    monkeypatch.setattr(service.repo, "get_remote_server", lambda _: server)
    def connect(current):
        assert current.password == "latest"
        return client, sftp
    monkeypatch.setattr(service, "_open_sftp", connect)
    events = []
    def prepared(identifier):
        assert service.repo.get_trial(identifier).remote_run_dir.endswith(identifier)
        assert not client.commands and not sftp.uploads
        events.append("prepared")
    def launching():
        assert sftp.uploads and not any("nohup" in c for c in client.commands)
        events.append("launching")
    result = service.launch_remote_trial(config.experiment_id, remote_server_id="remote_001",
        prepared_request=snapshot, trial_id="fixed_trial", on_trial_prepared=prepared, on_launch_attempt=launching)
    assert result["remote_run_dir"] == "/old/runs/experiments/exp_contract/fixed_trial"
    assert events == ["prepared", "launching"]
    request = json.loads((Path(service.repo.get_trial("fixed_trial").run_dir) / "request.json").read_text())
    assert request["dataset_yaml"] == "/data/old.yaml" and request["trial_id"] == "fixed_trial"
    assert any("nohup /old/python" in c for c in client.commands)


def test_preparing_reuses_trial_and_launching_never_retries(setup_remote, monkeypatch):
    service, config, client, sftp = setup_remote
    snapshot = service.prepare_remote_trial_request(config.experiment_id, "remote_001")
    monkeypatch.setattr(service, "_open_sftp", lambda _: (_ for _ in ()).throw(OSError("connect")))
    kwargs = dict(remote_server_id="remote_001", prepared_request=snapshot, trial_id="fixed")
    with pytest.raises(OSError):
        service.launch_remote_trial(config.experiment_id, **kwargs)
    original = service.repo.get_trial("fixed")
    monkeypatch.setattr(service, "_open_sftp", lambda _: (client, sftp))
    original_exec = service._exec_remote
    def execute(c, command, **opts):
        if "nohup" in command:
            raise OSError("ambiguous launch")
        return original_exec(c, command, **opts)
    monkeypatch.setattr(service, "_exec_remote", execute)
    monkeypatch.setattr(service, "_remove_remote_tree", lambda *a: pytest.fail("deleted evidence"))
    phases = []
    monkeypatch.setattr(service, "_analyze_remote_dataset", lambda *a: {"launch_snapshot": True})
    def attempted():
        assert service.repo.get_trial("fixed").dataset_analysis == {"launch_snapshot": True}
        phases.append("launching")
    with pytest.raises(OSError):
        service.launch_remote_trial(config.experiment_id, **kwargs, on_launch_attempt=attempted)
    assert phases == ["launching"]
    assert len(service.repo.list_trials(config.experiment_id)) == 1
    assert service.repo.get_trial("fixed").run_dir == original.run_dir
    with pytest.raises(ServiceError, match="launch attempt"):
        service.launch_remote_trial(config.experiment_id, **kwargs)


@pytest.mark.parametrize("value", [0, 65, True, 1.5, "2.5", None])
def test_parallel_invalid(value):
    with pytest.raises(ServiceError):
        OrchestratorService._validate_remote_parallel(value)


@pytest.mark.parametrize("state,alive,locked,expected", [
    ("completed", True, True, "running"), ("completed", True, False, "running"),
    ("completed", False, False, "completed"), ("failed", False, False, "failed"),
    ("running", False, False, "failed"), ("completed", False, True, "unknown"),
])
def test_worker_process_and_terminal_evidence(tmp_path, monkeypatch, state, alive, locked, expected):
    identity = {"pid": 123, "starttime": "100", "boot_id": "boot"}
    marker = {"trial_id": "fixed", "identity": identity}
    (tmp_path / "started.json").write_text(json.dumps(marker))
    (tmp_path / "status.json").write_text(json.dumps({**marker, "status": state}))
    (tmp_path / "worker.lock").touch()
    def flock(*args):
        if locked:
            raise BlockingIOError()
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(flock=flock, LOCK_EX=1, LOCK_NB=2))
    monkeypatch.setattr(worker, "_identity", lambda _: identity if alive else {**identity, "starttime": "999"})
    assert worker.check_status(tmp_path, "fixed")["state"] == expected
    assert worker.check_status(tmp_path, "other")["state"] == "unknown"
    (tmp_path / "status.json").unlink()
    assert worker.check_status(tmp_path, "fixed")["state"] == "unknown"


def test_status_connection_error_is_unknown(setup_remote, monkeypatch):
    service, config, _, _ = setup_remote
    service.launch_remote_trial(config.experiment_id, remote_server_id="remote_001", trial_id="fixed")
    monkeypatch.setattr(service, "_open_ssh", lambda *a, **kw: (_ for _ in ()).throw(OSError("offline")))
    assert service.check_remote_trial_status("fixed") == {"state": "unknown", "error": "offline"}


def test_server_parallel_roundtrip(setup_remote):
    service, _, _, _ = setup_remote
    updated = service.update_remote_server("remote_001", max_parallel_training_tasks=64)
    assert updated["remote_server"]["max_parallel_training_tasks"] == 64
    assert service.list_remote_servers()["remote_servers"][0]["max_parallel_training_tasks"] == 64
    created = service.create_remote_server(name="second", host="host", username="user",
        auth_type="password", password="secret", remote_python="python", default_runs_root="/runs",
        max_parallel_training_tasks=2)
    assert created["remote_server"]["max_parallel_training_tasks"] == 2


def test_duplicate_worker_marker_prevents_training(tmp_path, monkeypatch):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"run_dir": str(tmp_path), "trial_id": "fixed"}))
    (tmp_path / "started.json").write_text('{}')
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(flock=lambda *a: None, LOCK_EX=1, LOCK_NB=2))
    monkeypatch.setattr(sys, "argv", ["worker", str(request)])
    monkeypatch.setattr(worker, "_identity", lambda *a: pytest.fail("duplicate started"))
    assert worker.main() == 0


def test_completed_worker_marker_prevents_training(tmp_path, monkeypatch):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"run_dir": str(tmp_path), "trial_id": "fixed"}))
    (tmp_path / "status.json").write_text('{"status":"completed"}')
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(flock=lambda *a: None, LOCK_EX=1, LOCK_NB=2))
    monkeypatch.setattr(sys, "argv", ["worker", str(request)])
    monkeypatch.setattr(worker, "_identity", lambda *a: pytest.fail("duplicate started"))
    assert worker.main() == 0


def test_linux_worker_runs_only_once(tmp_path):
    if not sys.platform.startswith("linux"):
        pytest.skip("real Linux flock/proc integration")
    import os
    import subprocess
    import time
    (tmp_path / "ultralytics.py").write_text(
        "from pathlib import Path\nclass YOLO:\n"
        " def __init__(self,*a): pass\n"
        " def train(self,**kw):\n"
        "  with Path('calls').open('a') as f: f.write('call\\n')\n"
        "  import time; time.sleep(0.2)\n")
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"run_dir": str(tmp_path), "trial_id": "fixed",
        "params": {}, "pretrained_model": "fake", "dataset_yaml": "fake"}))
    command = [sys.executable, worker.__file__, str(request)]
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    first = subprocess.Popen(command, env=env)
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "status.json").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        subprocess.run(command, env=env, check=True, timeout=5)
        assert first.wait(timeout=5) == 0
        assert worker.check_status(tmp_path, "fixed")["state"] == "completed"
        subprocess.run(command, env=env, check=True, timeout=5)
        assert (tmp_path / "calls").read_text().splitlines() == ["call"]
    finally:
        if first.poll() is None:
            first.kill()
            first.wait()


def test_serial_changed_dataset_reuses_shared_copy(setup_remote, tmp_path, monkeypatch):
    service, config, _, sftp = setup_remote
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    yaml = dataset / "data.yaml"
    yaml.write_text("names: [part]\ntrain: images/train\nval: images/val\n")
    snapshot = service.prepare_remote_trial_request(config.experiment_id, "remote_001")
    snapshot.update(upload_local_dataset=True, dataset_root="", dataset_yaml="",
                    local_dataset_root=str(dataset), local_dataset_yaml=str(yaml))
    uploaded_roots = []
    refreshed = []
    monkeypatch.setattr(service, "_refresh_managed_remote_dataset", lambda *a: refreshed.append(a[3]))
    monkeypatch.setattr(service, "_upload_local_dataset", lambda s, root, y, remote: uploaded_roots.append(remote))
    for identifier in ("first", "same"):
        service.launch_remote_trial(config.experiment_id, remote_server_id="remote_001",
                                    prepared_request=snapshot, trial_id=identifier)
    assert len(uploaded_roots) == 1
    yaml.write_text("names: [changed]\ntrain: images/train\nval: images/val\n")
    service.launch_remote_trial(config.experiment_id, remote_server_id="remote_001",
                                prepared_request=snapshot, trial_id="changed")
    assert len(uploaded_roots) == 2 and uploaded_roots[0] == uploaded_roots[1]
    assert refreshed == [uploaded_roots[0]]


@pytest.mark.parametrize("remote_status,prior,expected", [
    ("failed", "TRAINING", "FAILED"), ("completed", "TRAINING", "COMPLETED"),
    (None, "FAILED", "FAILED"), (None, "COMPLETED", "COMPLETED"),
    ("running", "FAILED", "FAILED"),
])
def test_sync_terminal_without_results(setup_remote, monkeypatch, remote_status, prior, expected):
    service, config, _, _ = setup_remote
    service.launch_remote_trial(config.experiment_id, remote_server_id="remote_001", trial_id="fixed")
    service.repo.update_trial("fixed", status=prior)
    def download(sftp, remote, local):
        if remote.endswith("status.json") and remote_status is not None:
            Path(local).write_text(json.dumps({"status": remote_status, "trial_id": "fixed"}))
        else:
            raise OSError("missing artifact")
    monkeypatch.setattr(service, "_download_remote_file", download)
    monkeypatch.setattr(service, "_download_top_level_visualizations", lambda *a: None)
    result = service.sync_remote_trial("fixed")
    assert result["internal_status"] == expected
    assert service.repo.get_trial("fixed").status == expected
    if remote_status in {"failed", "completed"}:
        assert result["sync_error"]


def test_completed_sync_parse_failure_is_still_completed(setup_remote, monkeypatch):
    import backend.service as service_module
    service, config, _, sftp = setup_remote
    service.launch_remote_trial(config.experiment_id, remote_server_id="remote_001", trial_id="fixed")
    def download(sftp, remote, local):
        if remote.endswith("status.json"):
            Path(local).write_text('{"status":"completed","trial_id":"fixed"}')
        elif remote.endswith("results.csv"):
            Path(local).write_text("bad csv")
        else:
            raise OSError("missing")
    sftp.paths.add("/old/runs/experiments/exp_contract/fixed/results.csv")
    monkeypatch.setattr(service, "_download_remote_file", download)
    monkeypatch.setattr(service, "_download_top_level_visualizations", lambda *a: None)
    monkeypatch.setattr(service_module, "_valid_epoch_count", lambda *a: 1)
    monkeypatch.setattr(service_module, "build_summary", lambda *a: (_ for _ in ()).throw(ValueError("broken metrics")))
    result = service.sync_remote_trial("fixed")
    assert result["internal_status"] == "COMPLETED"
    assert "parse failed" in result["sync_error"]


def test_status_deadline_closes_client_and_returns_unknown(setup_remote, monkeypatch):
    import backend.service as service_module
    service, config, _, _ = setup_remote
    service.launch_remote_trial(config.experiment_id, remote_server_id="remote_001", trial_id="fixed")
    closed = []
    client = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(service, "_open_ssh", lambda *a, **kw: client)
    class ImmediateTimer:
        def __init__(self, seconds, callback):
            assert seconds == 10.0
            self.callback = callback
        def start(self):
            self.callback()
        def cancel(self):
            pass
    monkeypatch.setattr(service_module.threading, "Timer", ImmediateTimer)
    monkeypatch.setattr(service, "_exec_remote", lambda *a: '{"state":"completed","error":""}')
    assert service.check_remote_trial_status("fixed") == {"state": "unknown", "error": "remote status check timed out"}
    assert closed


def test_worker_permission_error_and_incomplete_identity_are_unknown(tmp_path, monkeypatch):
    identity = {"pid": 123, "starttime": "100", "boot_id": "boot"}
    marker = {"trial_id": "fixed", "identity": identity}
    (tmp_path / "started.json").write_text(json.dumps(marker))
    (tmp_path / "status.json").write_text(json.dumps({**marker, "status": "running"}))
    (tmp_path / "worker.lock").touch()
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(flock=lambda *a: None, LOCK_EX=1, LOCK_NB=2))
    monkeypatch.setattr(worker, "_identity", lambda *a: (_ for _ in ()).throw(PermissionError("proc")))
    assert worker.check_status(tmp_path, "fixed")["state"] == "unknown"
    marker["identity"].pop("starttime")
    (tmp_path / "started.json").write_text(json.dumps(marker))
    (tmp_path / "status.json").write_text(json.dumps({**marker, "status": "running"}))
    monkeypatch.setattr(worker, "_identity", lambda *a: pytest.fail("incomplete identity used"))
    assert worker.check_status(tmp_path, "fixed")["state"] == "unknown"


@pytest.mark.parametrize("state,process,expected", [
    ("completed", "absent", "completed"),
    ("failed", "absent", "failed"),
    ("running", "matching", "running"),
    ("running", "absent", "unknown"),
    ("running", "wrong_worker", "unknown"),
    ("running", "wrong_request", "unknown"),
    ("running", "permission", "unknown"),
    ("running", "cmdline_permission", "unknown"),
    ("completed", "matching", "unknown"),
    ("completed", "permission", "unknown"),
])
def test_legacy_managed_recovery(tmp_path, monkeypatch, state, process, expected):
    (tmp_path / "status.json").write_text(json.dumps({"status": state, "pid": 123, "error": "legacy error"}))
    (tmp_path / "request.json").write_text(json.dumps({"run_dir": str(tmp_path)}))
    original_stat, original_read = Path.stat, Path.read_bytes
    proc = Path("/proc")
    def stat(path, *args, **kwargs):
        if path == proc:
            return SimpleNamespace()
        if path == proc / "123":
            if process == "absent":
                raise FileNotFoundError("dead")
            if process == "permission":
                raise PermissionError("proc")
            return SimpleNamespace()
        return original_stat(path, *args, **kwargs)
    def read(path):
        if path == proc / "123" / "cmdline":
            if process == "cmdline_permission":
                raise PermissionError("cmdline")
            worker_path = tmp_path / ("other.py" if process == "wrong_worker" else "remote_train_worker.py")
            request_path = tmp_path / ("other.json" if process == "wrong_request" else "request.json")
            import os
            return b"python\x00" + os.fsencode(str(worker_path)) + b"\x00" + os.fsencode(str(request_path)) + b"\x00"
        return original_read(path)
    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(Path, "read_bytes", read)
    before = sorted(p.name for p in tmp_path.iterdir())
    result = worker.check_status(tmp_path, "legacy_trial")
    assert result["state"] == expected
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    if expected in {"completed", "failed"}:
        assert result["error"] == "legacy error"


@pytest.mark.parametrize("case", ["no_pid", "boolean_pid", "no_status", "no_request", "wrong_run_dir",
                                  "new_request", "new_identity", "new_trial_id", "invalid_marker"])
def test_legacy_missing_or_new_format_evidence_is_unknown(tmp_path, monkeypatch, case):
    status = {"status": "completed", "pid": 123}
    request = {"run_dir": str(tmp_path)}
    if case == "no_pid":
        status.pop("pid")
    elif case == "boolean_pid":
        status["pid"] = True
    elif case == "wrong_run_dir":
        request["run_dir"] = str(tmp_path / "another")
    elif case == "new_request":
        request["trial_id"] = "fixed"
    elif case == "new_identity":
        status["identity"] = {}
    elif case == "new_trial_id":
        status["trial_id"] = "fixed"
    elif case == "invalid_marker":
        (tmp_path / "started.json").write_text("broken JSON")
    if case != "no_status":
        (tmp_path / "status.json").write_text(json.dumps(status))
    if case != "no_request":
        (tmp_path / "request.json").write_text(json.dumps(request))
    original_stat = Path.stat
    def stat(path, *args, **kwargs):
        if path == Path("/proc") or path == Path("/proc/123"):
            pytest.fail("insufficient evidence must not inspect a process")
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", stat)
    assert worker.check_status(tmp_path, "fixed")["state"] == "unknown"


class _OwnedDatasetSftp(_Sftp):
    def __init__(self):
        super().__init__()
        self.directories = {"/", "/runs", "/runs/experiments", "/runs/experiments/exp_contract",
                            "/runs/experiments/exp_contract/dataset", "/runs/experiments/exp_contract/dataset/nested"}
        self.manifest = "/runs/experiments/exp_contract/dataset.manifest.json"
        self.root = "/runs/experiments/exp_contract/dataset"
        self.paths.update(self.directories | {self.manifest, self.root + "/nested/deleted.txt", "/outside.txt"})
        self.texts[self.manifest] = "a" * 64
        self.links = set()
        self.removed = []
        self.redirect = None

    def normalize(self, path):
        return self.redirect or path

    def lstat(self, path):
        import stat
        if path not in self.paths:
            raise FileNotFoundError(path)
        mode = stat.S_IFLNK if path in self.links else stat.S_IFDIR if path in self.directories else stat.S_IFREG
        return SimpleNamespace(st_mode=mode)

    def listdir_attr(self, directory):
        import posixpath
        return [SimpleNamespace(filename=posixpath.basename(p)) for p in sorted(self.paths)
                if posixpath.dirname(p) == directory]

    def remove(self, path):
        assert path.startswith(self.root + "/")
        self.removed.append(path)
        self.paths.remove(path)

    def rmdir(self, path):
        assert path.startswith(self.root + "/")
        self.removed.append(path)
        self.paths.remove(path)
        self.directories.remove(path)

    def mkdir(self, path):
        super().mkdir(path)
        self.directories.add(path)


def test_managed_dataset_refresh_clears_only_owned_root_and_can_retry(setup_remote):
    service, _, _, _ = setup_remote
    sftp = _OwnedDatasetSftp()
    # A symlink is removed as an entry, never traversed.
    sftp.paths.add(sftp.root + "/external_link")
    sftp.links.add(sftp.root + "/external_link")
    service._refresh_managed_remote_dataset(sftp, "/runs", "exp_contract", sftp.root, "a" * 64)
    assert sftp.root in sftp.paths and "/outside.txt" in sftp.paths
    assert not any(p.startswith(sftp.root + "/") for p in sftp.paths)
    assert sftp.texts[sftp.manifest] == "refreshing:" + "a" * 64
    # Interrupted upload leaves a marker that forces the next preparation to refresh.
    sftp.paths.add(sftp.root + "/partial.txt")
    service._refresh_managed_remote_dataset(sftp, "/runs", "exp_contract", sftp.root, sftp.texts[sftp.manifest])
    assert sftp.root + "/partial.txt" not in sftp.paths


@pytest.mark.parametrize("case", ["other_root", "traversal", "relative", "bad_hash", "redirect", "root_link", "manifest_link"])
def test_managed_dataset_refresh_rejects_unproven_paths(setup_remote, case):
    service, _, _, _ = setup_remote
    sftp = _OwnedDatasetSftp()
    root, work, digest = sftp.root, "/runs", "a" * 64
    if case == "other_root": root = "/outside"
    elif case == "traversal": work = "/runs/../runs"
    elif case == "relative": work = "runs"
    elif case == "bad_hash": digest = "unowned"
    elif case == "redirect": sftp.redirect = "/elsewhere"
    elif case == "root_link": sftp.links.add(root)
    elif case == "manifest_link": sftp.links.add(sftp.manifest)
    with pytest.raises(ServiceError):
        service._refresh_managed_remote_dataset(sftp, work, "exp_contract", root, digest)
    assert not sftp.removed
    assert sftp.texts[sftp.manifest] == "a" * 64


def test_changed_dataset_upload_removes_deleted_files_and_publishes_hash_last(setup_remote, tmp_path, monkeypatch):
    service, config, client, _ = setup_remote
    sftp = _OwnedDatasetSftp()
    monkeypatch.setattr(service, "_open_sftp", lambda _: (client, sftp))
    dataset = tmp_path / "local"
    dataset.mkdir()
    (dataset / "data.yaml").write_text("names: [part]\ntrain: labels\nval: labels\n")
    (dataset / "current.txt").write_text("metadata only")
    snapshot = service.prepare_remote_trial_request(config.experiment_id, "remote_001")
    snapshot.update(upload_local_dataset=True, local_dataset_root=str(dataset),
                    local_dataset_yaml=str(dataset / "data.yaml"), default_runs_root="/runs")
    original_upload = service._upload_remote_text
    def publish(s, text, path):
        if path == sftp.manifest and not text.startswith("refreshing:"):
            assert sftp.root + "/current.txt" in sftp.paths
            assert "/runs/experiments/exp_contract/data.remote.yaml" in sftp.paths
            assert sftp.root + "/nested/deleted.txt" not in sftp.paths
        original_upload(s, text, path)
    monkeypatch.setattr(service, "_upload_remote_text", publish)
    service.launch_remote_trial(config.experiment_id, remote_server_id="remote_001", prepared_request=snapshot, trial_id="fresh")
    assert len(sftp.texts[sftp.manifest]) == 64
    assert "/outside.txt" in sftp.paths


def test_managed_refresh_failure_retains_invalidated_cache_for_retry(setup_remote, monkeypatch):
    service, _, _, _ = setup_remote
    sftp = _OwnedDatasetSftp()
    monkeypatch.setattr(sftp, "remove", lambda path: (_ for _ in ()).throw(PermissionError("cannot remove")))
    with pytest.raises(PermissionError):
        service._refresh_managed_remote_dataset(sftp, "/runs", "exp_contract", sftp.root, "a" * 64)
    assert sftp.texts[sftp.manifest] == "refreshing:" + "a" * 64
    assert sftp.root + "/nested/deleted.txt" in sftp.paths
