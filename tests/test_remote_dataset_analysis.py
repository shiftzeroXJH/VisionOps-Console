from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from backend.constants import SEARCH_SPACE, STATE_READY, STOP_CONDITIONS, TASK_BASELINES
from backend.models import ExperimentConfig, RemoteServer, TrialRecord
from backend.service import OrchestratorService


def test_remote_analyzer_runs_shared_code_with_unicode_paths(tmp_path, monkeypatch):
    dataset = tmp_path / "数据集"
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
        (dataset / "images" / split / "sample.jpg").write_bytes(b"")
        (dataset / "labels" / split / "sample.txt").write_text("0 .5 .5 .1 .1\n")
    yaml = dataset / "data.yaml"
    yaml.write_text("train: images/train\nval: images/val\nnames: [零件]\n", encoding="utf-8")
    service = OrchestratorService(db_path=":memory:")

    def execute(_client, command):
        args = shlex.split(command)
        assert args[:2] == ["/remote/python", "-c"]
        return subprocess.check_output([sys.executable, "-c", args[2]], text=True, encoding="utf-8")

    monkeypatch.setattr(service, "_exec_remote", execute)
    result = service._analyze_remote_dataset(None, "/remote/python", str(yaml), str(tmp_path))
    assert result["status"] == "completed"
    assert result["totals"]["total_instances"] == 2
    assert result["classes"][0]["class_name"] == "零件"


@pytest.mark.parametrize("snapshot", [
    {"totals": {"total_instances": 999}},
    {"source": "remote", "status": "completed", "totals": {"total_instances": 7}},
    {},
    {"source": "remote", "status": "failed", "warnings": ["unavailable at launch"]},
])
def test_sync_preserves_historical_snapshot_without_reanalysis(tmp_path, monkeypatch, snapshot):
    service = OrchestratorService(db_path=":memory:")
    service.repo.create_experiment(ExperimentConfig(
        experiment_id="exp", description="remote", project="test", task_type="obb",
        dataset_root="missing-local", dataset_yaml="missing-local/data.yaml",
        pretrained_model="model.pt", save_root=str(tmp_path), status=STATE_READY,
        initial_params=dict(TASK_BASELINES["obb"]), search_space=SEARCH_SPACE,
        stop_conditions=STOP_CONDITIONS,
    ))
    service.repo.create_remote_server(RemoteServer(
        remote_server_id="server", name="remote", host="host", port=22,
        username="user", auth_type="password", remote_python="/remote/python",
    ))
    service.repo.create_trial(TrialRecord(
        trial_id="trial", display_name="trial", experiment_id="exp", iteration=1,
        params=dict(TASK_BASELINES["obb"]), status="TRAINING", run_dir=str(tmp_path / "run"),
        source="remote_sftp", remote_server_id="server", remote_run_dir="/runs/trial",
        dataset_analysis=snapshot,
    ))

    class Connection:
        def close(self):
            pass

        def stat(self, path):
            raise OSError(path)

    monkeypatch.setattr(service, "_open_sftp", lambda server: (Connection(), Connection()))
    monkeypatch.setattr(service, "_download_top_level_visualizations", lambda *args: None)
    monkeypatch.setattr(service, "_remote_text", lambda *args: json.dumps({"dataset_yaml": "/actual/data.yaml"}))

    def download(sftp, remote, local):
        if remote.endswith("status.json"):
            Path(local).write_text('{"status":"running"}')
        else:
            raise OSError(remote)

    monkeypatch.setattr(service, "_download_remote_file", download)
    def unexpected_analysis(*args):
        pytest.fail("Refreshing a historical trial must never reanalyze its dataset")

    monkeypatch.setattr(service, "_analyze_remote_dataset", unexpected_analysis)
    monkeypatch.setattr(service, "_exec_remote", unexpected_analysis)
    for _ in range(2):
        service.sync_remote_trial("trial")
        assert service.repo.get_trial("trial").sync_status == "synced"
        assert service.repo.get_trial("trial").dataset_analysis == snapshot
        assert service.get_summary("trial")["dataset_analysis"] == snapshot


def test_remote_registration_does_not_require_local_dataset(tmp_path, monkeypatch):
    service = OrchestratorService(db_path=":memory:")
    service.repo.create_experiment(ExperimentConfig(
        experiment_id="exp", description="remote", project="test", task_type="obb",
        dataset_root="missing", dataset_yaml="missing/data.yaml", pretrained_model="model.pt",
        save_root=str(tmp_path), status=STATE_READY, initial_params=dict(TASK_BASELINES["obb"]),
        search_space=SEARCH_SPACE, stop_conditions=STOP_CONDITIONS,
    ))
    service.repo.create_remote_server(RemoteServer(
        remote_server_id="server", name="remote", host="host", port=22,
        username="user", auth_type="password",
    ))
    monkeypatch.setattr(service, "_read_remote_text", lambda *args: "model: model.pt\ndata: /remote/data.yaml\n")
    monkeypatch.setattr(service, "_extra_param_schema", lambda: {})
    result = service.register_remote_trial("exp", remote_server_id="server", remote_run_dir="/runs/trial")
    trial = service.repo.get_trial(result["trial_id"])
    assert trial.dataset_analysis == {}
    assert "data: /remote/data.yaml" in (Path(trial.run_dir) / "args.yaml").read_text(encoding="utf-8")
