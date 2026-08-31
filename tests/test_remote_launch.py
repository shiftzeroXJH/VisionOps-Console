from __future__ import annotations

from pathlib import Path

from backend.constants import SEARCH_SPACE, STATE_READY, STOP_CONDITIONS, TASK_BASELINES
from backend.models import ExperimentConfig, RemoteServer
from backend.service import OrchestratorService


class _Stream:
    def __init__(self, value: str = "", code: int = 0) -> None:
        self.value = value
        self.channel = self
        self.code = code

    def recv_exit_status(self) -> int:
        return self.code

    def read(self) -> str:
        return self.value


class _Client:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def exec_command(self, command: str):
        self.commands.append(command)
        if command.startswith("test -r") and "remote-model.pt" in command:
            return None, _Stream("", 0), _Stream()
        if command.startswith("find "):
            return None, _Stream("/remote/data/data.yaml\n"), _Stream()
        return None, _Stream(), _Stream()

    def close(self) -> None:
        pass


class _Sftp:
    def __init__(self) -> None:
        self.paths = {"/"}
        self.uploads: list[tuple[str, str]] = []

    def stat(self, path: str):
        if path not in self.paths:
            raise OSError(path)
        return object()

    def mkdir(self, path: str) -> None:
        self.paths.add(path)

    def put(self, local: str, remote: str) -> None:
        self.uploads.append((local, remote))
        self.paths.add(remote)

    def close(self) -> None:
        pass


def test_launch_remote_trial_uploads_worker_and_persists_trial(tmp_path: Path, monkeypatch) -> None:
    service = OrchestratorService(db_path=tmp_path / "state.sqlite")
    experiment_id = "exp_remote"
    service.repo.create_experiment(
        ExperimentConfig(
            experiment_id=experiment_id,
            description="remote test",
            project="re",
            task_type="obb",
            dataset_root=str(tmp_path),
            dataset_yaml=str(tmp_path / "data.yaml"),
            pretrained_model=str(tmp_path / "local.pt"),
            save_root=str(tmp_path / "runs"),
            status=STATE_READY,
            initial_params=dict(TASK_BASELINES["obb"]),
            search_space=SEARCH_SPACE,
            stop_conditions=STOP_CONDITIONS,
            remote_configs={"remote_001": {"dataset_root": "/remote/data", "pretrained_model": "/remote/remote-model.pt"}},
        )
    )
    server = RemoteServer(
        remote_server_id="remote_001",
        name="test",
        host="host",
        port=22,
        username="user",
        auth_type="password",
        default_runs_root="/remote/runs",
        remote_python="/opt/python",
        password="secret",
    )
    service.repo.create_remote_server(server)
    client, sftp = _Client(), _Sftp()
    monkeypatch.setattr(service, "_open_sftp", lambda _server: (client, sftp))

    result = service.launch_remote_trial(experiment_id, remote_server_id="remote_001")

    assert result["status"] == "TRAINING"
    trial = service.repo.get_trial(result["trial_id"])
    assert trial.remote_run_dir == f"/remote/runs/experiments/{experiment_id}/{result['trial_id']}"
    uploaded_names = {Path(remote).name for _local, remote in sftp.uploads}
    assert {"request.json", "remote_train_worker.py", "yolo26n.pt"} <= uploaded_names
    assert any("remote_train_worker.py" in command and "nohup" in command for command in client.commands)
