from io import BytesIO
from types import SimpleNamespace

import pytest

from backend.core.gpu_monitor import parse_gpu, query_gpu
from backend.models import RemoteServer
from backend.service import OrchestratorService


@pytest.mark.parametrize("used,free,total,percent", [(0, 100, 100, 0), (65, 35, 100, 65), (100, 0, 100, 100)])
def test_memory_percentage(used, free, total, percent):
    result = parse_gpu(f'"NVIDIA GPU, test", {used}, {free}, {total}\n')
    assert result["memory_used_percent"] == percent
    assert result["memory_free_mib"] == free
    assert result["gpu_name"] == "NVIDIA GPU, test"


@pytest.mark.parametrize("output", ["", "No devices were found", "GPU,N/A,1,2", "GPU,0,0,0", "GPU,-1,2,2", "GPU,3,0,2", "GPU,nan,0,2", "GPU,0,inf,2", "GPU,0,3,2", "GPU,0,1,2\nGPU,0,1,2"])
def test_invalid_memory(output):
    with pytest.raises(ValueError, match="无效"):
        parse_gpu(output)


class Client:
    def __init__(self, output=b"GPU, 25, 75, 100", error=b"", code=0):
        self.output, self.error, self.code = output, error, code
        self.closed = False

    def exec_command(self, command, timeout):
        assert "--id=0" in command and "utilization.memory" not in command
        assert timeout == 5
        stdout = BytesIO(self.output)
        stdout.channel = SimpleNamespace(recv_exit_status=lambda: self.code)
        return None, stdout, BytesIO(self.error)

    def close(self):
        self.closed = True


@pytest.mark.parametrize("error,output,code,message", [
    (b"command not found", b"", 127, "nvidia-smi"),
    (b"", b"No devices were found", 6, "GPU 0"),
    (b"driver failure", b"", 1, "驱动"),
])
def test_command_errors(error, output, code, message):
    with pytest.raises(ValueError, match=message):
        query_gpu(Client(output, error, code))


@pytest.fixture
def service(monkeypatch):
    service = OrchestratorService(db_path=":memory:")
    monkeypatch.setattr(service.repo, "get_remote_server", lambda _: object())
    return service


def test_service_success_closes_connection(service, monkeypatch):
    client = Client()
    monkeypatch.setattr(service, "_open_ssh", lambda server, timeout: client)
    result = service.get_remote_gpu_status("remote_001")
    assert result["status"] == "ok" and result["captured_at"]
    assert result["memory_used_percent"] == 25
    assert client.closed


@pytest.mark.parametrize("exception", [TimeoutError(), RuntimeError("secret password"), ValueError("GPU 返回的显存数据无效")])
def test_service_query_failures_close_connection(service, monkeypatch, exception):
    client = Client()
    monkeypatch.setattr(service, "_open_ssh", lambda server, timeout: client)
    def fail(_):
        raise exception
    monkeypatch.setattr("backend.core.gpu_monitor.query_gpu", fail)
    result = service.get_remote_gpu_status("remote_001")
    assert result["status"] == "unavailable"
    assert "memory_used_percent" not in result
    assert result["captured_at"] is None
    assert "secret" not in result["error"]
    assert client.closed


def test_command_deadline_closes_connection(monkeypatch):
    client = Client()
    class Timer:
        def __init__(self, seconds, callback):
            assert seconds == 5
            self.callback = callback
        def start(self):
            self.callback()
        def cancel(self):
            pass
    monkeypatch.setattr("backend.core.gpu_monitor.threading.Timer", Timer)
    with pytest.raises(TimeoutError):
        query_gpu(client)
    assert client.closed


@pytest.mark.parametrize("auth_type", ["key", "password"])
def test_ssh_auth_timeout_and_cleanup(service, monkeypatch, auth_type):
    import paramiko
    client = Client()
    client.set_missing_host_key_policy = lambda _: None
    def connect(**kwargs):
        assert kwargs["timeout"] == kwargs["banner_timeout"] == kwargs["auth_timeout"] == 5
        assert ("key_filename" if auth_type == "key" else "password") in kwargs
        raise RuntimeError("authentication failed: secret")
    client.connect = connect
    monkeypatch.setattr(paramiko, "SSHClient", lambda: client)
    server = RemoteServer("id", "test", "host", 22, "user", auth_type, private_key_path="key", password="secret")
    monkeypatch.setattr(service.repo, "get_remote_server", lambda _: server)
    result = service.get_remote_gpu_status("id")
    assert result["status"] == "unavailable" and "secret" not in result["error"]
    assert client.closed
