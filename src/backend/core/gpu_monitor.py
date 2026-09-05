"""One-shot NVIDIA GPU memory query; no remote agent or persistent state."""
from __future__ import annotations

import csv
import math
import threading
from typing import Any

COMMAND = "nvidia-smi --id=0 --query-gpu=name,memory.used,memory.free,memory.total --format=csv,noheader,nounits"


def query_gpu(client: Any) -> dict[str, Any]:
    expired = threading.Event()

    def timeout() -> None:
        expired.set()
        client.close()

    timer = threading.Timer(5, timeout)
    timer.daemon = True
    timer.start()
    try:
        _, stdout, stderr = client.exec_command(COMMAND, timeout=5)
        output = stdout.read().decode("utf-8", errors="replace")
        error = stderr.read().decode("utf-8", errors="replace")
        code = stdout.channel.recv_exit_status()
        if expired.is_set():
            raise TimeoutError()
        if code != 0:
            if code == 127 or "command not found" in error.lower():
                raise ValueError("服务器未安装或无法找到 nvidia-smi")
            if "no devices" in (output + error).lower() or "not found" in output.lower():
                raise ValueError("未找到 GPU 0")
            raise ValueError("GPU 查询失败，请检查 NVIDIA 驱动和访问权限")
        return parse_gpu(output)
    except Exception as exc:
        if expired.is_set():
            raise TimeoutError() from exc
        raise
    finally:
        timer.cancel()


def parse_gpu(output: str) -> dict[str, Any]:
    rows = list(csv.reader(output.strip().splitlines(), skipinitialspace=True))
    try:
        if len(rows) != 1 or len(rows[0]) != 4:
            raise ValueError()
        name, *values = rows[0]
        used, free, total = map(float, values)
        if not name.strip() or not all(math.isfinite(v) for v in (used, free, total)):
            raise ValueError()
        if total <= 0 or not 0 <= used <= total or not 0 <= free <= total:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError("GPU 返回的显存数据无效") from None
    return {"gpu_name": name.strip(), "memory_used_mib": used, "memory_free_mib": free,
            "memory_total_mib": total, "memory_used_percent": round(used / total * 100, 1)}
