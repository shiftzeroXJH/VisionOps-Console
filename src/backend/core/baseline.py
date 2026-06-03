from __future__ import annotations

from typing import Any

from backend.constants import SEARCH_SPACE, TASK_BASELINES
from backend.core.constraints import validate_param_value


class BaselineError(ValueError):
    pass


def build_initial_params(task_type: str, overrides: dict[str, Any]) -> dict[str, Any]:
    if task_type not in TASK_BASELINES:
        raise BaselineError(f"unsupported task type: {task_type}")

    params = dict(TASK_BASELINES[task_type])
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in SEARCH_SPACE:
            raise BaselineError(f"unsupported initial param: {key}")
        try:
            params[key] = validate_param_value(key, value)
        except ValueError as exc:
            raise BaselineError(str(exc)) from exc
    return params

