from __future__ import annotations

from typing import Any

from backend.constants import SEARCH_SPACE


def validate_param_value(name: str, value: Any) -> Any:
    if name not in SEARCH_SPACE:
        raise ValueError(f"param '{name}' is not in the search space")

    rule = SEARCH_SPACE[name]
    rule_type = rule["type"]

    if rule_type == "choice":
        if value not in rule["values"]:
            raise ValueError(f"invalid value for '{name}': {value}")
        return value

    if rule_type == "int":
        if isinstance(value, bool):
            raise ValueError(f"invalid value for '{name}': {value}")
        if not isinstance(value, int):
            raise ValueError(f"invalid value for '{name}': {value}")
        if value < int(rule["min"]) or value > int(rule["max"]):
            raise ValueError(f"invalid value for '{name}': {value}")
        step = rule.get("step")
        if step is not None and value % int(step) != 0:
            raise ValueError(f"invalid value for '{name}': {value}")
        return value

    if rule_type == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"invalid value for '{name}': {value}")
        float_value = float(value)
        if float_value < float(rule["min"]) or float_value > float(rule["max"]):
            raise ValueError(f"invalid value for '{name}': {value}")
        return float_value

    raise ValueError(f"unsupported search rule type for '{name}': {rule_type}")

