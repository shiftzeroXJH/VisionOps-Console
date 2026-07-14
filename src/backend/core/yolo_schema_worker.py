from __future__ import annotations

import json


def _value_type(value: object) -> str:
    if value is None:
        return "json"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return "json"


def main() -> int:
    from ultralytics.cfg import DEFAULT_CFG_DICT

    schema = {
        name: {"type": _value_type(value)}
        for name, value in sorted(DEFAULT_CFG_DICT.items())
    }
    print(json.dumps(schema, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
