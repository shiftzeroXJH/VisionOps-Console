from backend.constants import SEARCH_SPACE, TASK_BASELINES
from backend.core.baseline import build_initial_params
from backend.core.constraints import validate_param_value


def test_erasing_is_not_a_fixed_parameter() -> None:
    assert "erasing" not in SEARCH_SPACE


def test_erasing_is_not_in_task_baseline() -> None:
    assert "erasing" not in TASK_BASELINES["detection"]


def test_legacy_erasing_default_is_ignored_in_initial_params() -> None:
    params = build_initial_params("detection", {"erasing": 0.25})
    assert "erasing" not in params


def test_erasing_is_not_validated_as_a_fixed_parameter() -> None:
    try:
        validate_param_value("erasing", 0.5)
    except ValueError as exc:
        assert "not in the search space" in str(exc)
    else:
        raise AssertionError("expected erasing to be extra-only")

