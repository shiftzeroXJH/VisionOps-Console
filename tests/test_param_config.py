from backend.constants import SEARCH_SPACE, TASK_BASELINES
from backend.core.baseline import build_initial_params
from backend.core.constraints import validate_param_value


def test_erasing_is_in_search_space() -> None:
    assert SEARCH_SPACE["erasing"] == {"type": "float", "min": 0.0, "max": 1.0}


def test_erasing_is_in_task_baseline() -> None:
    assert TASK_BASELINES["detection"]["erasing"] == 0.0


def test_erasing_can_be_used_in_initial_params() -> None:
    params = build_initial_params("detection", {"erasing": 0.25})
    assert params["erasing"] == 0.25


def test_erasing_validation_rejects_out_of_range_values() -> None:
    try:
        validate_param_value("erasing", 1.5)
    except ValueError as exc:
        assert "invalid value for 'erasing'" in str(exc)
    else:
        raise AssertionError("expected validate_param_value to reject erasing > 1.0")

