from __future__ import annotations

import pytest

from backend.constants import STATE_COMPLETED
from backend.core.baseline import build_initial_params
from backend.models import ExperimentConfig, TrialRecord
from backend.service import OrchestratorService, ServiceError, TemplateNameConflictError


def _create_trial(
    service: OrchestratorService,
    tmp_path,
    *,
    experiment_id: str,
    trial_id: str,
    task_type: str = "detection",
    params: dict | None = None,
) -> TrialRecord:
    service.repo.create_experiment(
        ExperimentConfig(
            experiment_id=experiment_id,
            description=experiment_id,
            project="project",
            task_type=task_type,
            dataset_root=str(tmp_path),
            dataset_yaml=str(tmp_path / "data.yaml"),
            pretrained_model="model.pt",
            save_root=str(tmp_path / "runs"),
            status=STATE_COMPLETED,
            initial_params=build_initial_params(task_type, {}),
            search_space={},
            stop_conditions={},
        )
    )
    trial = TrialRecord(
        trial_id=trial_id,
        display_name=trial_id,
        experiment_id=experiment_id,
        iteration=1,
        params=params or build_initial_params(task_type, {}),
        status=STATE_COMPLETED,
        run_dir=str(tmp_path / "runs" / trial_id),
    )
    service.repo.create_trial(trial)
    return trial


def test_template_uses_trimmed_name_and_exact_trial_params(tmp_path) -> None:
    service = OrchestratorService(db_path=tmp_path / "templates.sqlite")
    params = build_initial_params("detection", {"epochs": 42})
    params["close_mosaic"] = 5
    _create_trial(
        service,
        tmp_path,
        experiment_id="exp_source",
        trial_id="trial_source",
        params=params,
    )

    result = service.save_trial_hyperparameter_template("trial_source", "  AOI baseline  ")

    assert result["overwritten"] is False
    assert result["template"]["name"] == "AOI baseline"
    assert result["template"]["params"] == params
    assert result["template"]["source_trial_id"] == "trial_source"
    assert result["template"]["source_task_type"] == "detection"
    assert service.list_hyperparameter_templates()["templates"] == [result["template"]]


def test_template_name_is_case_insensitive_and_explicit_overwrite_keeps_id(tmp_path) -> None:
    service = OrchestratorService(db_path=tmp_path / "overwrite.sqlite")
    first_params = build_initial_params("detection", {"epochs": 20})
    second_params = build_initial_params("segment", {"epochs": 80})
    _create_trial(
        service,
        tmp_path,
        experiment_id="exp_first",
        trial_id="trial_first",
        params=first_params,
    )
    _create_trial(
        service,
        tmp_path,
        experiment_id="exp_second",
        trial_id="trial_second",
        task_type="segment",
        params=second_params,
    )
    first = service.save_trial_hyperparameter_template("trial_first", "Shared")

    with pytest.raises(TemplateNameConflictError):
        service.save_trial_hyperparameter_template("trial_second", "shared")

    overwritten = service.save_trial_hyperparameter_template("trial_second", "shared", overwrite=True)
    assert overwritten["overwritten"] is True
    assert overwritten["template"]["template_id"] == first["template"]["template_id"]
    assert overwritten["template"]["params"] == second_params
    assert overwritten["template"]["source_task_type"] == "segment"
    assert len(service.list_hyperparameter_templates()["templates"]) == 1


@pytest.mark.parametrize("task_type", ["detection", "segment", "obb"])
def test_template_params_validate_in_each_supported_task_type(tmp_path, task_type: str) -> None:
    service = OrchestratorService(db_path=tmp_path / f"{task_type}.sqlite")
    source = _create_trial(
        service,
        tmp_path,
        experiment_id="exp_source",
        trial_id="trial_source",
    )
    _create_trial(
        service,
        tmp_path,
        experiment_id="exp_target",
        trial_id="trial_target",
        task_type=task_type,
    )
    saved = service.save_trial_hyperparameter_template(source.trial_id, "cross-task")

    validation = service.validate_params("exp_target", params=saved["template"]["params"])

    assert validation["valid"] is True
    assert validation["normalized_params"] == source.params


def test_incompatible_template_is_rejected_by_target_experiment_validation(tmp_path, monkeypatch) -> None:
    service = OrchestratorService(db_path=tmp_path / "invalid.sqlite")
    params = build_initial_params("detection", {})
    params["removed_in_future"] = True
    _create_trial(
        service,
        tmp_path,
        experiment_id="exp_source",
        trial_id="trial_source",
        params=params,
    )
    _create_trial(
        service,
        tmp_path,
        experiment_id="exp_target",
        trial_id="trial_target",
    )
    saved = service.save_trial_hyperparameter_template("trial_source", "old-template")
    monkeypatch.setattr(service, "_extra_param_schema", lambda: {})

    validation = service.validate_params("exp_target", params=saved["template"]["params"])

    assert validation["valid"] is False
    assert validation["errors"]["removed_in_future"] == "unsupported parameter"


def test_template_survives_source_trial_deletion_and_can_be_deleted_explicitly(tmp_path) -> None:
    service = OrchestratorService(db_path=tmp_path / "delete.sqlite")
    _create_trial(
        service,
        tmp_path,
        experiment_id="exp_source",
        trial_id="trial_source",
    )
    saved = service.save_trial_hyperparameter_template("trial_source", "keep-me")
    service.repo.delete_trial("trial_source")

    assert service.list_hyperparameter_templates()["templates"][0]["params"] == saved["template"]["params"]
    assert service.delete_hyperparameter_template(saved["template"]["template_id"])["deleted"] is True
    assert service.list_hyperparameter_templates()["templates"] == []

    with pytest.raises(ServiceError, match="not found"):
        service.delete_hyperparameter_template(saved["template"]["template_id"])


@pytest.mark.parametrize("name", ["", "   ", "x" * 81])
def test_template_name_validation(tmp_path, name: str) -> None:
    service = OrchestratorService(db_path=tmp_path / "names.sqlite")
    _create_trial(
        service,
        tmp_path,
        experiment_id="exp_source",
        trial_id="trial_source",
    )

    with pytest.raises(ServiceError):
        service.save_trial_hyperparameter_template("trial_source", name)
