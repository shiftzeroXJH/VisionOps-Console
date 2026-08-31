from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from backend.service import OrchestratorService, ServiceError, TemplateNameConflictError
from backend.workbench import WorkbenchError
from backend.jobs import JobStore
from backend.training_queue import TrainingCapacityError, TrainingQueue

job_store = JobStore()
service = OrchestratorService(db_path=os.environ.get("YOLO_DB_PATH"))
training_queue = TrainingQueue(service)


@asynccontextmanager
async def lifespan(_: FastAPI):
    training_queue.start()
    yield


app = FastAPI(title="yolo-platform", version="0.1.0", lifespan=lifespan)


def _invoke_sync(action: str, callback: Any) -> dict[str, Any]:
    try:
        return callback()
    except (ServiceError, WorkbenchError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc), "action": action}) from exc


def _invoke_async(kind: str, experiment_id: str, callback: Any) -> dict[str, Any]:
    job = job_store.start(kind, experiment_id, callback)
    return {
        "job_id": job.job_id,
        "kind": job.kind,
        "experiment_id": job.experiment_id,
        "status": job.status,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/experiments")
def list_experiments() -> dict[str, Any]:
    return _invoke_sync("list-experiments", service.list_experiments)


@app.get("/api/hyperparameter-templates")
def list_hyperparameter_templates() -> dict[str, Any]:
    return _invoke_sync("list-hyperparameter-templates", service.list_hyperparameter_templates)


@app.delete("/api/hyperparameter-templates/{template_id}")
def delete_hyperparameter_template(template_id: str) -> dict[str, Any]:
    return _invoke_sync(
        "delete-hyperparameter-template",
        lambda: service.delete_hyperparameter_template(template_id),
    )


@app.get("/api/workbench/models")
def list_workbench_models() -> dict[str, Any]:
    return _invoke_sync("list-workbench-models", service.workbench.list_models)


@app.post("/api/workbench/sessions")
def create_workbench_session() -> dict[str, Any]:
    return _invoke_sync("create-workbench-session", service.workbench.create_session)


@app.get("/api/workbench/sessions/{session_id}")
def get_workbench_session(session_id: str) -> dict[str, Any]:
    return _invoke_sync("get-workbench-session", lambda: service.workbench.get_session(session_id))


@app.delete("/api/workbench/sessions/{session_id}")
def delete_workbench_session(session_id: str) -> dict[str, Any]:
    return _invoke_sync("delete-workbench-session", lambda: service.workbench.delete_session(session_id))


@app.post("/api/workbench/sessions/{session_id}/images")
async def upload_workbench_images(
    session_id: str,
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    try:
        uploads = [(upload.filename or "image", upload.file) for upload in files]
        return _invoke_sync(
            "upload-workbench-images",
            lambda: service.workbench.add_images(session_id, uploads),
        )
    finally:
        for upload in files:
            await upload.close()


@app.delete("/api/workbench/sessions/{session_id}/images")
def delete_workbench_images(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload or {})
    return _invoke_sync(
        "delete-workbench-images",
        lambda: service.workbench.delete_images(session_id, list(body.get("image_ids") or [])),
    )


@app.patch("/api/workbench/sessions/{session_id}/images/{image_id}/roi")
def update_workbench_image_roi(session_id: str, image_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload or {})
    return _invoke_sync(
        "update-workbench-image-roi",
        lambda: service.workbench.set_image_roi(session_id, image_id, body.get("roi")),
    )


@app.post("/api/workbench/sessions/{session_id}/images/{image_id}/rotate")
def rotate_workbench_image(session_id: str, image_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload or {})
    return _invoke_sync(
        "rotate-workbench-image",
        lambda: service.workbench.rotate_image(session_id, image_id, str(body.get("direction", ""))),
    )


@app.post("/api/workbench/sessions/{session_id}/infer")
def infer_workbench_images(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload or {})
    return _invoke_async(
        "workbench-inference",
        session_id,
        lambda: service.workbench.infer(session_id, body),
    )


@app.get("/api/workbench/sessions/{session_id}/images/{image_id}/file")
def get_workbench_image(session_id: str, image_id: str) -> FileResponse:
    try:
        return FileResponse(service.workbench.image_path(session_id, image_id))
    except (WorkbenchError, OSError) as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc), "action": "get-workbench-image"}) from exc


@app.post("/api/workbench/datasets/inspect")
def inspect_workbench_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload or {})
    return _invoke_sync(
        "inspect-workbench-dataset",
        lambda: service.workbench.inspect_dataset(str(body.get("dataset_path", ""))),
    )


@app.post("/api/workbench/evaluations")
def create_workbench_evaluation(payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload or {})
    return _invoke_async(
        "workbench-evaluation",
        "workbench",
        lambda: service.workbench.evaluate(body),
    )


@app.get("/api/workbench/evaluations")
def list_workbench_evaluations(dataset_path: str) -> dict[str, Any]:
    return _invoke_sync(
        "list-workbench-evaluations",
        lambda: service.workbench.list_evaluations(dataset_path),
    )


@app.get("/api/workbench/evaluations/{evaluation_id}")
def get_workbench_evaluation(evaluation_id: str, dataset_path: str) -> dict[str, Any]:
    return _invoke_sync(
        "get-workbench-evaluation",
        lambda: service.workbench.get_evaluation(dataset_path, evaluation_id),
    )


@app.get("/api/workbench/evaluations/{evaluation_id}/images/{image_id}/file")
def get_workbench_evaluation_image(evaluation_id: str, image_id: str) -> FileResponse:
    try:
        return FileResponse(service.workbench.evaluation_image_path(evaluation_id, image_id))
    except (WorkbenchError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": str(exc), "action": "get-workbench-evaluation-image"},
        ) from exc


@app.post("/api/settings/clear-validation-cache")
def clear_validation_cache() -> dict[str, Any]:
    return _invoke_sync("clear-validation-cache", service.clear_validation_preview_cache)


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    return _invoke_sync("get-settings", service.get_settings)


@app.patch("/api/settings")
def update_settings(payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload or {})
    result = _invoke_sync(
        "update-settings",
        lambda: service.update_settings(
            yolo_python=body.get("yolo_python"),
            max_parallel_training_tasks=body.get("max_parallel_training_tasks"),
        ),
    )
    training_queue.notify_settings_changed()
    return result


@app.get("/api/training-tasks")
def list_training_tasks() -> dict[str, Any]:
    return _invoke_sync("list-training-tasks", training_queue.list_tasks)


@app.post("/api/training-tasks/{queue_id}/cancel")
def cancel_training_task(queue_id: str) -> dict[str, Any]:
    return _invoke_sync("cancel-training-task", lambda: training_queue.cancel(queue_id))


@app.patch("/api/training-tasks/{queue_id}")
def reorder_training_task(queue_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    if "position" not in body:
        raise HTTPException(
            status_code=400,
            detail={"error": "position is required", "action": "reorder-training-task"},
        )
    return _invoke_sync(
        "reorder-training-task",
        lambda: training_queue.reorder(queue_id, int(body["position"])),
    )


@app.get("/api/remote-servers")
def list_remote_servers() -> dict[str, Any]:
    return _invoke_sync("list-remote-servers", service.list_remote_servers)


@app.post("/api/remote-servers")
def create_remote_server(payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    return _invoke_sync(
        "create-remote-server",
        lambda: service.create_remote_server(
            name=body.get("name", ""),
            host=body["host"],
            port=int(body.get("port", 22)),
            username=body["username"],
            auth_type=body.get("auth_type", "key"),
            private_key_path=body.get("private_key_path"),
            password_ref=body.get("password_ref"),
            default_runs_root=body.get("default_runs_root"),
            remote_python=body.get("remote_python"),
            password=body.get("password"),
        ),
    )


@app.post("/api/remote-servers/{remote_server_id}/test")
def test_remote_server(remote_server_id: str) -> dict[str, Any]:
    return _invoke_sync("test-remote-server", lambda: service.test_remote_server(remote_server_id))


@app.patch("/api/remote-servers/{remote_server_id}")
def update_remote_server(remote_server_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload or {})
    return _invoke_sync("update-remote-server", lambda: service.update_remote_server(remote_server_id, **body))


@app.post("/api/experiments")
def create_experiment(payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    return _invoke_sync(
        "create-experiment",
        lambda: service.create_experiment(
            description=body.get("description", ""),
            project=body.get("project"),
            remote_configs=body.get("remote_configs"),
            task_type=body["task_type"],
            dataset_root=body["dataset_root"],
            dataset_yaml=body.get("dataset_yaml"),
            pretrained=body["pretrained"],
            save_root=body["save_root"],
            initial_params=body.get("initial_params"),
        ),
    )


@app.get("/api/projects/{project}")
def get_project_settings(project: str) -> dict[str, Any]:
    return _invoke_sync("get-project-settings", lambda: service.get_project_settings(project))


@app.patch("/api/projects/{project}")
def update_project_settings(project: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload or {})
    return _invoke_sync(
        "update-project-settings",
        lambda: service.update_project_settings(
            project,
            name=body.get("name"),
            default_export_dir=body.get("default_export_dir"),
        ),
    )


@app.delete("/api/projects/{project}")
def delete_project(project: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = dict(payload or {})
    return _invoke_sync(
        "delete-project",
        lambda: service.delete_project(project, confirmation=str(body.get("confirmation", ""))),
    )


@app.get("/api/experiments/{experiment_id}")
def get_experiment(experiment_id: str) -> dict[str, Any]:
    return _invoke_sync("get-experiment", lambda: service.get_experiment_detail(experiment_id))


@app.patch("/api/experiments/{experiment_id}")
def update_experiment(experiment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    return _invoke_sync(
        "update-experiment",
        lambda: service.update_experiment(
            experiment_id,
            description=body.get("description"),
            project=body.get("project"),
            dataset_root=body.get("dataset_root"),
            dataset_yaml=body.get("dataset_yaml"),
            pretrained_model=body.get("pretrained_model"),
            remote_configs=body.get("remote_configs"),
        ),
    )


@app.delete("/api/experiments/{experiment_id}")
def delete_experiment(experiment_id: str, keep_files: bool = True, force: bool = False) -> dict[str, Any]:
    return _invoke_sync(
        "delete-experiment",
        lambda: service.delete_task(experiment_id, keep_files=keep_files, force=force),
    )


@app.post("/api/experiments/{experiment_id}/cancel")
def cancel_experiment(experiment_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = dict(payload or {})
    return _invoke_sync("cancel-experiment", lambda: service.cancel_task(experiment_id, body.get("reason")))


@app.get("/api/experiments/{experiment_id}/comparison")
def compare_experiment(experiment_id: str) -> dict[str, Any]:
    return _invoke_sync("compare-experiment", lambda: service.compare_experiment(experiment_id))


@app.get("/api/experiments/{experiment_id}/params")
def get_experiment_params(experiment_id: str) -> dict[str, Any]:
    return _invoke_sync("get-experiment-params", lambda: service.get_param_metadata(experiment_id))


@app.post("/api/experiments/{experiment_id}/params/validate")
def validate_experiment_params(experiment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    return _invoke_sync(
        "validate-experiment-params",
        lambda: service.validate_params(
            experiment_id,
            params=body.get("params"),
            param_updates=body.get("param_updates"),
        ),
    )


@app.post("/api/experiments/{experiment_id}/trials/run")
def run_experiment_trial(experiment_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = dict(payload or {})
    try:
        return training_queue.submit(
            experiment_id,
            params=body.get("params"),
            pretrained=body.get("pretrained") or body.get("model"),
            note=body.get("note"),
            reason=body.get("reason"),
            enqueue_if_busy=bool(body.get("enqueue_if_busy", False)),
        )
    except TrainingCapacityError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": str(exc),
                "code": "TRAINING_CAPACITY_REACHED",
                "running_count": exc.running_count,
                "max_parallel_training_tasks": exc.max_parallel,
            },
        ) from exc


    except (ServiceError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": str(exc), "action": "run-experiment-trial"},
        ) from exc


@app.post("/api/experiments/{experiment_id}/trials/remote-run")
def run_remote_experiment_trial(experiment_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = dict(payload or {})
    return _invoke_sync(
        "run-remote-experiment-trial",
        lambda: service.launch_remote_trial(
            experiment_id,
            remote_server_id=body["remote_server_id"],
            params=body.get("params"),
            pretrained=body.get("pretrained") or body.get("model"),
            note=body.get("note"),
        ),
    )


@app.post("/api/experiments/{experiment_id}/trials/import")
def import_experiment_trial(experiment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    return _invoke_sync(
        "import-experiment-trial",
        lambda: service.import_run(
            experiment_id,
            run_dir=body["run_dir"],
            params=body.get("params"),
            pretrained=body.get("pretrained") or body.get("model"),
            note=body.get("note"),
        ),
    )


@app.post("/api/experiments/{experiment_id}/trials/remote-register")
def register_remote_trial(experiment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    return _invoke_sync(
        "register-remote-trial",
        lambda: service.register_remote_trial(
            experiment_id,
            remote_server_id=body["remote_server_id"],
            remote_run_dir=body["remote_run_dir"],
            note=body.get("note"),
        ),
    )


@app.post("/api/experiments/{experiment_id}/trials/import-remote")
def import_remote_trial(experiment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    return _invoke_sync(
        "import-remote-trial",
        lambda: service.import_remote_run(
            experiment_id,
            remote_server_id=body["remote_server_id"],
            remote_run_dir=body["remote_run_dir"],
            note=body.get("note"),
        ),
    )


@app.get("/api/trials/{trial_id}/summary")
def get_api_summary(trial_id: str, compact: bool = False) -> dict[str, Any]:
    return _invoke_sync("get-api-summary", lambda: service.get_summary(trial_id, compact=compact))


@app.post("/api/trials/{trial_id}/hyperparameter-templates")
def save_trial_hyperparameter_template(trial_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload or {})
    try:
        return service.save_trial_hyperparameter_template(
            trial_id,
            body.get("name", ""),
            overwrite=bool(body.get("overwrite", False)),
        )
    except TemplateNameConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": str(exc), "code": "TEMPLATE_NAME_CONFLICT"},
        ) from exc
    except (ServiceError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": str(exc), "action": "save-trial-hyperparameter-template"},
        ) from exc


@app.get("/api/trials/{trial_id}/continuation")
def get_trial_continuation(trial_id: str) -> dict[str, Any]:
    return _invoke_sync("get-trial-continuation", lambda: service.get_continuation_options(trial_id))


@app.post("/api/trials/{trial_id}/continue")
def continue_trial(trial_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = dict(payload or {})
    try:
        return training_queue.submit_continuation(
            trial_id,
            additional_epochs=body.get("additional_epochs"),
            lr0=body.get("lr0"),
            patience=body.get("patience"),
            note=body.get("note"),
            enqueue_if_busy=bool(body.get("enqueue_if_busy", False)),
        )
    except TrainingCapacityError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": str(exc),
                "code": "TRAINING_CAPACITY_REACHED",
                "running_count": exc.running_count,
                "max_parallel_training_tasks": exc.max_parallel,
            },
        ) from exc
    except (ServiceError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": str(exc), "action": "continue-trial"},
        ) from exc


@app.patch("/api/trials/{trial_id}")
def rename_trial(trial_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload or {})
    return _invoke_sync(
        "rename-trial",
        lambda: service.rename_trial(trial_id, body.get("display_name", "")),
    )


@app.post("/api/trials/{trial_id}/remote-sync")
def sync_remote_trial(trial_id: str) -> dict[str, Any]:
    return _invoke_sync("sync-remote-trial", lambda: service.sync_remote_trial(trial_id))


@app.post("/api/trials/{trial_id}/export-onnx")
def export_trial_onnx(trial_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    return _invoke_sync(
        "export-trial-onnx",
        lambda: service.export_trial_onnx(
            trial_id,
            model_name=body["model_name"],
            output_dir=body["output_dir"],
        ),
    )


@app.post("/api/trials/{trial_id}/validate-preview")
def validate_trial_preview(trial_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = dict(payload or {})

    def start_job() -> dict[str, Any]:
        trial = service.repo.get_trial(trial_id)
        return _invoke_async(
            "validate-trial-preview",
            trial.experiment_id,
            lambda: service.validate_trial_preview(
                trial_id,
                image_limit=int(body.get("image_limit", 50)),
                conf=float(body.get("conf", 0.25)),
            ),
        )

    return _invoke_sync("start-validate-trial-preview", start_job)


@app.get("/api/trials/{trial_id}/validation-preview")
def get_validation_preview(trial_id: str) -> dict[str, Any]:
    return _invoke_sync(
        "get-validation-preview",
        lambda: service.get_validation_preview(trial_id),
    )


@app.delete("/api/trials/{trial_id}")
def delete_trial(trial_id: str, keep_files: bool = True, force: bool = False) -> dict[str, Any]:
    return _invoke_sync(
        "delete-trial",
        lambda: service.delete_trial(trial_id, keep_files=keep_files, force=force),
    )


@app.get("/inspect-dataset")
def inspect_dataset(dataset_root: str) -> dict[str, Any]:
    return _invoke_sync("inspect-dataset", lambda: service.inspect_dataset(dataset_root))


@app.get("/api/experiments/{experiment_id}/curves")
def get_experiment_curves(experiment_id: str) -> dict[str, Any]:
    return _invoke_sync(
        "get-experiment-curves",
        lambda: service.get_experiment_curves(experiment_id),
    )


@app.get("/api/trials/{trial_id}/visualizations")
def get_trial_visualizations(trial_id: str) -> dict[str, Any]:
    return _invoke_sync(
        "get-trial-visualizations",
        lambda: service.get_trial_visualizations(trial_id),
    )


@app.get("/api/trials/{trial_id}/files/{filename}")
def get_trial_file(trial_id: str, filename: str) -> FileResponse:
    try:
        path = service.get_trial_file_path(trial_id, filename)
        return FileResponse(path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc), "action": "get-trial-file"})


@app.get("/api/trials/{trial_id}/validation-previews/{validation_id}/files/{filename}")
def get_validation_preview_file(trial_id: str, validation_id: str, filename: str) -> FileResponse:
    try:
        path = service.get_validation_preview_file_path(trial_id, validation_id, filename)
        return FileResponse(path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc), "action": "get-validation-preview-file"})


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return _invoke_sync("get-job", lambda: job_store.get(job_id).to_dict())


def _frontend_dist() -> Path:
    configured = os.environ.get("YOLO_FRONTEND_DIST", "").strip()
    candidates = [
        Path(configured) if configured else None,
        Path.cwd() / "frontend" / "dist",
        Path(__file__).resolve().parents[2] / "frontend" / "dist",
    ]
    for candidate in candidates:
        if candidate and candidate.joinpath("index.html").exists():
            return candidate.resolve()
    return (Path(configured) if configured else Path.cwd() / "frontend" / "dist").resolve()


dist_dir = _frontend_dist()
if dist_dir.joinpath("index.html").exists():
    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="frontend")


def main() -> None:
    dist = _frontend_dist()
    if not dist.joinpath("index.html").exists():
        print(
            f"frontend build not found: {dist / 'index.html'}\n"
            "Run `npm ci` and `npm run build` in the frontend directory for built mode.",
        )
    host = os.environ.get("YOLO_HOST", "127.0.0.1")
    port = int(os.environ.get("YOLO_PORT", "8765"))
    uvicorn.run(
        "backend.api:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
