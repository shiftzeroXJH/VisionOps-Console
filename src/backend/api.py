from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from backend.service import OrchestratorService, ServiceError
from backend.jobs import JobStore

app = FastAPI(title="yolo-platform", version="0.1.0")
job_store = JobStore()
service = OrchestratorService(db_path=os.environ.get("YOLO_DB_PATH"))


def _invoke_sync(action: str, callback: Any) -> dict[str, Any]:
    try:
        return callback()
    except (ServiceError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
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


@app.post("/api/settings/clear-validation-cache")
def clear_validation_cache() -> dict[str, Any]:
    return _invoke_sync("clear-validation-cache", service.clear_validation_preview_cache)


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
        ),
    )


@app.post("/api/remote-servers/{remote_server_id}/test")
def test_remote_server(remote_server_id: str) -> dict[str, Any]:
    return _invoke_sync("test-remote-server", lambda: service.test_remote_server(remote_server_id))


@app.post("/api/experiments")
def create_experiment(payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    return _invoke_sync(
        "create-experiment",
        lambda: service.create_experiment(
            description=body.get("description", ""),
            project=body.get("project"),
            task_type=body["task_type"],
            dataset_root=body["dataset_root"],
            dataset_yaml=body.get("dataset_yaml"),
            pretrained=body["pretrained"],
            save_root=body["save_root"],
            goal=body["goal"],
            initial_params=body.get("initial_params"),
        ),
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
    return _invoke_async(
        "run-experiment-trial",
        experiment_id,
        lambda: service.run_trial(
            experiment_id,
            params=body.get("params"),
            pretrained=body.get("pretrained") or body.get("model"),
            note=body.get("note"),
            reason=body.get("reason"),
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
            "Run `npm install` and `npm run build` in the frontend directory for deployment mode.",
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

