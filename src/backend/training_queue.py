from __future__ import annotations

from threading import RLock, Thread
from typing import Any, Callable
from uuid import uuid4

from backend.constants import (
    DEFAULT_MAX_PARALLEL_TRAINING_TASKS,
    MAX_PARALLEL_TRAINING_SETTING_KEY,
    MAX_PARALLEL_TRAINING_TASKS_LIMIT,
    STATE_ANALYZING,
    STATE_CANCELLED,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_QUEUED,
    STATE_READY,
    STATE_RETRAINING,
    STATE_TRAINING,
)
from backend.models import TrainingTask
from backend.service import OrchestratorService, ServiceError
from backend.utils import utc_now_iso


QUEUE_STATUS_QUEUED = "QUEUED"
QUEUE_STATUS_RUNNING = "RUNNING"
QUEUE_STATUS_COMPLETED = "COMPLETED"
QUEUE_STATUS_FAILED = "FAILED"
QUEUE_STATUS_CANCELLED = "CANCELLED"

class TrainingCapacityError(ServiceError):
    def __init__(self, running_count: int, max_parallel: int) -> None:
        super().__init__("parallel training capacity reached")
        self.running_count = running_count
        self.max_parallel = max_parallel


class TrainingQueue:
    def __init__(
        self,
        service: OrchestratorService,
        *,
        run_trial: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.service = service
        self.repo = service.repo
        self._run_trial_callback = run_trial or service.run_trial
        self._lock = RLock()
        self._started = False

    def max_parallel(self) -> int:
        raw = self.repo.get_setting(
            MAX_PARALLEL_TRAINING_SETTING_KEY,
            str(DEFAULT_MAX_PARALLEL_TRAINING_TASKS),
        )
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = DEFAULT_MAX_PARALLEL_TRAINING_TASKS
        return max(1, min(MAX_PARALLEL_TRAINING_TASKS_LIMIT, value))

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._recover_interrupted_tasks_locked()
            self._dispatch_locked()

    def notify_settings_changed(self) -> None:
        with self._lock:
            self._dispatch_locked()

    def submit(
        self,
        experiment_id: str,
        *,
        params: dict[str, Any] | None,
        pretrained: str | None,
        note: str | None,
        reason: str | None,
        enqueue_if_busy: bool,
    ) -> dict[str, Any]:
        prepared = self.service.prepare_trial_request(
            experiment_id,
            params=params,
            pretrained=pretrained,
            note=note,
            reason=reason,
        )
        return self._submit_prepared(
            experiment_id,
            prepared,
            enqueue_if_busy=enqueue_if_busy,
        )

    def submit_continuation(
        self,
        trial_id: str,
        *,
        additional_epochs: int,
        lr0: float | None,
        patience: int | None,
        note: str | None,
        enqueue_if_busy: bool,
    ) -> dict[str, Any]:
        prepared = self.service.prepare_continuation_request(
            trial_id,
            additional_epochs=additional_epochs,
            lr0=lr0,
            patience=patience,
            note=note,
        )
        return self._submit_prepared(
            prepared["experiment_id"],
            prepared,
            enqueue_if_busy=enqueue_if_busy,
            parent_trial_id=trial_id,
            training_mode="continued",
        )

    def _submit_prepared(
        self,
        experiment_id: str,
        prepared: dict[str, Any],
        *,
        enqueue_if_busy: bool,
        parent_trial_id: str = "",
        training_mode: str = "fresh",
    ) -> dict[str, Any]:
        with self._lock:
            if parent_trial_id and any(
                task.parent_trial_id == parent_trial_id
                for task in self.repo.list_training_tasks((QUEUE_STATUS_RUNNING, QUEUE_STATUS_QUEUED))
            ):
                raise ServiceError("this trial already has an active continuation task")
            running_count = len(self.repo.list_training_tasks((QUEUE_STATUS_RUNNING,)))
            if running_count >= self.max_parallel() and not enqueue_if_busy:
                raise TrainingCapacityError(running_count, self.max_parallel())
            task = TrainingTask(
                queue_id=f"queue_{uuid4().hex}",
                experiment_id=experiment_id,
                params=prepared["params"],
                pretrained=prepared["pretrained"],
                note=prepared["note"],
                reason=prepared["reason"],
                status=QUEUE_STATUS_QUEUED,
                position=self.repo.next_training_task_position(),
                created_at=utc_now_iso(),
                parent_trial_id=parent_trial_id,
                training_mode=training_mode,
            )
            self.repo.create_training_task(task)
            if not self._experiment_has_running_task(experiment_id):
                self.repo.update_experiment_status(experiment_id, STATE_QUEUED)
            self._dispatch_locked()
            current = self.repo.get_training_task(task.queue_id)
            return {
                "disposition": "started" if current.status == QUEUE_STATUS_RUNNING else "queued",
                "training_task": self._task_payload(current),
            }

    def list_tasks(self) -> dict[str, Any]:
        with self._lock:
            tasks = self.repo.list_training_tasks((QUEUE_STATUS_RUNNING, QUEUE_STATUS_QUEUED))
            payloads = [self._task_payload(task) for task in tasks]
            running = [item for item in payloads if item["status"] == QUEUE_STATUS_RUNNING]
            queued = [item for item in payloads if item["status"] == QUEUE_STATUS_QUEUED]
            for position, item in enumerate(queued, start=1):
                item["position"] = position
            return {
                "max_parallel_training_tasks": self.max_parallel(),
                "running_count": len(running),
                "queued_count": len(queued),
                "running": running,
                "queued": queued,
            }

    def cancel(self, queue_id: str) -> dict[str, Any]:
        with self._lock:
            task = self.repo.get_training_task(queue_id)
            if task.status != QUEUE_STATUS_QUEUED:
                raise ServiceError("only queued training tasks can be cancelled")
            self.repo.update_training_task(
                queue_id,
                status=QUEUE_STATUS_CANCELLED,
                finished_at=utc_now_iso(),
            )
            self._restore_experiment_status_locked(task.experiment_id)
            self._dispatch_locked()
            return {"training_task": self._task_payload(self.repo.get_training_task(queue_id))}

    def reorder(self, queue_id: str, target_position: int) -> dict[str, Any]:
        with self._lock:
            self.repo.reorder_queued_training_task(queue_id, target_position)
            task = self.repo.get_training_task(queue_id)
            return {"training_task": self._task_payload(task)}

    def _dispatch_locked(self) -> None:
        if not self._started:
            return
        running = self.repo.list_training_tasks((QUEUE_STATUS_RUNNING,))
        available = self.max_parallel() - len(running)
        if available <= 0:
            return
        running_experiments = {task.experiment_id for task in running}
        queued = self.repo.list_training_tasks((QUEUE_STATUS_QUEUED,))
        for task in queued:
            if available <= 0:
                break
            if task.experiment_id in running_experiments:
                continue
            self.repo.update_training_task(
                task.queue_id,
                status=QUEUE_STATUS_RUNNING,
                started_at=utc_now_iso(),
            )
            self.repo.update_experiment_status(task.experiment_id, STATE_TRAINING)
            running_experiments.add(task.experiment_id)
            available -= 1
            Thread(target=self._run_task, args=(task.queue_id,), daemon=True).start()

    def _run_task(self, queue_id: str) -> None:
        task = self.repo.get_training_task(queue_id)

        def on_trial_started(trial_id: str) -> None:
            self.repo.update_training_task(queue_id, trial_id=trial_id)

        result: dict[str, Any] | None = None
        error = ""
        try:
            result = self._run_trial_callback(
                task.experiment_id,
                params=task.params,
                pretrained=task.pretrained,
                note=task.note,
                reason=task.reason,
                parent_trial_id=task.parent_trial_id,
                training_mode=task.training_mode,
                on_trial_started=on_trial_started,
            )
        except Exception as exc:
            error = str(exc)
        with self._lock:
            try:
                current = self.repo.get_training_task(queue_id)
            except KeyError:
                self._dispatch_locked()
                return
            trial_id = current.trial_id or str((result or {}).get("trial_id") or "")
            internal_status = str((result or {}).get("internal_status") or STATE_FAILED)
            if internal_status == STATE_COMPLETED:
                queue_status = QUEUE_STATUS_COMPLETED
            elif internal_status == STATE_CANCELLED:
                queue_status = QUEUE_STATUS_CANCELLED
            else:
                queue_status = QUEUE_STATUS_FAILED
                if trial_id:
                    try:
                        trial = self.repo.get_trial(trial_id)
                        if trial.status in {STATE_TRAINING, STATE_RETRAINING, STATE_ANALYZING}:
                            self.repo.update_trial(trial_id, status=STATE_FAILED)
                    except KeyError:
                        pass
                self.repo.update_experiment_status(task.experiment_id, STATE_FAILED)
            self.repo.update_training_task(
                queue_id,
                status=queue_status,
                trial_id=trial_id,
                error=error,
                finished_at=utc_now_iso(),
            )
            if self._experiment_has_queued_task(task.experiment_id):
                self.repo.update_experiment_status(task.experiment_id, STATE_QUEUED)
            self._dispatch_locked()

    def _recover_interrupted_tasks_locked(self) -> None:
        for task in self.repo.list_training_tasks((QUEUE_STATUS_RUNNING,)):
            self.repo.update_training_task(
                task.queue_id,
                status=QUEUE_STATUS_FAILED,
                error="backend restarted while training was running",
                finished_at=utc_now_iso(),
            )
            if task.trial_id:
                try:
                    trial = self.repo.get_trial(task.trial_id)
                    if trial.status in {STATE_TRAINING, STATE_RETRAINING, STATE_ANALYZING}:
                        self.repo.update_trial(task.trial_id, status=STATE_FAILED)
                except KeyError:
                    pass
            self.repo.update_experiment_status(task.experiment_id, STATE_FAILED)
        for task in self.repo.list_training_tasks((QUEUE_STATUS_QUEUED,)):
            self.repo.update_experiment_status(task.experiment_id, STATE_QUEUED)

    def _restore_experiment_status_locked(self, experiment_id: str) -> None:
        if self._experiment_has_running_task(experiment_id):
            self.repo.update_experiment_status(experiment_id, STATE_TRAINING)
            return
        if self._experiment_has_queued_task(experiment_id):
            self.repo.update_experiment_status(experiment_id, STATE_QUEUED)
            return
        trials = self.repo.list_trials(experiment_id)
        self.repo.update_experiment_status(experiment_id, trials[-1].status if trials else STATE_READY)

    def _experiment_has_running_task(self, experiment_id: str) -> bool:
        return any(
            task.experiment_id == experiment_id
            for task in self.repo.list_training_tasks((QUEUE_STATUS_RUNNING,))
        )

    def _experiment_has_queued_task(self, experiment_id: str) -> bool:
        return any(
            task.experiment_id == experiment_id
            for task in self.repo.list_training_tasks((QUEUE_STATUS_QUEUED,))
        )

    def _task_payload(self, task: TrainingTask) -> dict[str, Any]:
        config = self.repo.get_experiment(task.experiment_id)
        parent_display_name = ""
        if task.parent_trial_id:
            try:
                parent_display_name = self.repo.get_trial(task.parent_trial_id).display_name
            except KeyError:
                parent_display_name = task.parent_trial_id
        return {
            "queue_id": task.queue_id,
            "experiment_id": task.experiment_id,
            "experiment_name": config.description,
            "project": config.project,
            "task_type": config.task_type,
            "model": task.pretrained,
            "params": task.params,
            "status": task.status,
            "position": task.position,
            "trial_id": task.trial_id,
            "error": task.error,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "finished_at": task.finished_at,
            "parent_trial_id": task.parent_trial_id,
            "training_mode": task.training_mode,
            "parent_display_name": parent_display_name,
        }
