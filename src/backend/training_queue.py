from __future__ import annotations

from threading import Event, RLock, Thread
from typing import Any, Callable
from uuid import NAMESPACE_URL, uuid4, uuid5

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
REMOTE_POLL_INTERVAL_SECONDS = 30

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
        self._stop = Event()
        self._monitor: Thread | None = None
        self._remote_launching: set[str] = set()
        self._checking: set[str] = set()

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
            self._stop.clear()
            self._adopt_remote_trials_locked()
            self._recover_interrupted_tasks_locked()
            self._dispatch_locked()
            self._monitor = Thread(target=self._monitor_remote, daemon=True)
            self._monitor.start()

    def stop(self) -> None:
        with self._lock:
            self._started = False
            self._stop.set()
        if self._monitor:
            self._monitor.join(timeout=1)

    def submit_remote(
        self, experiment_id: str, *, remote_server_id: str, params: dict[str, Any] | None = None,
        pretrained: str | None = None, note: str | None = None, idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        token = str(idempotency_key or uuid4().hex)
        if len(token) > 200:
            raise ServiceError("idempotency_key must not exceed 200 characters")
        queue_id = "remote_" + uuid5(NAMESPACE_URL, f"{experiment_id}:{remote_server_id}:{token}").hex
        with self._lock:
            try:
                existing = self.repo.get_training_task(queue_id)
                return self._submission_payload(existing)
            except KeyError:
                pass
        prepared = self.service.prepare_remote_trial_request(
            experiment_id, remote_server_id=remote_server_id, params=params, pretrained=pretrained, note=note,
        )
        with self._lock:
            try:
                return self._submission_payload(self.repo.get_training_task(queue_id))
            except KeyError:
                pass
            task = TrainingTask(
                queue_id=queue_id, experiment_id=experiment_id,
                params=dict(prepared["params"]), pretrained=str(prepared.get("remote_model") or prepared["pretrained"]),
                note=str(prepared.get("note") or ""), reason="", status=QUEUE_STATUS_QUEUED,
                position=self.repo.next_training_task_position("remote", remote_server_id),
                source="remote", remote_server_id=remote_server_id, request_snapshot=prepared,
                trial_id="trial_" + uuid4().hex, phase="preparing", created_at=utc_now_iso(),
            )
            self.repo.create_training_task(task)
            self.repo.update_experiment_status(experiment_id, STATE_QUEUED)
            self._dispatch_locked()
            return self._submission_payload(self.repo.get_training_task(queue_id))

    def _submission_payload(self, task: TrainingTask) -> dict[str, Any]:
        payload = self._task_payload(task)
        if task.status == QUEUE_STATUS_QUEUED:
            pending = [item.queue_id for item in self.repo.list_training_tasks((QUEUE_STATUS_QUEUED,))
                       if item.source == task.source and item.remote_server_id == task.remote_server_id]
            payload["position"] = pending.index(task.queue_id) + 1
        return {"disposition": "queued" if task.status == QUEUE_STATUS_QUEUED else "started",
                "training_task": payload}

    def notify_settings_changed(self) -> None:
        with self._lock:
            self._dispatch_locked()

    def track_registered_remote_trials(self) -> None:
        with self._lock:
            self._adopt_remote_trials_locked()

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
            running_count = len([task for task in self.repo.list_training_tasks((QUEUE_STATUS_RUNNING,)) if task.source == "local"])
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
            targets = [("local", "本地训练", "local", "", self.max_parallel())]
            targets += [(server.remote_server_id, server.name, "remote", server.remote_server_id, server.max_parallel_training_tasks)
                        for server in self.repo.list_remote_servers()]
            failures = self.repo.list_training_tasks((QUEUE_STATUS_FAILED,))
            groups = []
            for target_id, name, source, server_id, limit in targets:
                target_tasks = [task for task in tasks if task.source == source and task.remote_server_id == server_id]
                running = [self._task_payload(task) for task in target_tasks if task.status == QUEUE_STATUS_RUNNING]
                queued = [self._task_payload(task) for task in target_tasks if task.status == QUEUE_STATUS_QUEUED]
                blocked = source == "remote" and any(item["phase"] == "unknown" for item in running)
                for position, item in enumerate(queued, start=1):
                    item["position"] = position
                    item["waiting_reason"] = ("服务器状态待确认，暂停派发" if blocked else
                        "等待同实验任务结束" if any(r["experiment_id"] == item["experiment_id"] for r in running) else
                        "等待运行名额" if len(running) >= limit else "等待调度")
                recent = sorted((task for task in failures if task.source == source and task.remote_server_id == server_id),
                                key=lambda task: task.finished_at, reverse=True)
                groups.append({"target_id": target_id, "name": name, "source": source, "remote_server_id": server_id,
                               "max_parallel_training_tasks": limit, "running_count": len(running), "queued_count": len(queued),
                               "running": running, "queued": queued, "blocked": blocked,
                               "last_failure": self._task_payload(recent[0]) if recent else None})
            local = groups[0]
            return {
                "max_parallel_training_tasks": self.max_parallel(),
                "running_count": local["running_count"], "queued_count": local["queued_count"],
                "running": local["running"], "queued": local["queued"], "groups": groups,
                "total_running_count": sum(group["running_count"] for group in groups),
                "total_queued_count": sum(group["queued_count"] for group in groups),
            }

    def cancel(self, queue_id: str) -> dict[str, Any]:
        with self._lock:
            task = self.repo.get_training_task(queue_id)
            if task.status != QUEUE_STATUS_QUEUED:
                raise ServiceError("only queued training tasks can be cancelled")
            if not self.repo.cancel_queued_training_task(queue_id):
                raise ServiceError("task has already started; it can no longer be cancelled from the queue")
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
        queued = self.repo.list_training_tasks((QUEUE_STATUS_QUEUED,))
        for task in queued:
            limit = self.max_parallel() if task.source == "local" else self.repo.get_remote_server(task.remote_server_id).max_parallel_training_tasks
            if not self.repo.claim_training_task(task.queue_id, limit):
                continue
            self.repo.update_experiment_status(task.experiment_id, STATE_TRAINING)
            if task.source == "remote":
                self._remote_launching.add(task.queue_id)
                Thread(target=self._run_remote_task, args=(task.queue_id,), daemon=True).start()
            else:
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
            self.repo.update_experiment_status(task.experiment_id, internal_status)
            if self._experiment_has_queued_task(task.experiment_id):
                self.repo.update_experiment_status(task.experiment_id, STATE_QUEUED)
            self._dispatch_locked()

    def _adopt_remote_trials_locked(self) -> None:
        linked = {task.trial_id for task in self.repo.list_training_tasks() if task.trial_id}
        for config in self.repo.list_experiments():
            for trial in self.repo.list_trials(config.experiment_id):
                if (trial.source != "remote_sftp" or not trial.remote_server_id or trial.trial_id in linked
                        or trial.status not in {STATE_TRAINING, STATE_RETRAINING, STATE_ANALYZING}):
                    continue
                self.repo.create_training_task(TrainingTask(
                    queue_id=f"remote_legacy_{trial.trial_id}", experiment_id=trial.experiment_id,
                    params=dict(trial.params), pretrained=trial.model, note=trial.note, reason="",
                    status=QUEUE_STATUS_RUNNING, position=0, trial_id=trial.trial_id,
                    created_at=trial.created_at, started_at=trial.created_at,
                    source="remote", remote_server_id=trial.remote_server_id, phase="unknown",
                    error="正在核对已有远程训练",
                ))
                self.repo.update_experiment_status(trial.experiment_id, STATE_TRAINING)
                linked.add(trial.trial_id)

    def _run_remote_task(self, queue_id: str) -> None:
        task = self.repo.get_training_task(queue_id)

        def prepared(trial_id: str) -> None:
            with self._lock:
                self.repo.update_training_task(queue_id, trial_id=trial_id)

        def launching() -> None:
            with self._lock:
                if not self._started:
                    raise ServiceError("后端正在停止，任务保留在队列中")
                self.repo.update_training_task(queue_id, phase="launching")

        try:
            self.service.launch_remote_trial(
                task.experiment_id, remote_server_id=task.remote_server_id,
                params=task.params, pretrained=task.pretrained, note=task.note,
                prepared_request=task.request_snapshot, trial_id=task.trial_id,
                on_trial_prepared=prepared, on_launch_attempt=launching,
            )
            with self._lock:
                self.repo.update_training_task(queue_id, phase="running", error="")
        except Exception as exc:
            with self._lock:
                current = self.repo.get_training_task(queue_id)
                if current.phase in {"launching", "running", "unknown"}:
                    self.repo.update_training_task(queue_id, phase="unknown", error="启动结果待确认，请重新检查远程状态")
                elif not self._started:
                    self.repo.update_training_task(queue_id, status=QUEUE_STATUS_QUEUED, phase="preparing", error="", started_at="")
                else:
                    self._finish_remote_locked(current, "failed", str(exc))
        finally:
            with self._lock:
                self._remote_launching.discard(queue_id)
                self._dispatch_locked()

    def _finish_remote_locked(self, task: TrainingTask, state: str, error: str) -> None:
        status = STATE_COMPLETED if state == "completed" else STATE_FAILED
        self.repo.update_training_task(task.queue_id, status=QUEUE_STATUS_COMPLETED if state == "completed" else QUEUE_STATUS_FAILED,
                                       phase="running", error=error, finished_at=utc_now_iso())
        if task.trial_id:
            try:
                self.repo.update_trial(task.trial_id, status=status,
                                       remote_training_status="completed" if state == "completed" else "maybe_stopped")
                self.repo.add_event(task.experiment_id, "REMOTE_QUEUE_FINISHED", {"queue_id": task.queue_id, "status": status, "error": error}, task.trial_id)
            except KeyError:
                pass
        self.repo.update_experiment_status(task.experiment_id, status)

    def recheck(self, queue_id: str) -> dict[str, Any]:
        with self._lock:
            task = self.repo.get_training_task(queue_id)
            if task.source != "remote":
                raise ServiceError("only remote training tasks can be rechecked")
            if task.status != QUEUE_STATUS_RUNNING or queue_id in self._checking or queue_id in self._remote_launching:
                return {"training_task": self._task_payload(task)}
            self._checking.add(queue_id)
        try:
            try:
                result = self.service.check_remote_trial_status(task.trial_id)
            except Exception:
                result = {"state": "unknown", "error": "无法核对远程训练状态，请检查服务器连接"}
            with self._lock:
                current = self.repo.get_training_task(queue_id)
                if current.status == QUEUE_STATUS_RUNNING:
                    state = result.get("state", "unknown")
                    if state in {"completed", "failed"}:
                        self._finish_remote_locked(current, state, str(result.get("error") or ""))
                    else:
                        self.repo.update_training_task(queue_id, phase="running" if state == "running" else "unknown",
                                                       error=str(result.get("error") or "") if state != "running" else "")
                    self._dispatch_locked()
                return {"training_task": self._task_payload(self.repo.get_training_task(queue_id))}
        finally:
            with self._lock:
                self._checking.discard(queue_id)

    def _monitor_remote(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                if not self._started:
                    return
                tasks = [task for task in self.repo.list_training_tasks((QUEUE_STATUS_RUNNING,))
                         if task.source == "remote" and task.queue_id not in self._checking
                         and task.queue_id not in self._remote_launching]
                for task in tasks:
                    Thread(target=self._safe_recheck, args=(task.queue_id,), daemon=True).start()
            if self._stop.wait(REMOTE_POLL_INTERVAL_SECONDS):
                return

    def _safe_recheck(self, queue_id: str) -> None:
        try:
            self.recheck(queue_id)
        except Exception:
            # Deleted tasks and transient database failures must not stop the monitor.
            pass

    def _recover_interrupted_tasks_locked(self) -> None:
        for task in self.repo.list_training_tasks((QUEUE_STATUS_RUNNING,)):
            if task.source == "remote":
                if task.phase == "preparing":
                    self.repo.update_training_task(task.queue_id, status=QUEUE_STATUS_QUEUED, error="", started_at="")
                else:
                    self.repo.update_training_task(task.queue_id, phase="unknown", error="后端重启，正在核对远程状态")
                continue
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
        epoch_count = 0
        if task.source == "remote" and task.trial_id:
            try:
                epoch_count = self.repo.get_trial(task.trial_id).last_synced_epoch_count
            except KeyError:
                pass
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
            "source": task.source,
            "remote_server_id": task.remote_server_id,
            "phase": "preparing" if task.phase == "launching" else task.phase,
            "last_synced_epoch_count": epoch_count if epoch_count > 0 else None,
        }
