from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.models import ExperimentConfig, RemoteServer, TrainingTask, TrialRecord
from backend.utils import utc_now_iso


DEFAULT_PROJECT_NAME = "未分组"


def default_project_name(description: str) -> str:
    normalized = str(description or "").strip()
    return normalized[:2] if normalized else DEFAULT_PROJECT_NAME


class Repository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._memory_connection: sqlite3.Connection | None = None
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if self._memory_connection is None:
                self._memory_connection = sqlite3.connect(self.db_path, check_same_thread=False)
                self._memory_connection.row_factory = sqlite3.Row
                self._configure_connection(self._memory_connection)
            return self._memory_connection
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        self._configure_connection(connection)
        return connection

    def _configure_connection(self, connection: sqlite3.Connection) -> None:
        for statement in (
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
            "PRAGMA temp_store=MEMORY",
        ):
            try:
                connection.execute(statement)
            except sqlite3.OperationalError:
                continue

    def _ensure_schema(self) -> None:
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    description TEXT NOT NULL DEFAULT '',
                    project TEXT NOT NULL DEFAULT '',
                    task_type TEXT NOT NULL,
                    dataset_root TEXT NOT NULL,
                    dataset_yaml TEXT NOT NULL,
                    pretrained_model TEXT NOT NULL,
                    save_root TEXT NOT NULL,
                    status TEXT NOT NULL,
                    initial_params TEXT NOT NULL,
                    search_space TEXT NOT NULL,
                    stop_conditions TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trials (
                    trial_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL DEFAULT '',
                    experiment_id TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    params_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    dataset_analysis_json TEXT NOT NULL DEFAULT '{}',
                    run_dir TEXT NOT NULL,
                    summary_path TEXT,
                    status TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'trained',
                    note TEXT NOT NULL DEFAULT '',
                    reason TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    model_source TEXT NOT NULL DEFAULT 'experiment_default',
                    params_source TEXT NOT NULL DEFAULT 'manual',
                    remote_server_id TEXT NOT NULL DEFAULT '',
                    remote_run_dir TEXT NOT NULL DEFAULT '',
                    sync_status TEXT NOT NULL DEFAULT '',
                    sync_error TEXT NOT NULL DEFAULT '',
                    remote_training_status TEXT NOT NULL DEFAULT '',
                    last_remote_csv_size INTEGER,
                    last_remote_csv_mtime REAL,
                    last_synced_epoch_count INTEGER NOT NULL DEFAULT 0,
                    unchanged_sync_count INTEGER NOT NULL DEFAULT 0,
                    last_synced_at TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                );

                CREATE TABLE IF NOT EXISTS remote_servers (
                    remote_server_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    username TEXT NOT NULL,
                    auth_type TEXT NOT NULL,
                    private_key_path TEXT NOT NULL DEFAULT '',
                    password_ref TEXT NOT NULL DEFAULT '',
                    default_runs_root TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    trial_id TEXT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_tasks (
                    queue_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    pretrained TEXT NOT NULL DEFAULT '',
                    note TEXT NOT NULL DEFAULT '',
                    reason TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    trial_id TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                );

                CREATE INDEX IF NOT EXISTS idx_training_tasks_status_position
                ON training_tasks (status, position, created_at);
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(experiments)").fetchall()
            }
            if "description" not in columns:
                conn.execute("ALTER TABLE experiments ADD COLUMN description TEXT NOT NULL DEFAULT ''")
            if "project" not in columns:
                conn.execute("ALTER TABLE experiments ADD COLUMN project TEXT NOT NULL DEFAULT ''")
            legacy_columns = {"session_key", "auto_iterate", "confirm_timeout"}
            if legacy_columns.intersection(columns) or "goal_config" in columns:
                self._backup_database()
                self._rebuild_experiments_table(conn)
                columns = {
                    row["name"]
                    for row in conn.execute("PRAGMA table_info(experiments)").fetchall()
                }
            conn.execute(
                """
                UPDATE experiments
                SET project = CASE
                    WHEN TRIM(description) = '' THEN ?
                    ELSE SUBSTR(TRIM(description), 1, 2)
                END
                WHERE TRIM(COALESCE(project, '')) = ''
                """,
                (DEFAULT_PROJECT_NAME,),
            )
            trial_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(trials)").fetchall()
            }
            if "display_name" not in trial_columns:
                conn.execute("ALTER TABLE trials ADD COLUMN display_name TEXT NOT NULL DEFAULT ''")
            if "source" not in trial_columns:
                conn.execute("ALTER TABLE trials ADD COLUMN source TEXT NOT NULL DEFAULT 'trained'")
            if "note" not in trial_columns:
                conn.execute("ALTER TABLE trials ADD COLUMN note TEXT NOT NULL DEFAULT ''")
            if "reason" not in trial_columns:
                conn.execute("ALTER TABLE trials ADD COLUMN reason TEXT NOT NULL DEFAULT ''")
            trial_defaults = {
                "dataset_analysis_json": "TEXT NOT NULL DEFAULT '{}'",
                "model": "TEXT NOT NULL DEFAULT ''",
                "model_source": "TEXT NOT NULL DEFAULT 'experiment_default'",
                "params_source": "TEXT NOT NULL DEFAULT 'manual'",
                "remote_server_id": "TEXT NOT NULL DEFAULT ''",
                "remote_run_dir": "TEXT NOT NULL DEFAULT ''",
                "sync_status": "TEXT NOT NULL DEFAULT ''",
                "sync_error": "TEXT NOT NULL DEFAULT ''",
                "remote_training_status": "TEXT NOT NULL DEFAULT ''",
                "last_remote_csv_size": "INTEGER",
                "last_remote_csv_mtime": "REAL",
                "last_synced_epoch_count": "INTEGER NOT NULL DEFAULT 0",
                "unchanged_sync_count": "INTEGER NOT NULL DEFAULT 0",
                "last_synced_at": "TEXT NOT NULL DEFAULT ''",
            }
            for column, definition in trial_defaults.items():
                if column not in trial_columns:
                    conn.execute(f"ALTER TABLE trials ADD COLUMN {column} {definition}")

            # WAITING_USER_CONFIRM was only used for the removed target-threshold
            # flow. Keep remote uncertainty distinguishable, but complete all
            # ordinary historical trials during the schema migration.
            conn.execute(
                """
                UPDATE trials
                SET status = 'COMPLETED'
                WHERE status = 'WAITING_USER_CONFIRM'
                  AND COALESCE(remote_training_status, '') <> 'maybe_stopped'
                """
            )
            conn.execute(
                """
                UPDATE experiments
                SET status = 'COMPLETED'
                WHERE status = 'WAITING_USER_CONFIRM'
                  AND NOT EXISTS (
                    SELECT 1 FROM trials
                    WHERE trials.experiment_id = experiments.experiment_id
                      AND trials.status = 'WAITING_USER_CONFIRM'
                  )
                """
            )

    def _backup_database(self) -> None:
        if self.db_path == ":memory:":
            return
        db_path = Path(self.db_path)
        if not db_path.exists():
            return
        backup_path = db_path.with_name(
            f"{db_path.name}.schema-backup-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        )
        shutil.copy2(db_path, backup_path)

    def _rebuild_experiments_table(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE experiments_new (
                experiment_id TEXT PRIMARY KEY,
                description TEXT NOT NULL DEFAULT '',
                project TEXT NOT NULL DEFAULT '',
                task_type TEXT NOT NULL,
                dataset_root TEXT NOT NULL,
                dataset_yaml TEXT NOT NULL,
                pretrained_model TEXT NOT NULL,
                save_root TEXT NOT NULL,
                status TEXT NOT NULL,
                initial_params TEXT NOT NULL,
                search_space TEXT NOT NULL,
                stop_conditions TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            INSERT INTO experiments_new (
                experiment_id, description, project, task_type, dataset_root, dataset_yaml,
                pretrained_model, save_root, status, initial_params,
                search_space, stop_conditions, created_at
            )
            SELECT
                experiment_id, description, project, task_type, dataset_root, dataset_yaml,
                pretrained_model, save_root, status, initial_params,
                search_space, stop_conditions, created_at
            FROM experiments;

            DROP TABLE experiments;
            ALTER TABLE experiments_new RENAME TO experiments;
            """
        )

    def _next_id(self, prefix: str, table: str, column: str) -> str:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {column} AS value FROM {table} WHERE {column} LIKE ?",
                (f"{prefix}_%",),
            ).fetchall()
        max_index = 0
        for row in rows:
            raw_value = row["value"]
            try:
                _, suffix = str(raw_value).rsplit("_", 1)
                max_index = max(max_index, int(suffix))
            except (ValueError, TypeError):
                continue
        return f"{prefix}_{max_index + 1:03d}"

    def next_experiment_id(self) -> str:
        return self._next_id("exp", "experiments", "experiment_id")

    def next_trial_id(self) -> str:
        return self._next_id("trial", "trials", "trial_id")

    def trial_id_exists(self, trial_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM trials WHERE trial_id = ? LIMIT 1",
                (trial_id,),
            ).fetchone()
        return row is not None

    def get_setting(self, key: str, default: str = "") -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return default
        return str(row["value"] or "")

    def set_setting(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, str(value or ""), utc_now_iso()),
            )

    @staticmethod
    def _training_task_from_row(row: sqlite3.Row) -> TrainingTask:
        return TrainingTask(
            queue_id=row["queue_id"],
            experiment_id=row["experiment_id"],
            params=json.loads(row["params_json"]),
            pretrained=row["pretrained"],
            note=row["note"],
            reason=row["reason"],
            status=row["status"],
            position=int(row["position"]),
            trial_id=row["trial_id"],
            error=row["error"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )

    def create_training_task(self, task: TrainingTask) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO training_tasks (
                    queue_id, experiment_id, params_json, pretrained, note, reason,
                    status, position, trial_id, error, created_at, started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.queue_id,
                    task.experiment_id,
                    json.dumps(task.params),
                    task.pretrained,
                    task.note,
                    task.reason,
                    task.status,
                    task.position,
                    task.trial_id,
                    task.error,
                    task.created_at or utc_now_iso(),
                    task.started_at,
                    task.finished_at,
                ),
            )

    def get_training_task(self, queue_id: str) -> TrainingTask:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM training_tasks WHERE queue_id = ?",
                (queue_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"training task not found: {queue_id}")
        return self._training_task_from_row(row)

    def list_training_tasks(self, statuses: tuple[str, ...] | None = None) -> list[TrainingTask]:
        query = "SELECT * FROM training_tasks"
        params: list[Any] = []
        if statuses:
            query += f" WHERE status IN ({','.join('?' for _ in statuses)})"
            params.extend(statuses)
        query += " ORDER BY position ASC, created_at ASC, queue_id ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._training_task_from_row(row) for row in rows]

    def update_training_task(
        self,
        queue_id: str,
        *,
        status: str | None = None,
        position: int | None = None,
        trial_id: str | None = None,
        error: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> None:
        values_by_column = {
            "status": status,
            "position": position,
            "trial_id": trial_id,
            "error": error,
            "started_at": started_at,
            "finished_at": finished_at,
        }
        assignments: list[str] = []
        values: list[Any] = []
        for column, value in values_by_column.items():
            if value is not None:
                assignments.append(f"{column} = ?")
                values.append(value)
        if not assignments:
            return
        values.append(queue_id)
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE training_tasks SET {', '.join(assignments)} WHERE queue_id = ?",
                values,
            )
        if cursor.rowcount == 0:
            raise KeyError(f"training task not found: {queue_id}")

    def reorder_queued_training_task(self, queue_id: str, target_position: int) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT queue_id FROM training_tasks WHERE status = 'QUEUED' ORDER BY position, created_at, queue_id"
            ).fetchall()
            queue_ids = [str(row["queue_id"]) for row in rows]
            if queue_id not in queue_ids:
                raise KeyError(f"queued training task not found: {queue_id}")
            queue_ids.remove(queue_id)
            index = max(0, min(int(target_position) - 1, len(queue_ids)))
            queue_ids.insert(index, queue_id)
            for position, current_id in enumerate(queue_ids, start=1):
                conn.execute(
                    "UPDATE training_tasks SET position = ? WHERE queue_id = ?",
                    (position, current_id),
                )

    def next_training_task_position(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(position), 0) + 1 AS next_position FROM training_tasks WHERE status = 'QUEUED'"
            ).fetchone()
        return int(row["next_position"])

    def delete_training_tasks_for_experiment(self, experiment_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM training_tasks WHERE experiment_id = ?", (experiment_id,))
        return int(cursor.rowcount or 0)

    def trial_display_name_exists(
        self,
        experiment_id: str,
        display_name: str,
        *,
        exclude_trial_id: str | None = None,
    ) -> bool:
        query = "SELECT 1 FROM trials WHERE experiment_id = ? AND display_name = ?"
        params: list[Any] = [experiment_id, display_name]
        if exclude_trial_id is not None:
            query += " AND trial_id != ?"
            params.append(exclude_trial_id)
        query += " LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        return row is not None

    def create_experiment(self, config: ExperimentConfig) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, description, project, task_type, dataset_root, dataset_yaml, pretrained_model,
                    save_root, status,
                    initial_params, search_space, stop_conditions, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    config.experiment_id,
                    config.description,
                    config.project,
                    config.task_type,
                    config.dataset_root,
                    config.dataset_yaml,
                    config.pretrained_model,
                    config.save_root,
                    config.status,
                    json.dumps(config.initial_params),
                    json.dumps(config.search_space),
                    json.dumps(config.stop_conditions),
                    utc_now_iso(),
                ),
            )

    def get_experiment(self, experiment_id: str) -> ExperimentConfig:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"experiment not found: {experiment_id}")
        return ExperimentConfig(
            experiment_id=row["experiment_id"],
            description=row["description"],
            project=row["project"] or default_project_name(row["description"]),
            task_type=row["task_type"],
            dataset_root=row["dataset_root"],
            dataset_yaml=row["dataset_yaml"],
            pretrained_model=row["pretrained_model"],
            save_root=row["save_root"],
            status=row["status"],
            initial_params=json.loads(row["initial_params"]),
            search_space=json.loads(row["search_space"]),
            stop_conditions=json.loads(row["stop_conditions"]),
        )

    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE experiments SET status = ? WHERE experiment_id = ?",
                (status, experiment_id),
            )

    def update_experiment(self, experiment_id: str, *, description: str | None = None, project: str | None = None) -> None:
        assignments: list[str] = []
        values: list[Any] = []
        if description is not None:
            assignments.append("description = ?")
            values.append(description)
        if project is not None:
            assignments.append("project = ?")
            values.append(project)
        if not assignments:
            return
        values.append(experiment_id)
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE experiments SET {', '.join(assignments)} WHERE experiment_id = ?",
                values,
            )
            if cursor.rowcount == 0:
                raise KeyError(f"experiment not found: {experiment_id}")

    def update_experiment_description(self, experiment_id: str, description: str) -> None:
        self.update_experiment(experiment_id, description=description)

    def delete_experiment(self, experiment_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))

    def list_experiments(self) -> list[ExperimentConfig]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC, experiment_id DESC"
            ).fetchall()
        return [
            ExperimentConfig(
                experiment_id=row["experiment_id"],
                description=row["description"],
                project=row["project"] or default_project_name(row["description"]),
                task_type=row["task_type"],
                dataset_root=row["dataset_root"],
                dataset_yaml=row["dataset_yaml"],
                pretrained_model=row["pretrained_model"],
                save_root=row["save_root"],
                status=row["status"],
                initial_params=json.loads(row["initial_params"]),
                search_space=json.loads(row["search_space"]),
                stop_conditions=json.loads(row["stop_conditions"]),
            )
            for row in rows
        ]

    def stale_unstarted_experiments(self, cutoff_iso: str, status: str) -> list[ExperimentConfig]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.*
                FROM experiments e
                LEFT JOIN trials t ON t.experiment_id = e.experiment_id
                WHERE e.status = ? AND e.created_at < ?
                GROUP BY e.experiment_id
                HAVING COUNT(t.trial_id) = 0
                ORDER BY e.created_at ASC, e.experiment_id ASC
                """,
                (status, cutoff_iso),
            ).fetchall()
        return [
            ExperimentConfig(
                experiment_id=row["experiment_id"],
                description=row["description"],
                project=row["project"] or default_project_name(row["description"]),
                task_type=row["task_type"],
                dataset_root=row["dataset_root"],
                dataset_yaml=row["dataset_yaml"],
                pretrained_model=row["pretrained_model"],
                save_root=row["save_root"],
                status=row["status"],
                initial_params=json.loads(row["initial_params"]),
                search_space=json.loads(row["search_space"]),
                stop_conditions=json.loads(row["stop_conditions"]),
            )
            for row in rows
        ]

    def create_trial(self, trial: TrialRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trials (
                    trial_id, display_name, experiment_id, iteration, params_json, metrics_json, dataset_analysis_json, run_dir,
                    summary_path, status, source, note, reason, model, model_source,
                    params_source, remote_server_id, remote_run_dir, sync_status, sync_error,
                    remote_training_status, last_remote_csv_size, last_remote_csv_mtime,
                    last_synced_epoch_count, unchanged_sync_count, last_synced_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trial.trial_id,
                    trial.display_name,
                    trial.experiment_id,
                    trial.iteration,
                    json.dumps(trial.params),
                    json.dumps(trial.metrics),
                    json.dumps(trial.dataset_analysis),
                    trial.run_dir,
                    trial.summary_path,
                    trial.status,
                    trial.source,
                    trial.note,
                    trial.reason,
                    trial.model,
                    trial.model_source,
                    trial.params_source,
                    trial.remote_server_id,
                    trial.remote_run_dir,
                    trial.sync_status,
                    trial.sync_error,
                    trial.remote_training_status,
                    trial.last_remote_csv_size,
                    trial.last_remote_csv_mtime,
                    trial.last_synced_epoch_count,
                    trial.unchanged_sync_count,
                    trial.last_synced_at,
                    trial.created_at or utc_now_iso(),
                ),
            )

    def update_trial(
        self,
        trial_id: str,
        *,
        display_name: str | None = None,
        status: str | None = None,
        metrics: dict[str, Any] | None = None,
        dataset_analysis: dict[str, Any] | None = None,
        summary_path: str | None = None,
        run_dir: str | None = None,
        model: str | None = None,
        model_source: str | None = None,
        params_source: str | None = None,
        sync_status: str | None = None,
        sync_error: str | None = None,
        remote_training_status: str | None = None,
        last_remote_csv_size: int | None = None,
        last_remote_csv_mtime: float | None = None,
        last_synced_epoch_count: int | None = None,
        unchanged_sync_count: int | None = None,
        last_synced_at: str | None = None,
    ) -> None:
        assignments: list[str] = []
        values: list[Any] = []
        if display_name is not None:
            assignments.append("display_name = ?")
            values.append(display_name)
        if status is not None:
            assignments.append("status = ?")
            values.append(status)
        if metrics is not None:
            assignments.append("metrics_json = ?")
            values.append(json.dumps(metrics))
        if dataset_analysis is not None:
            assignments.append("dataset_analysis_json = ?")
            values.append(json.dumps(dataset_analysis))
        if summary_path is not None:
            assignments.append("summary_path = ?")
            values.append(summary_path)
        optional_values = {
            "run_dir": run_dir,
            "model": model,
            "model_source": model_source,
            "params_source": params_source,
            "sync_status": sync_status,
            "sync_error": sync_error,
            "remote_training_status": remote_training_status,
            "last_remote_csv_size": last_remote_csv_size,
            "last_remote_csv_mtime": last_remote_csv_mtime,
            "last_synced_epoch_count": last_synced_epoch_count,
            "unchanged_sync_count": unchanged_sync_count,
            "last_synced_at": last_synced_at,
        }
        for column, value in optional_values.items():
            if value is not None:
                assignments.append(f"{column} = ?")
                values.append(value)
        if not assignments:
            return
        values.append(trial_id)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE trials SET {', '.join(assignments)} WHERE trial_id = ?",
                values,
            )

    def get_trial(self, trial_id: str) -> TrialRecord:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM trials WHERE trial_id = ?",
                (trial_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"trial not found: {trial_id}")
        return TrialRecord(
            trial_id=row["trial_id"],
            display_name=row["display_name"] or row["trial_id"],
            experiment_id=row["experiment_id"],
            iteration=int(row["iteration"]),
            params=json.loads(row["params_json"]),
            status=row["status"],
            run_dir=row["run_dir"],
            summary_path=row["summary_path"],
            metrics=json.loads(row["metrics_json"]),
            dataset_analysis=json.loads(row["dataset_analysis_json"]),
            source=row["source"],
            note=row["note"],
            reason=row["reason"],
            model=row["model"],
            model_source=row["model_source"],
            params_source=row["params_source"],
            remote_server_id=row["remote_server_id"],
            remote_run_dir=row["remote_run_dir"],
            sync_status=row["sync_status"],
            sync_error=row["sync_error"],
            remote_training_status=row["remote_training_status"],
            last_remote_csv_size=row["last_remote_csv_size"],
            last_remote_csv_mtime=row["last_remote_csv_mtime"],
            last_synced_epoch_count=int(row["last_synced_epoch_count"]),
            unchanged_sync_count=int(row["unchanged_sync_count"]),
            last_synced_at=row["last_synced_at"],
            created_at=row["created_at"],
        )

    def list_trials(self, experiment_id: str) -> list[TrialRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trials WHERE experiment_id = ? ORDER BY iteration ASC",
                (experiment_id,),
            ).fetchall()
        return [
            TrialRecord(
                trial_id=row["trial_id"],
                display_name=row["display_name"] or row["trial_id"],
                experiment_id=row["experiment_id"],
                iteration=int(row["iteration"]),
                params=json.loads(row["params_json"]),
                status=row["status"],
                run_dir=row["run_dir"],
                summary_path=row["summary_path"],
                metrics=json.loads(row["metrics_json"]),
                dataset_analysis=json.loads(row["dataset_analysis_json"]),
                source=row["source"],
                note=row["note"],
                reason=row["reason"],
                model=row["model"],
                model_source=row["model_source"],
                params_source=row["params_source"],
                remote_server_id=row["remote_server_id"],
                remote_run_dir=row["remote_run_dir"],
                sync_status=row["sync_status"],
                sync_error=row["sync_error"],
                remote_training_status=row["remote_training_status"],
                last_remote_csv_size=row["last_remote_csv_size"],
                last_remote_csv_mtime=row["last_remote_csv_mtime"],
                last_synced_epoch_count=int(row["last_synced_epoch_count"]),
                unchanged_sync_count=int(row["unchanged_sync_count"]),
                last_synced_at=row["last_synced_at"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def create_remote_server(self, server: RemoteServer) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO remote_servers (
                    remote_server_id, name, host, port, username, auth_type,
                    private_key_path, password_ref, default_runs_root, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    server.remote_server_id,
                    server.name,
                    server.host,
                    int(server.port),
                    server.username,
                    server.auth_type,
                    server.private_key_path,
                    server.password_ref,
                    server.default_runs_root,
                    utc_now_iso(),
                ),
            )

    def list_remote_servers(self) -> list[RemoteServer]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM remote_servers ORDER BY created_at DESC, remote_server_id DESC"
            ).fetchall()
        return [
            RemoteServer(
                remote_server_id=row["remote_server_id"],
                name=row["name"],
                host=row["host"],
                port=int(row["port"]),
                username=row["username"],
                auth_type=row["auth_type"],
                private_key_path=row["private_key_path"],
                password_ref=row["password_ref"],
                default_runs_root=row["default_runs_root"],
            )
            for row in rows
        ]

    def get_remote_server(self, remote_server_id: str) -> RemoteServer:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM remote_servers WHERE remote_server_id = ?",
                (remote_server_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"remote server not found: {remote_server_id}")
        return RemoteServer(
            remote_server_id=row["remote_server_id"],
            name=row["name"],
            host=row["host"],
            port=int(row["port"]),
            username=row["username"],
            auth_type=row["auth_type"],
            private_key_path=row["private_key_path"],
            password_ref=row["password_ref"],
            default_runs_root=row["default_runs_root"],
        )

    def delete_trials_for_experiment(self, experiment_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM trials WHERE experiment_id = ?", (experiment_id,))
        return int(cursor.rowcount or 0)

    def delete_trial(self, trial_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM trials WHERE trial_id = ?", (trial_id,))
        return int(cursor.rowcount or 0)

    def add_event(
        self,
        experiment_id: str,
        event_type: str,
        payload: dict[str, Any],
        trial_id: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events (experiment_id, trial_id, event_type, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (experiment_id, trial_id, event_type, json.dumps(payload), utc_now_iso()),
            )

    def delete_events_for_experiment(self, experiment_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM events WHERE experiment_id = ?", (experiment_id,))
        return int(cursor.rowcount or 0)

    def delete_events_for_trial(self, trial_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM events WHERE trial_id = ?", (trial_id,))
        return int(cursor.rowcount or 0)

    def latest_event(self, experiment_id: str, event_type: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload_json FROM events
                WHERE experiment_id = ? AND event_type = ?
                ORDER BY event_id DESC LIMIT 1
                """,
                (experiment_id, event_type),
            ).fetchone()
        return None if row is None else json.loads(row["payload_json"])

    def latest_event_for_types(
        self,
        experiment_id: str,
        event_types: list[str],
    ) -> dict[str, Any] | None:
        if not event_types:
            return None
        placeholders = ", ".join("?" for _ in event_types)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT event_type, payload_json, created_at FROM events
                WHERE experiment_id = ? AND event_type IN ({placeholders})
                ORDER BY event_id DESC LIMIT 1
                """,
                (experiment_id, *event_types),
            ).fetchone()
        if row is None:
            return None
        return {
            "event_type": row["event_type"],
            "payload": json.loads(row["payload_json"]),
            "created_at": row["created_at"],
        }

    def recent_summaries(self, experiment_id: str, limit: int = 3) -> list[dict[str, Any]]:
        trials = [trial for trial in self.list_trials(experiment_id) if trial.summary_path]
        summaries: list[dict[str, Any]] = []
        for trial in trials[-limit:]:
            if trial.summary_path and Path(trial.summary_path).exists():
                summaries.append(json.loads(Path(trial.summary_path).read_text(encoding="utf-8")))
        return summaries
