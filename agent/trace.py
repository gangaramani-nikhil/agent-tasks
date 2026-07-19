"""SQLite tracing for agent runs.

Every decision and observation the agent makes is written here, so a run can
be reconstructed step by step after the fact. Plain sqlite3, no ORM.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    task       TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at   REAL,
    final      TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL REFERENCES runs(run_id),
    step       INTEGER NOT NULL,
    kind       TEXT NOT NULL,          -- 'decision' | 'observation'
    payload    TEXT NOT NULL,          -- JSON
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id, step);
"""


class Tracer:
    """Writes agent runs and their steps to a SQLite database."""

    def __init__(self, db_path: str | Path = "traces.db") -> None:
        self.db_path = str(db_path)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def start_run(self, task: str) -> str:
        run_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, task, started_at) VALUES (?, ?, ?)",
                (run_id, task, time.time()),
            )
        return run_id

    def log(self, run_id: str, step: int, kind: str, payload: Any) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO steps (run_id, step, kind, payload, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (run_id, step, kind, json.dumps(payload), time.time()),
            )

    def end_run(self, run_id: str, final: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET ended_at = ?, final = ? WHERE run_id = ?",
                (time.time(), final, run_id),
            )

    def steps_for(self, run_id: str) -> list[dict]:
        """Return the recorded steps for a run, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT step, kind, payload, created_at FROM steps"
                " WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        return [
            {"step": s, "kind": k, "payload": json.loads(p), "created_at": t}
            for s, k, p, t in rows
        ]
