from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from ..proxy.schemas import BBoxFrame


class DetectionCache:
    """Transactional, detector-versioned checkpoints for long vision runs."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS bbox_frames (
                detector TEXT NOT NULL,
                task TEXT NOT NULL,
                episode TEXT NOT NULL,
                frame INTEGER NOT NULL,
                camera TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (detector, task, episode, frame, camera)
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "DetectionCache":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def upsert(self, records: Iterable[BBoxFrame]) -> int:
        rows = [
            (
                record.detector,
                record.task,
                record.episode,
                record.frame,
                record.camera,
                json.dumps(record.to_dict(), sort_keys=True),
            )
            for record in records
        ]
        if not rows:
            return 0
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO bbox_frames(detector, task, episode, frame, camera, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(detector, task, episode, frame, camera)
                DO UPDATE SET payload=excluded.payload
                """,
                rows,
            )
        return len(rows)

    def load(self, detector: str) -> dict[tuple[str, str, int, str], BBoxFrame]:
        cursor = self.connection.execute(
            """
            SELECT payload FROM bbox_frames
            WHERE detector = ?
            ORDER BY task, episode, frame, camera
            """,
            (detector,),
        )
        records = [BBoxFrame.from_dict(json.loads(row[0])) for row in cursor]
        return {record.key(): record for record in records}

    def detector_counts(self) -> dict[str, int]:
        cursor = self.connection.execute(
            "SELECT detector, COUNT(*) FROM bbox_frames GROUP BY detector ORDER BY detector"
        )
        return {str(detector): int(count) for detector, count in cursor}

    def delete_camera(self, detector: str, camera: str) -> int:
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM bbox_frames WHERE detector = ? AND camera = ?",
                (detector, camera),
            )
        return int(cursor.rowcount)
