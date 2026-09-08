import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

DEFAULT_DB_PATH = "sessions.db"


@dataclass
class Session:
    id: int
    start_time: datetime
    end_time: datetime | None
    device_name: str
    threshold: float
    sample_count: int
    bark_count: int
    peak_volume: float


@dataclass
class Sample:
    id: int
    session_id: int
    timestamp: float
    volume: float
    is_bark: int


def init_db(db_path: str = DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                device_name TEXT NOT NULL,
                threshold REAL NOT NULL,
                sample_count INTEGER DEFAULT 0,
                bark_count INTEGER DEFAULT 0,
                peak_volume REAL DEFAULT 0.0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                volume REAL NOT NULL,
                is_bark INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_samples_session ON samples (session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_samples_timestamp ON samples (timestamp)"
        )
        conn.commit()
    finally:
        conn.close()


def create_session(db_path: str = DEFAULT_DB_PATH, device_name: str = "", threshold: float = 1000.0) -> int | None:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "INSERT INTO sessions (start_time, device_name, threshold) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), device_name, threshold),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def finish_session(
    session_id: int,
    db_path: str = DEFAULT_DB_PATH,
    sample_count: int = 0,
    bark_count: int = 0,
    peak_volume: float = 0.0,
):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            UPDATE sessions
            SET end_time = ?, sample_count = ?, bark_count = ?, peak_volume = ?
            WHERE id = ?
            """,
            (datetime.now(timezone.utc).isoformat(), sample_count, bark_count, peak_volume, session_id),
        )
        conn.commit()
    finally:
        conn.close()


def insert_samples(session_id: int, samples: list[tuple], db_path: str = DEFAULT_DB_PATH):
    if not samples:
        return
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO samples (session_id, timestamp, volume, is_bark) VALUES (?, ?, ?, ?)",
            samples,
        )
        conn.commit()
    finally:
        conn.close()


def get_sessions(db_path: str = DEFAULT_DB_PATH) -> list[Session]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, start_time, end_time, device_name, threshold, sample_count, bark_count, peak_volume FROM sessions ORDER BY start_time DESC"
        ).fetchall()
        return [
            Session(
                id=r[0],
                start_time=datetime.fromisoformat(r[1]),
                end_time=datetime.fromisoformat(r[2]) if r[2] else None,
                device_name=r[3],
                threshold=r[4],
                sample_count=r[5],
                bark_count=r[6],
                peak_volume=r[7],
            )
            for r in rows
        ]
    finally:
        conn.close()


def get_session_samples(session_id: int, db_path: str = DEFAULT_DB_PATH) -> list[Sample]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, session_id, timestamp, volume, is_bark FROM samples WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        return [
            Sample(id=r[0], session_id=r[1], timestamp=r[2], volume=r[3], is_bark=r[4])
            for r in rows
        ]
    finally:
        conn.close()


def get_session_stats(session_id: int, db_path: str = DEFAULT_DB_PATH) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT sample_count, bark_count, peak_volume FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return {"sample_count": 0, "bark_count": 0, "peak_volume": 0.0}
        return {"sample_count": row[0], "bark_count": row[1], "peak_volume": row[2]}
    finally:
        conn.close()


def delete_session(session_id: int, db_path: str = DEFAULT_DB_PATH):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM samples WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()
