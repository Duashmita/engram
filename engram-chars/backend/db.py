"""
SQLite persistence layer for the Engram live-demo backend.

Self-contained stdlib-sqlite3 module: no FastAPI imports, dict-in/dict-out.
Designed for the single-container Modal deployment (requests are fully
serialized there), but guarded with a module lock so local multi-threaded
dev (uvicorn --reload, background workers) is safe too.

Tables
------
  npcs        npc registry (preset|custom)
  speakers    one row per device_id, labelled "Visitor 1", "Visitor 2", ...
              in first-seen order
  sessions    one row per /start, closed by /end or TTL purge
  npc_files   snapshot store for the per-device memory payloads
              (fields: memories, longterm, keystore, state);
              speaker_scope = device_id (current behavior) or 'shared'
  memories    append-only attribution ledger (upsert on (npc_id, memory_id))
  waitlist    email signups

The Modal layer points ``on_write`` at ``volume.commit`` so every committed
write transaction is flushed to the persistent volume.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Callable, Optional

DB_PATH = os.environ.get("ENGRAM_DB_PATH", "/data/engram.sqlite3")

# Hook invoked (no args) after every committed write transaction.
# The Modal layer sets this to volume.commit; local dev leaves it None.
on_write: Optional[Callable[[], None]] = None

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None
_conn_path: Optional[str] = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS npcs (
    npc_id     TEXT PRIMARY KEY,
    name       TEXT,
    kind       TEXT CHECK (kind IN ('preset', 'custom')),
    created_at REAL,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS speakers (
    speaker_id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id  TEXT UNIQUE NOT NULL,
    label      TEXT,
    first_seen REAL,
    last_seen  REAL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    npc_id     TEXT,
    speaker_id INTEGER,
    started_at REAL,
    ended_at   REAL,
    turn_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS npc_files (
    npc_id        TEXT NOT NULL,
    speaker_scope TEXT NOT NULL,
    filename      TEXT NOT NULL,
    content       TEXT,
    updated_at    REAL,
    PRIMARY KEY (npc_id, speaker_scope, filename)
);

CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    npc_id     TEXT NOT NULL,
    speaker_id INTEGER,
    memory_id  TEXT NOT NULL,
    text       TEXT,
    source     TEXT,
    importance INTEGER,
    tags_json  TEXT,
    created_at REAL,
    UNIQUE (npc_id, memory_id)
);

CREATE TABLE IF NOT EXISTS waitlist (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    email      TEXT UNIQUE NOT NULL,
    note       TEXT,
    created_at REAL
);

CREATE INDEX IF NOT EXISTS idx_sessions_npc ON sessions (npc_id, started_at);
CREATE INDEX IF NOT EXISTS idx_memories_npc ON memories (npc_id);
"""


# ---------------------------------------------------------------------------
# Connection / init
# ---------------------------------------------------------------------------

def _resolve_path() -> str:
    """Resolve DB_PATH, falling back to ./engram.sqlite3 when the configured
    directory (e.g. /data on a box without the Modal volume) isn't writable."""
    global DB_PATH
    directory = os.path.dirname(DB_PATH) or "."
    try:
        os.makedirs(directory, exist_ok=True)
        if not os.access(directory, os.W_OK):
            raise PermissionError(directory)
    except Exception:
        DB_PATH = os.path.join(".", "engram.sqlite3")
    return DB_PATH


def _connect() -> sqlite3.Connection:
    """Lazy singleton connection (re-opened if DB_PATH changed, for tests)."""
    global _conn, _conn_path
    path = _resolve_path()
    if _conn is not None and _conn_path == path:
        return _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.OperationalError:
        pass  # WAL unsupported on some network filesystems; default is fine
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    conn.commit()
    _conn, _conn_path = conn, path
    return conn


def init_db() -> None:
    """Idempotent: open the connection and create any missing tables."""
    with _lock:
        _connect()


def _fire_on_write() -> None:
    hook = on_write
    if hook is None:
        return
    try:
        hook()
    except Exception:
        pass  # persistence hook must never break request handling


def _write(fn: Callable[[sqlite3.Connection], Any]) -> Any:
    """Run fn(conn) inside a committed transaction, then fire on_write."""
    with _lock:
        conn = _connect()
        try:
            result = fn(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    _fire_on_write()
    return result


def _read(fn: Callable[[sqlite3.Connection], Any]) -> Any:
    with _lock:
        conn = _connect()
        return fn(conn)


# ---------------------------------------------------------------------------
# Speakers
# ---------------------------------------------------------------------------

def get_or_create_speaker(device_id: str) -> dict:
    """Return the speaker row for device_id, creating it (with the next
    "Visitor N" label in first-seen order) when unseen. Updates last_seen."""
    if not device_id:
        raise ValueError("device_id required")
    now = time.time()

    def op(conn: sqlite3.Connection):
        row = conn.execute(
            "SELECT * FROM speakers WHERE device_id = ?", (device_id,)
        ).fetchone()
        if row is not None:
            conn.execute(
                "UPDATE speakers SET last_seen = ? WHERE speaker_id = ?",
                (now, row["speaker_id"]),
            )
            d = dict(row)
            d["last_seen"] = now
            return d
        n = conn.execute("SELECT COUNT(*) FROM speakers").fetchone()[0]
        label = f"Visitor {n + 1}"
        cur = conn.execute(
            "INSERT INTO speakers (device_id, label, first_seen, last_seen) "
            "VALUES (?, ?, ?, ?)",
            (device_id, label, now, now),
        )
        return {
            "speaker_id": cur.lastrowid,
            "device_id": device_id,
            "label": label,
            "first_seen": now,
            "last_seen": now,
        }

    return _write(op)


# ---------------------------------------------------------------------------
# NPC snapshots (drop-in for the old modal.Dict payloads)
# ---------------------------------------------------------------------------

def save_npc_snapshot(npc_id: str, scope: str, files: dict) -> None:
    """Store a snapshot payload (e.g. {memories, longterm, keystore, state}).

    Each field becomes one npc_files row; content is JSON-encoded so both
    dict/list fields and the raw keystore string round-trip exactly. Rows for
    fields absent from this snapshot are removed (true snapshot semantics).
    """
    if not npc_id or not scope or not isinstance(files, dict):
        return
    now = time.time()

    def op(conn: sqlite3.Connection):
        names = list(files.keys())
        if names:
            placeholders = ",".join("?" * len(names))
            conn.execute(
                f"DELETE FROM npc_files WHERE npc_id = ? AND speaker_scope = ? "
                f"AND filename NOT IN ({placeholders})",
                (npc_id, scope, *names),
            )
        else:
            conn.execute(
                "DELETE FROM npc_files WHERE npc_id = ? AND speaker_scope = ?",
                (npc_id, scope),
            )
        for filename, value in files.items():
            conn.execute(
                "INSERT INTO npc_files (npc_id, speaker_scope, filename, content, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (npc_id, speaker_scope, filename) "
                "DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at",
                (npc_id, scope, filename, json.dumps(value, ensure_ascii=False), now),
            )

    _write(op)


def load_npc_snapshot(npc_id: str, scope: str) -> Optional[dict]:
    """Return the snapshot payload dict, or None when nothing is stored."""
    if not npc_id or not scope:
        return None

    def op(conn: sqlite3.Connection):
        rows = conn.execute(
            "SELECT filename, content FROM npc_files "
            "WHERE npc_id = ? AND speaker_scope = ?",
            (npc_id, scope),
        ).fetchall()
        if not rows:
            return None
        out = {}
        for row in rows:
            try:
                out[row["filename"]] = json.loads(row["content"])
            except (TypeError, json.JSONDecodeError):
                out[row["filename"]] = row["content"]
        return out

    return _read(op)


# ---------------------------------------------------------------------------
# NPC registry + sessions
# ---------------------------------------------------------------------------

def record_session_start(
    session_id: str,
    npc_id: str,
    device_id: Optional[str],
    npc_name: Optional[str] = None,
    kind: Optional[str] = None,
) -> None:
    """Insert a session row (and upsert the npc registry row)."""
    now = time.time()
    speaker_id = None
    if device_id:
        speaker_id = get_or_create_speaker(device_id)["speaker_id"]

    def op(conn: sqlite3.Connection):
        if kind in ("preset", "custom"):
            conn.execute(
                "INSERT INTO npcs (npc_id, name, kind, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (npc_id) DO UPDATE SET "
                "name = excluded.name, kind = excluded.kind, updated_at = excluded.updated_at",
                (npc_id, npc_name or npc_id, kind, now, now),
            )
        conn.execute(
            "INSERT OR REPLACE INTO sessions "
            "(session_id, npc_id, speaker_id, started_at, ended_at, turn_count) "
            "VALUES (?, ?, ?, ?, NULL, 0)",
            (session_id, npc_id, speaker_id, now),
        )

    _write(op)


def record_session_end(session_id: str, turn_count: int) -> None:
    """Close a session. Idempotent; unknown session_ids are a no-op."""
    now = time.time()

    def op(conn: sqlite3.Connection):
        conn.execute(
            "UPDATE sessions SET ended_at = COALESCE(ended_at, ?), turn_count = ? "
            "WHERE session_id = ?",
            (now, int(turn_count or 0), session_id),
        )

    _write(op)


# ---------------------------------------------------------------------------
# Memory attribution ledger
# ---------------------------------------------------------------------------

def append_memories(npc_id: str, device_id: Optional[str], memories: list) -> int:
    """Upsert memory rows on (npc_id, memory_id). Returns rows written.

    Each item: {"memory_id" (or "id"), "text", "source", "importance",
    "tags" (dict, stored as JSON)}. The original speaker attribution is kept
    on conflict (a memory belongs to whoever was present when it formed)."""
    if not npc_id or not memories:
        return 0
    speaker_id = None
    if device_id:
        speaker_id = get_or_create_speaker(device_id)["speaker_id"]
    now = time.time()

    def op(conn: sqlite3.Connection):
        written = 0
        for m in memories:
            if not isinstance(m, dict):
                continue
            memory_id = str(m.get("memory_id") or m.get("id") or "").strip()
            if not memory_id:
                continue
            tags = m.get("tags")
            try:
                tags_json = json.dumps(tags, ensure_ascii=False) if tags is not None else None
            except (TypeError, ValueError):
                tags_json = None
            importance = m.get("importance")
            try:
                importance = int(importance) if importance is not None else None
            except (TypeError, ValueError):
                importance = None
            conn.execute(
                "INSERT INTO memories "
                "(npc_id, speaker_id, memory_id, text, source, importance, tags_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (npc_id, memory_id) DO UPDATE SET "
                "text = excluded.text, source = excluded.source, "
                "importance = excluded.importance, tags_json = excluded.tags_json",
                (npc_id, speaker_id, memory_id, str(m.get("text") or ""),
                 str(m.get("source") or ""), importance, tags_json, now),
            )
            written += 1
        return written

    return _write(op)


# ---------------------------------------------------------------------------
# Speaker context (for the SITUATIONAL CONTEXT persona line)
# ---------------------------------------------------------------------------

def speaker_context(npc_id: str, device_id: str) -> dict:
    """Who is talking to this NPC, and who else has.

    Call BEFORE record_session_start so "prior sessions" excludes the one
    being started. distinct_speakers always includes the current speaker."""
    speaker = get_or_create_speaker(device_id)
    sid = speaker["speaker_id"]

    def op(conn: sqlite3.Connection):
        prior = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE npc_id = ? AND speaker_id = ?",
            (npc_id, sid),
        ).fetchone()[0]
        has_snapshot = conn.execute(
            "SELECT 1 FROM npc_files WHERE npc_id = ? AND speaker_scope = ? LIMIT 1",
            (npc_id, device_id),
        ).fetchone() is not None
        distinct = conn.execute(
            "SELECT COUNT(DISTINCT speaker_id) FROM sessions "
            "WHERE npc_id = ? AND speaker_id IS NOT NULL",
            (npc_id,),
        ).fetchone()[0]
        if prior == 0:
            distinct += 1  # current speaker has no session row yet
        total = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE npc_id = ?", (npc_id,)
        ).fetchone()[0]
        others = [
            row["label"]
            for row in conn.execute(
                "SELECT sp.label AS label, MAX(se.started_at) AS latest "
                "FROM sessions se JOIN speakers sp ON sp.speaker_id = se.speaker_id "
                "WHERE se.npc_id = ? AND se.speaker_id != ? "
                "GROUP BY se.speaker_id ORDER BY latest DESC LIMIT 3",
                (npc_id, sid),
            ).fetchall()
        ]
        return {
            "speaker_label": speaker["label"],
            "is_returning": bool(prior > 0 or has_snapshot),
            "distinct_speakers": int(distinct),
            "total_sessions": int(total),
            "other_recent_speakers": others,
        }

    return _read(op)


# ---------------------------------------------------------------------------
# Waitlist
# ---------------------------------------------------------------------------

def add_waitlist(email: str, note: str = "") -> bool:
    """Insert an email; returns True when new, False when a duplicate."""
    email = (email or "").strip().lower()
    if not email:
        return False
    now = time.time()

    def op(conn: sqlite3.Connection):
        cur = conn.execute(
            "INSERT OR IGNORE INTO waitlist (email, note, created_at) VALUES (?, ?, ?)",
            (email, (note or "").strip(), now),
        )
        return cur.rowcount > 0

    return _write(op)


def list_waitlist() -> list:
    def op(conn: sqlite3.Connection):
        rows = conn.execute(
            "SELECT id, email, note, created_at FROM waitlist ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    return _read(op)
