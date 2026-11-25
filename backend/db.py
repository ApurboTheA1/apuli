"""SQLite + FTS5 persistence layer for vault index, chat recall, memories, and self-portrait."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .config import DATA_DIR, DB_PATH


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def db_session(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_column(
    conn: sqlite3.Connection, table: str, column: str, definition: str
) -> None:
    cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db(db_path: Path = DB_PATH) -> None:
    """Create all tables and FTS5 indexes, and run safe schema migrations."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with db_session(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS notes (
                path TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                folder TEXT NOT NULL,
                note_date TEXT,
                tags_json TEXT NOT NULL DEFAULT '[]',
                links_json TEXT NOT NULL DEFAULT '[]',
                frontmatter_json TEXT NOT NULL DEFAULT '{}',
                mtime REAL NOT NULL,
                content_hash TEXT NOT NULL,
                word_count INTEGER NOT NULL DEFAULT 0,
                preview TEXT NOT NULL DEFAULT '',
                indexed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                heading TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                embedding BLOB,
                embedding_dim INTEGER NOT NULL DEFAULT 0,
                embedding_model TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (note_path) REFERENCES notes(path) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_note_path ON chunks(note_path);
            CREATE INDEX IF NOT EXISTS idx_notes_folder ON notes(folder);
            CREATE INDEX IF NOT EXISTS idx_notes_date ON notes(note_date);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                note_path UNINDEXED,
                title,
                folder,
                tags,
                heading,
                content,
                tokenize='porter unicode61'
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'companion',
                model TEXT NOT NULL DEFAULT 'qwen2.5:7b',
                folder_scope TEXT NOT NULL DEFAULT 'all',
                last_distilled_msg_id INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                thinking TEXT NOT NULL DEFAULT '',
                sources_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);

            CREATE TABLE IF NOT EXISTS chat_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                message_id INTEGER NOT NULL UNIQUE,
                conv_title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                embedding BLOB,
                embedding_dim INTEGER NOT NULL DEFAULT 0,
                embedding_model TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chat_chunks_conv ON chat_chunks(conversation_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS chat_chunks_fts USING fts5(
                chat_chunk_id UNINDEXED,
                conversation_id UNINDEXED,
                conv_title,
                content,
                tokenize='porter unicode61'
            );

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL DEFAULT 'insight',
                content TEXT NOT NULL,
                source_info TEXT NOT NULL DEFAULT '',
                pinned INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS self_profile (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                summary_md TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                note_count_snapshot INTEGER NOT NULL DEFAULT 0,
                memory_count_snapshot INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        # Safe migrations for pre-existing databases
        _ensure_column(
            conn, "conversations", "last_distilled_msg_id", "INTEGER NOT NULL DEFAULT 0"
        )
        _ensure_column(
            conn, "self_profile", "note_count_snapshot", "INTEGER NOT NULL DEFAULT 0"
        )
        _ensure_column(
            conn, "self_profile", "memory_count_snapshot", "INTEGER NOT NULL DEFAULT 0"
        )


def list_conversations(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    with db_session(db_path) as conn:
        rows = conn.execute(
            """
            SELECT c.*, COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]


def get_conversation_messages(
    conversation_id: str, db_path: Path = DB_PATH
) -> List[Dict[str, Any]]:
    with db_session(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, conversation_id, role, content, thinking, sources_json, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()
        result = []
        for r in rows:
            item = dict(r)
            try:
                item["sources"] = json.loads(item.pop("sources_json") or "[]")
            except Exception:
                item["sources"] = []
            result.append(item)
        return result


def get_memories(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    with db_session(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, category, content, source_info, pinned, created_at
            FROM memories
            ORDER BY pinned DESC, id DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]


def get_self_profile(db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    with db_session(db_path) as conn:
        row = conn.execute(
            """
            SELECT summary_md, updated_at, note_count_snapshot, memory_count_snapshot
            FROM self_profile WHERE id = 1
            """
        ).fetchone()
        return dict(row) if row else None
