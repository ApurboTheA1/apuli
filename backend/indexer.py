"""Incremental Obsidian vault indexer + live watchdog file watcher."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import numpy as np
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .config import AppConfig, DB_PATH, SYSTEM_EXCLUDED_DIRS
from .db import db_session, utc_now_iso
from .obsidian_parser import ParsedNote, parse_obsidian_note
from .ollama_client import embed_texts, has_model


@dataclass
class IndexStatus:
    state: str = "idle"  # idle | scanning | indexing | embedding | error
    total_notes: int = 0
    processed_notes: int = 0
    indexed_notes: int = 0
    total_chunks: int = 0
    embedded_chunks: int = 0
    embeddings_available: bool = False
    current_file: str = ""
    last_indexed_at: Optional[str] = None
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def is_excluded(rel_path: Path, excluded_folders: List[str]) -> bool:
    parts = rel_path.parts
    excluded_set = {d.lower() for d in SYSTEM_EXCLUDED_DIRS} | {
        d.strip().lower() for d in excluded_folders if d.strip()
    }
    for part in parts[:-1]:
        if part.lower() in excluded_set or part.startswith("."):
            return True
    if parts and parts[-1].startswith("."):
        return True
    return False


def discover_vault_folders(vault_path: Path) -> List[Dict[str, object]]:
    """List top-level non-hidden folders in the vault with markdown counts."""
    if not vault_path.exists():
        return []
    folders = []
    hidden = {d.lower() for d in SYSTEM_EXCLUDED_DIRS}
    for item in sorted(vault_path.iterdir(), key=lambda p: p.name.lower()):
        if not item.is_dir():
            continue
        if item.name.startswith(".") or item.name.lower() in hidden:
            continue
        md_count = sum(1 for _ in item.rglob("*.md"))
        folders.append({"name": item.name, "note_count": md_count})
    return folders


def discover_journal_candidates(vault_path: Path) -> List[Dict[str, object]]:
    """List folders & depth-2 subfolders containing markdown notes, flagging likely journal folders."""
    if not vault_path.exists():
        return []
    hidden = {d.lower() for d in SYSTEM_EXCLUDED_DIRS}
    hints = ("journal", "daily", "weekly", "monthly", "yearly", "diary", "reflection", "log")
    results: List[Dict[str, object]] = []

    for top in sorted(vault_path.iterdir(), key=lambda p: p.name.lower()):
        if not top.is_dir() or top.name.startswith(".") or top.name.lower() in hidden:
            continue
        top_count = sum(1 for _ in top.rglob("*.md"))
        if top_count > 0:
            results.append(
                {
                    "path": top.name,
                    "note_count": top_count,
                    "suggested": any(h in top.name.lower() for h in hints),
                }
            )
        for sub in sorted(top.iterdir(), key=lambda p: p.name.lower()):
            if not sub.is_dir() or sub.name.startswith(".") or sub.name.lower() in hidden:
                continue
            sub_count = sum(1 for _ in sub.rglob("*.md"))
            if sub_count > 0:
                rel = f"{top.name}/{sub.name}"
                results.append(
                    {
                        "path": rel,
                        "note_count": sub_count,
                        "suggested": any(h in rel.lower() for h in hints),
                    }
                )
    return results


def _vec_to_blob(vec: List[float]) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tobytes()


class VaultIndexer:
    """Manages vault indexing, embedding generation, and watchdog updates."""

    def __init__(
        self,
        get_config: Callable[[], AppConfig],
        db_path: Path = DB_PATH,
        on_index_updated: Optional[Callable[[], None]] = None,
    ) -> None:
        self._get_config = get_config
        self._db_path = db_path
        self._on_index_updated = on_index_updated
        self.status = IndexStatus()
        self._lock = asyncio.Lock()
        self._observer: Optional[Observer] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pending_paths: Set[Path] = set()
        self._pending_lock = threading.Lock()
        self._debounce_task: Optional[asyncio.Task] = None
        self._refresh_stats()

    def _refresh_stats(self) -> None:
        with db_session(self._db_path) as conn:
            notes_cnt = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            chunks_cnt = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            emb_cnt = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
            ).fetchone()[0]
            self.status.indexed_notes = notes_cnt
            self.status.total_chunks = chunks_cnt
            self.status.embedded_chunks = emb_cnt
            self.status.embeddings_available = emb_cnt > 0

    def _upsert_note_sync(
        self,
        parsed: ParsedNote,
        embeddings: Optional[List[List[float]]],
        embed_model: str,
    ) -> None:
        with db_session(self._db_path) as conn:
            old_chunk_ids = [
                r[0]
                for r in conn.execute(
                    "SELECT id FROM chunks WHERE note_path = ?", (parsed.rel_path,)
                ).fetchall()
            ]
            for cid in old_chunk_ids:
                conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (cid,))

            conn.execute("DELETE FROM chunks WHERE note_path = ?", (parsed.rel_path,))
            conn.execute(
                """
                INSERT INTO notes (
                    path, title, folder, note_date, tags_json, links_json,
                    frontmatter_json, mtime, content_hash, word_count, preview, indexed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    title = excluded.title,
                    folder = excluded.folder,
                    note_date = excluded.note_date,
                    tags_json = excluded.tags_json,
                    links_json = excluded.links_json,
                    frontmatter_json = excluded.frontmatter_json,
                    mtime = excluded.mtime,
                    content_hash = excluded.content_hash,
                    word_count = excluded.word_count,
                    preview = excluded.preview,
                    indexed_at = excluded.indexed_at
                """,
                (
                    parsed.rel_path,
                    parsed.title,
                    parsed.folder,
                    parsed.note_date,
                    json.dumps(parsed.tags, ensure_ascii=False),
                    json.dumps(parsed.links, ensure_ascii=False),
                    json.dumps(parsed.frontmatter, ensure_ascii=False),
                    parsed.mtime,
                    parsed.content_hash,
                    parsed.word_count,
                    parsed.preview,
                    utc_now_iso(),
                ),
            )

            tags_str = " ".join(parsed.tags)
            for i, chunk in enumerate(parsed.chunks):
                blob = None
                dim = 0
                used_model = ""
                if embeddings and i < len(embeddings):
                    blob = _vec_to_blob(embeddings[i])
                    dim = len(embeddings[i])
                    used_model = embed_model

                cur = conn.execute(
                    """
                    INSERT INTO chunks (
                        note_path, chunk_index, heading, content,
                        embedding, embedding_dim, embedding_model
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        parsed.rel_path,
                        chunk.chunk_index,
                        chunk.heading,
                        chunk.content,
                        blob,
                        dim,
                        used_model,
                    ),
                )
                chunk_id = cur.lastrowid
                conn.execute(
                    """
                    INSERT INTO chunks_fts (
                        chunk_id, note_path, title, folder, tags, heading, content
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        parsed.rel_path,
                        parsed.title,
                        parsed.folder,
                        tags_str,
                        chunk.heading,
                        chunk.content,
                    ),
                )

    def _delete_note_sync(self, rel_path: str) -> None:
        with db_session(self._db_path) as conn:
            old_ids = [
                r[0]
                for r in conn.execute(
                    "SELECT id FROM chunks WHERE note_path = ?", (rel_path,)
                ).fetchall()
            ]
            for cid in old_ids:
                conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (cid,))
            conn.execute("DELETE FROM notes WHERE path = ?", (rel_path,))

    async def index_vault(self, force_reembed: bool = False) -> IndexStatus:
        """Scan the Obsidian vault and incrementally index changed notes."""
        async with self._lock:
            cfg = self._get_config()
            vault_root = Path(cfg.vault_path)
            if not vault_root.exists():
                self.status.state = "error"
                self.status.message = f"Vault path not found: {vault_root}"
                return self.status

            self.status.state = "scanning"
            self.status.message = "Scanning Obsidian vault..."
            self.status.processed_notes = 0

            all_md: List[Path] = []
            for p in vault_root.rglob("*.md"):
                if not p.is_file():
                    continue
                rel = p.relative_to(vault_root)
                if not is_excluded(rel, cfg.excluded_folders):
                    all_md.append(p)

            self.status.total_notes = len(all_md)
            disk_rel_paths = {p.relative_to(vault_root).as_posix() for p in all_md}

            with db_session(self._db_path) as conn:
                existing = {
                    r["path"]: (r["mtime"], r["content_hash"])
                    for r in conn.execute(
                        "SELECT path, mtime, content_hash FROM notes"
                    ).fetchall()
                }

            # Remove deleted or newly-excluded notes
            removed = set(existing.keys()) - disk_rel_paths
            for rpath in removed:
                self._delete_note_sync(rpath)

            embed_ready = await has_model(cfg.ollama_url, cfg.embedding_model)
            self.status.state = "indexing"

            to_parse: List[Tuple[Path, Optional[Tuple[float, str]]]] = []
            for file_path in all_md:
                rel_posix = file_path.relative_to(vault_root).as_posix()
                prev = existing.get(rel_posix)
                stat = file_path.stat()
                if not force_reembed and prev and abs(prev[0] - stat.st_mtime) < 1e-4:
                    continue
                to_parse.append((file_path, prev))

            changed_notes: List[ParsedNote] = []
            mtime_only_updates: List[Tuple[float, str]] = []
            if to_parse:
                detected = os.cpu_count() or 4
                requested = int(getattr(cfg, "parse_threads", 0) or 0)
                workers = max(1, min(64, requested if requested > 0 else detected))
                self.status.message = f"Parsing {len(to_parse)} notes across {workers} CPU thread{'s' if workers != 1 else ''}..."
                loop = asyncio.get_running_loop()
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    parsed_results = await loop.run_in_executor(
                        None,
                        lambda: list(
                            pool.map(lambda item: parse_obsidian_note(item[0], vault_root), to_parse)
                        ),
                    )
                for idx, ((_, prev), parsed) in enumerate(zip(to_parse, parsed_results), start=1):
                    self.status.processed_notes = idx
                    self.status.current_file = parsed.rel_path
                    if not force_reembed and prev and prev[1] == parsed.content_hash:
                        mtime_only_updates.append((parsed.mtime, parsed.rel_path))
                    else:
                        changed_notes.append(parsed)

            if mtime_only_updates:
                with db_session(self._db_path) as conn:
                    conn.executemany(
                        "UPDATE notes SET mtime = ? WHERE path = ?",
                        mtime_only_updates,
                    )

            if changed_notes:
                self.status.message = f"Indexing {len(changed_notes)} updated notes..."
                for i, parsed in enumerate(changed_notes, start=1):
                    self.status.current_file = parsed.rel_path
                    self._upsert_note_sync(parsed, None, cfg.embedding_model)
                    if i % 25 == 0:
                        self._refresh_stats()

            # Embed all unembedded chunks in 48-chunk cross-note batches
            if embed_ready:
                await self._embed_missing_chunks(cfg)

            self._refresh_stats()
            self.status.state = "idle"
            self.status.current_file = ""
            self.status.last_indexed_at = utc_now_iso()
            if embed_ready:
                self.status.message = (
                    f"Ready — {self.status.indexed_notes} notes indexed with semantic + keyword memory."
                )
            else:
                self.status.message = (
                    f"{self.status.indexed_notes} notes indexed (keyword search ready; pull '{cfg.embedding_model}' for deeper semantic memory)."
                )

            if self._on_index_updated:
                self._on_index_updated()
            return self.status

    async def _embed_missing_chunks(self, cfg: AppConfig) -> None:
        with db_session(self._db_path) as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.heading, c.content, n.title, n.folder
                FROM chunks c
                JOIN notes n ON n.path = c.note_path
                WHERE c.embedding IS NULL
                """
            ).fetchall()
            pending = [dict(r) for r in rows]

        if not pending:
            return

        self.status.state = "embedding"
        batch_size = max(1, min(128, int(getattr(cfg, "embed_batch_size", 48) or 48)))
        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            self.status.message = (
                f"Building semantic memory ({min(i + batch_size, len(pending))}/{len(pending)} chunks)..."
            )
            texts = [
                f"Note: {r['title']} ({r['folder']})\nSection: {r['heading']}\n\n{r['content']}"
                for r in batch
            ]
            vectors = await embed_texts(
                cfg.ollama_url,
                cfg.embedding_model,
                texts,
                keep_alive=getattr(cfg, "ollama_keep_alive", "20m"),
            )
            if not vectors:
                break
            with db_session(self._db_path) as conn:
                for row, vec in zip(batch, vectors):
                    conn.execute(
                        """
                        UPDATE chunks
                        SET embedding = ?, embedding_dim = ?, embedding_model = ?
                        WHERE id = ?
                        """,
                        (
                            _vec_to_blob(vec),
                            len(vec),
                            cfg.embedding_model,
                            row["id"],
                        ),
                    )
            self._refresh_stats()

    async def embed_missing_only(self) -> IndexStatus:
        """Compute embeddings for existing chunks without re-parsing files."""
        async with self._lock:
            cfg = self._get_config()
            if not await has_model(cfg.ollama_url, cfg.embedding_model):
                self.status.message = (
                    f"Embedding model '{cfg.embedding_model}' not found in Ollama."
                )
                return self.status
            await self._embed_missing_chunks(cfg)
            self._refresh_stats()
            self.status.state = "idle"
            self.status.message = (
                f"Semantic memory ready ({self.status.embedded_chunks}/{self.status.total_chunks} chunks)."
            )
            if self._on_index_updated:
                self._on_index_updated()
            return self.status

    def start_watcher(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        cfg = self._get_config()
        vault_root = Path(cfg.vault_path)
        if not vault_root.exists():
            return
        self.stop_watcher()

        indexer = self

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event):
                if event.is_directory:
                    return
                for path_str in (
                    getattr(event, "src_path", None),
                    getattr(event, "dest_path", None),
                ):
                    if path_str and path_str.endswith(".md"):
                        indexer._queue_path(Path(path_str))

        observer = Observer()
        observer.schedule(Handler(), str(vault_root), recursive=True)
        observer.daemon = True
        observer.start()
        self._observer = observer

    def stop_watcher(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=2)
            self._observer = None

    def _queue_path(self, file_path: Path) -> None:
        with self._pending_lock:
            self._pending_paths.add(file_path)
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._schedule_debounce)

    def _schedule_debounce(self) -> None:
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
        self._debounce_task = asyncio.create_task(self._process_pending_debounced())

    async def _process_pending_debounced(self) -> None:
        await asyncio.sleep(1.5)
        with self._pending_lock:
            paths = list(self._pending_paths)
            self._pending_paths.clear()
        if not paths:
            return

        async with self._lock:
            cfg = self._get_config()
            vault_root = Path(cfg.vault_path)
            embed_ready = await has_model(cfg.ollama_url, cfg.embedding_model)
            changed = False
            for file_path in paths:
                try:
                    rel = file_path.relative_to(vault_root)
                except ValueError:
                    continue
                rel_posix = rel.as_posix()
                if not file_path.exists() or is_excluded(rel, cfg.excluded_folders):
                    self._delete_note_sync(rel_posix)
                    changed = True
                    continue
                parsed = parse_obsidian_note(file_path, vault_root)
                embeddings = None
                if embed_ready:
                    chunk_inputs = [
                        f"Note: {parsed.title} ({parsed.folder})\nSection: {c.heading}\n\n{c.content}"
                        for c in parsed.chunks
                    ]
                    embeddings = await embed_texts(
                        cfg.ollama_url, cfg.embedding_model, chunk_inputs
                    )
                self._upsert_note_sync(parsed, embeddings, cfg.embedding_model)
                changed = True

            if changed:
                self._refresh_stats()
                self.status.last_indexed_at = utc_now_iso()
                if self._on_index_updated:
                    self._on_index_updated()
