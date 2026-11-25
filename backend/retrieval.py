"""Hybrid vault + cross-chat retrieval: Vector similarity + FTS5 BM25 + Obsidian [[wikilink]] graph + recency."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import AppConfig, DB_PATH
from .db import db_session
from .ollama_client import embed_texts, has_model

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has", "have",
    "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "just", "me", "more", "my", "myself", "no", "not", "now",
    "of", "on", "or", "our", "ours", "out", "she", "should", "so", "some",
    "such", "tell", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "those", "to", "too", "up", "us", "very", "was",
    "we", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "you", "your", "yourself", "about", "remember", "know",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")
RECENCY_HINTS = (
    "recent", "recently", "lately", "today", "yesterday", "this week",
    "this month", "last few days", "current", "currently",
)


@dataclass
class RetrievedChunk:
    chunk_id: int
    note_path: str
    title: str
    folder: str
    note_date: Optional[str]
    heading: str
    content: str
    tags: List[str]
    links: List[str]
    score: float
    match_reason: str


@dataclass
class RetrievedChatChunk:
    chat_chunk_id: int
    conversation_id: str
    conv_title: str
    content: str
    created_at: str
    score: float
    match_reason: str


def _vec_to_blob(vec: List[float]) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tobytes()


class HybridRetriever:
    """In-memory vector cache + SQLite FTS5 hybrid retriever for notes and past chats."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path
        self._chunk_ids: List[int] = []
        self._chunk_folders: List[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._dim: int = 0

        self._chat_ids: List[int] = []
        self._chat_convs: List[str] = []
        self._chat_matrix: Optional[np.ndarray] = None
        self._chat_dim: int = 0

        self.reload_vectors()

    def reload_vectors(self) -> None:
        with db_session(self._db_path) as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.embedding, c.embedding_dim, n.folder
                FROM chunks c
                JOIN notes n ON n.path = c.note_path
                WHERE c.embedding IS NOT NULL AND c.embedding_dim > 0
                ORDER BY c.id ASC
                """
            ).fetchall()

            chat_rows = conn.execute(
                """
                SELECT id, conversation_id, embedding, embedding_dim
                FROM chat_chunks
                WHERE embedding IS NOT NULL AND embedding_dim > 0
                ORDER BY id ASC
                """
            ).fetchall()

        # Vault note vectors
        if rows:
            dim = rows[-1]["embedding_dim"]
            ids: List[int] = []
            folders: List[str] = []
            vecs: List[np.ndarray] = []
            for r in rows:
                if r["embedding_dim"] != dim:
                    continue
                arr = np.frombuffer(r["embedding"], dtype=np.float32)
                if arr.shape[0] == dim:
                    ids.append(r["id"])
                    folders.append(r["folder"])
                    vecs.append(arr)
            if vecs:
                self._chunk_ids = ids
                self._chunk_folders = folders
                self._matrix = np.vstack(vecs)
                self._dim = dim
            else:
                self._chunk_ids, self._chunk_folders, self._matrix, self._dim = [], [], None, 0
        else:
            self._chunk_ids, self._chunk_folders, self._matrix, self._dim = [], [], None, 0

        # Past chat vectors
        if chat_rows:
            cdim = chat_rows[-1]["embedding_dim"]
            cids: List[int] = []
            cconvs: List[str] = []
            cvecs: List[np.ndarray] = []
            for r in chat_rows:
                if r["embedding_dim"] != cdim:
                    continue
                arr = np.frombuffer(r["embedding"], dtype=np.float32)
                if arr.shape[0] == cdim:
                    cids.append(r["id"])
                    cconvs.append(r["conversation_id"])
                    cvecs.append(arr)
            if cvecs:
                self._chat_ids = cids
                self._chat_convs = cconvs
                self._chat_matrix = np.vstack(cvecs)
                self._chat_dim = cdim
            else:
                self._chat_ids, self._chat_convs, self._chat_matrix, self._chat_dim = [], [], None, 0
        else:
            self._chat_ids, self._chat_convs, self._chat_matrix, self._chat_dim = [], [], None, 0

    def _build_fts_query(self, query: str) -> str:
        tokens = [
            t.lower()
            for t in TOKEN_RE.findall(query)
            if t.lower() not in STOP_WORDS
        ]
        if not tokens:
            tokens = [t.lower() for t in TOKEN_RE.findall(query)]
        seen = set()
        clean = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                clean.append(f'"{t}"')
        return " OR ".join(clean[:12])

    def _fts_search(
        self, query: str, folder_scope: str, limit: int = 24
    ) -> List[Tuple[int, float]]:
        fts_query = self._build_fts_query(query)
        if not fts_query:
            return []

        params: List[object] = [fts_query]
        folder_clause = ""
        if folder_scope and folder_scope != "all":
            folders = [f.strip() for f in folder_scope.split(",") if f.strip()]
            if folders:
                placeholders = ",".join("?" for _ in folders)
                folder_clause = f"AND n.folder IN ({placeholders})"
                params.extend(folders)
        params.append(limit)

        sql = f"""
            SELECT f.chunk_id, bm25(chunks_fts, 1.0, 3.0, 1.5, 2.0, 2.0, 1.0) AS rank
            FROM chunks_fts f
            JOIN notes n ON n.path = f.note_path
            WHERE chunks_fts MATCH ? {folder_clause}
            ORDER BY rank ASC
            LIMIT ?
        """
        with db_session(self._db_path) as conn:
            try:
                rows = conn.execute(sql, params).fetchall()
                return [(int(r["chunk_id"]), float(r["rank"])) for r in rows]
            except Exception:
                return []

    def _vector_search(
        self, query_vec: List[float], folder_scope: str, limit: int = 24
    ) -> List[Tuple[int, float]]:
        if self._matrix is None or len(query_vec) != self._dim:
            return []

        q = np.asarray(query_vec, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        sims = self._matrix @ q

        if folder_scope and folder_scope != "all":
            allowed = {f.strip().lower() for f in folder_scope.split(",") if f.strip()}
            mask = np.array(
                [f.lower() in allowed for f in self._chunk_folders], dtype=bool
            )
            sims = np.where(mask, sims, -1.0)

        top_n = min(limit, len(self._chunk_ids))
        if top_n <= 0:
            return []
        top_idx = np.argpartition(-sims, top_n - 1)[:top_n]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results: List[Tuple[int, float]] = []
        for idx in top_idx:
            score = float(sims[idx])
            if score > 0.05:
                results.append((self._chunk_ids[idx], score))
        return results

    def _recent_chunks(
        self, folder_scope: str, limit: int = 8
    ) -> List[int]:
        params: List[object] = []
        folder_clause = ""
        if folder_scope and folder_scope != "all":
            folders = [f.strip() for f in folder_scope.split(",") if f.strip()]
            if folders:
                placeholders = ",".join("?" for _ in folders)
                folder_clause = f"WHERE n.folder IN ({placeholders})"
                params.extend(folders)
        params.append(limit)
        with db_session(self._db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT c.id
                FROM chunks c
                JOIN notes n ON n.path = c.note_path
                {folder_clause}
                ORDER BY COALESCE(n.note_date, '') DESC, n.mtime DESC, c.chunk_index ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [int(r[0]) for r in rows]

    def _fetch_chunks_by_ids(self, chunk_ids: List[int]) -> Dict[int, dict]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        with db_session(self._db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT c.id AS chunk_id, c.note_path, c.heading, c.content,
                       n.title, n.folder, n.note_date, n.tags_json, n.links_json
                FROM chunks c
                JOIN notes n ON n.path = c.note_path
                WHERE c.id IN ({placeholders})
                """,
                chunk_ids,
            ).fetchall()
        out: Dict[int, dict] = {}
        for r in rows:
            item = dict(r)
            item["tags"] = json.loads(item.pop("tags_json") or "[]")
            item["links"] = json.loads(item.pop("links_json") or "[]")
            out[int(item["chunk_id"])] = item
        return out

    def _expand_wikilinks(
        self, top_items: List[dict], seen_paths: set, limit: int = 2
    ) -> List[RetrievedChunk]:
        link_titles: List[str] = []
        for item in top_items[:4]:
            for link in item.get("links", []):
                if link not in link_titles:
                    link_titles.append(link)
        if not link_titles:
            return []

        placeholders = ",".join("?" for _ in link_titles[:10])
        with db_session(self._db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT c.id AS chunk_id, c.note_path, c.heading, c.content,
                       n.title, n.folder, n.note_date, n.tags_json, n.links_json
                FROM notes n
                JOIN chunks c ON c.note_path = n.path AND c.chunk_index = 0
                WHERE LOWER(n.title) IN ({placeholders})
                LIMIT ?
                """,
                [t.lower() for t in link_titles[:10]] + [limit + 4],
            ).fetchall()

        linked: List[RetrievedChunk] = []
        for r in rows:
            if r["note_path"] in seen_paths:
                continue
            seen_paths.add(r["note_path"])
            linked.append(
                RetrievedChunk(
                    chunk_id=int(r["chunk_id"]),
                    note_path=r["note_path"],
                    title=r["title"],
                    folder=r["folder"],
                    note_date=r["note_date"],
                    heading=r["heading"],
                    content=r["content"],
                    tags=json.loads(r["tags_json"] or "[]"),
                    links=json.loads(r["links_json"] or "[]"),
                    score=0.35,
                    match_reason="linked note",
                )
            )
            if len(linked) >= limit:
                break
        return linked

    async def index_chat_exchange(
        self,
        conversation_id: str,
        message_id: int,
        conv_title: str,
        user_text: str,
        assistant_text: str,
        created_at: str,
        cfg: AppConfig,
    ) -> None:
        """Index a completed user+assistant exchange into chat_chunks for cross-chat recall."""
        combined = f"{cfg.user_name}: {user_text.strip()}\n{cfg.companion_name}: {assistant_text.strip()}"
        if len(combined) > 1800:
            combined = combined[:1800]

        blob = None
        dim = 0
        used_model = ""
        if await has_model(cfg.ollama_url, cfg.embedding_model):
            vecs = await embed_texts(cfg.ollama_url, cfg.embedding_model, [combined])
            if vecs:
                blob = _vec_to_blob(vecs[0])
                dim = len(vecs[0])
                used_model = cfg.embedding_model

        with db_session(self._db_path) as conn:
            existing = conn.execute(
                "SELECT id FROM chat_chunks WHERE message_id = ?", (message_id,)
            ).fetchone()
            if existing:
                return
            cur = conn.execute(
                """
                INSERT INTO chat_chunks (
                    conversation_id, message_id, conv_title, content,
                    created_at, embedding, embedding_dim, embedding_model
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    message_id,
                    conv_title,
                    combined,
                    created_at,
                    blob,
                    dim,
                    used_model,
                ),
            )
            chat_chunk_id = cur.lastrowid
            conn.execute(
                """
                INSERT INTO chat_chunks_fts (
                    chat_chunk_id, conversation_id, conv_title, content
                )
                VALUES (?, ?, ?, ?)
                """,
                (chat_chunk_id, conversation_id, conv_title, combined),
            )
        self.reload_vectors()

    async def backfill_chat_chunks(self, cfg: AppConfig) -> int:
        """Ensure any older conversations stored before upgrade are indexed for cross-chat recall."""
        with db_session(self._db_path) as conn:
            rows = conn.execute(
                """
                SELECT m.id, m.conversation_id, m.role, m.content, m.created_at, c.title
                FROM messages m
                JOIN conversations c ON c.id = m.conversation_id
                ORDER BY m.conversation_id, m.id ASC
                """
            ).fetchall()
            already = {
                r[0]
                for r in conn.execute("SELECT message_id FROM chat_chunks").fetchall()
            }

        pairs = []
        last_user_by_conv: Dict[str, str] = {}
        for r in rows:
            if r["role"] == "user":
                last_user_by_conv[r["conversation_id"]] = r["content"]
            elif r["role"] == "assistant":
                if r["id"] not in already:
                    user_txt = last_user_by_conv.get(r["conversation_id"], "")
                    pairs.append(
                        (
                            r["conversation_id"],
                            r["id"],
                            r["title"],
                            user_txt,
                            r["content"],
                            r["created_at"],
                        )
                    )

        for conv_id, msg_id, title, utxt, atxt, cat in pairs:
            await self.index_chat_exchange(conv_id, msg_id, title, utxt, atxt, cat, cfg)
        return len(pairs)

    async def retrieve_past_chats(
        self,
        query: str,
        cfg: AppConfig,
        exclude_conv_id: Optional[str] = None,
        top_k: int = 3,
        query_vec: Optional[List[float]] = None,
    ) -> List[RetrievedChatChunk]:
        """Search past conversations (excluding the current active one)."""
        if not cfg.enable_cross_chat_recall:
            return []

        fts_query = self._build_fts_query(query)
        fts_hits: List[Tuple[int, float]] = []
        if fts_query:
            params: List[object] = [fts_query]
            excl_sql = ""
            if exclude_conv_id:
                excl_sql = "AND conversation_id != ?"
                params.append(exclude_conv_id)
            params.append(12)
            with db_session(self._db_path) as conn:
                try:
                    rows = conn.execute(
                        f"""
                        SELECT chat_chunk_id, bm25(chat_chunks_fts, 1.5, 1.0) AS rank
                        FROM chat_chunks_fts
                        WHERE chat_chunks_fts MATCH ? {excl_sql}
                        ORDER BY rank ASC
                        LIMIT ?
                        """,
                        params,
                    ).fetchall()
                    fts_hits = [(int(r["chat_chunk_id"]), float(r["rank"])) for r in rows]
                except Exception:
                    fts_hits = []

        vec_hits: List[Tuple[int, float]] = []
        if self._chat_matrix is not None:
            if query_vec is None and await has_model(cfg.ollama_url, cfg.embedding_model):
                vecs = await embed_texts(cfg.ollama_url, cfg.embedding_model, [query])
                if vecs:
                    query_vec = vecs[0]
            if query_vec is not None and len(query_vec) == self._chat_dim:
                q = np.asarray(query_vec, dtype=np.float32)
                norm = np.linalg.norm(q)
                if norm > 0:
                    q = q / norm
                sims = self._chat_matrix @ q
                if exclude_conv_id:
                    mask = np.array(
                        [cid != exclude_conv_id for cid in self._chat_convs], dtype=bool
                    )
                    sims = np.where(mask, sims, -1.0)
                top_n = min(12, len(self._chat_ids))
                if top_n > 0:
                    top_idx = np.argpartition(-sims, top_n - 1)[:top_n]
                    top_idx = top_idx[np.argsort(-sims[top_idx])]
                    for idx in top_idx:
                        if float(sims[idx]) > 0.25:
                            vec_hits.append((self._chat_ids[idx], float(sims[idx])))

        scores: Dict[int, float] = {}
        reasons: Dict[int, List[str]] = {}
        rrf_k = 40.0
        for rank, (cid, _) in enumerate(vec_hits):
            scores[cid] = scores.get(cid, 0.0) + (1.2 / (rrf_k + rank + 1))
            reasons.setdefault(cid, []).append("semantic")
        for rank, (cid, _) in enumerate(fts_hits):
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))
            reasons.setdefault(cid, []).append("keyword")

        if not scores:
            return []

        ranked_ids = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)[: top_k * 2]
        placeholders = ",".join("?" for _ in ranked_ids)
        with db_session(self._db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT id, conversation_id, conv_title, content, created_at
                FROM chat_chunks
                WHERE id IN ({placeholders})
                """,
                ranked_ids,
            ).fetchall()
        row_map = {int(r["id"]): dict(r) for r in rows}

        out: List[RetrievedChatChunk] = []
        seen_convs = set()
        for cid in ranked_ids:
            item = row_map.get(cid)
            if not item or item["conversation_id"] in seen_convs:
                continue
            seen_convs.add(item["conversation_id"])
            out.append(
                RetrievedChatChunk(
                    chat_chunk_id=cid,
                    conversation_id=item["conversation_id"],
                    conv_title=item["conv_title"],
                    content=item["content"],
                    created_at=item["created_at"],
                    score=round(scores[cid], 4),
                    match_reason=" + ".join(dict.fromkeys(reasons.get(cid, ["past chat"]))),
                )
            )
            if len(out) >= top_k:
                break
        return out

    async def retrieve(
        self,
        query: str,
        cfg: AppConfig,
        folder_scope: str = "all",
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        k = top_k or cfg.top_k_chunks
        fts_hits = self._fts_search(query, folder_scope, limit=24)

        vec_hits: List[Tuple[int, float]] = []
        if self._matrix is not None and await has_model(
            cfg.ollama_url, cfg.embedding_model
        ):
            vecs = await embed_texts(cfg.ollama_url, cfg.embedding_model, [query])
            if vecs:
                vec_hits = self._vector_search(vecs[0], folder_scope, limit=24)

        scores: Dict[int, float] = {}
        reasons: Dict[int, List[str]] = {}
        rrf_k = 40.0

        for rank, (cid, _) in enumerate(vec_hits):
            scores[cid] = scores.get(cid, 0.0) + (1.25 / (rrf_k + rank + 1))
            reasons.setdefault(cid, []).append("semantic")

        for rank, (cid, _) in enumerate(fts_hits):
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (rrf_k + rank + 1))
            reasons.setdefault(cid, []).append("keyword")

        lower_q = query.lower()
        wants_recent = any(h in lower_q for h in RECENCY_HINTS)
        if wants_recent or not scores:
            recent_ids = self._recent_chunks(folder_scope, limit=6)
            for rank, cid in enumerate(recent_ids):
                boost = 0.8 if not scores else 0.5
                scores[cid] = scores.get(cid, 0.0) + (boost / (rrf_k + rank + 1))
                reasons.setdefault(cid, []).append("recent")

        if not scores:
            return []

        ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        candidate_ids = ranked_ids[: k * 3]
        details = self._fetch_chunks_by_ids(candidate_ids)

        today_year = datetime.now().year
        adjusted: List[Tuple[int, float]] = []
        for cid in candidate_ids:
            item = details.get(cid)
            if not item:
                continue
            sc = scores[cid]
            if folder_scope == "all" and item["folder"] in ("Personal", "People"):
                sc *= 1.12
            if wants_recent and item.get("note_date", "").startswith(str(today_year)):
                sc *= 1.15
            adjusted.append((cid, sc))

        adjusted.sort(key=lambda x: x[1], reverse=True)

        selected: List[RetrievedChunk] = []
        per_note_count: Dict[str, int] = {}
        selected_items: List[dict] = []
        seen_paths: set = set()

        for cid, sc in adjusted:
            item = details[cid]
            npath = item["note_path"]
            if per_note_count.get(npath, 0) >= 2:
                continue
            per_note_count[npath] = per_note_count.get(npath, 0) + 1
            seen_paths.add(npath)
            selected_items.append(item)
            selected.append(
                RetrievedChunk(
                    chunk_id=cid,
                    note_path=npath,
                    title=item["title"],
                    folder=item["folder"],
                    note_date=item["note_date"],
                    heading=item["heading"],
                    content=item["content"],
                    tags=item["tags"],
                    links=item["links"],
                    score=round(sc, 4),
                    match_reason=" + ".join(dict.fromkeys(reasons.get(cid, ["match"]))),
                )
            )
            if len(selected) >= k:
                break

        if cfg.enable_graph_expansion and selected_items:
            linked = self._expand_wikilinks(selected_items, seen_paths, limit=2)
            selected.extend(linked)

        return selected
