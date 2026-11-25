"""FastAPI application for Apuli — Local-First Obsidian Companion."""

from __future__ import annotations

import asyncio
import json
import os
import string
import urllib.parse
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .analytics import compute_vault_analytics, get_on_this_day, parse_temporal_filter
from .config import DB_PATH, load_config, save_config
from .db import (
    clear_vault_data,
    db_session,
    get_conversation_messages,
    get_memories,
    get_self_profile,
    init_db,
    list_conversations,
    set_ram_cache_enabled,
    utc_now_iso,
)
from .hardware import detect_hardware, recommend_models
from .indexer import VaultIndexer, discover_journal_candidates, discover_vault_folders
from .memory import (
    build_system_prompt,
    deep_synthesize_self_profile_stream,
    distill_memories_from_conversation,
    maybe_auto_distill,
    maybe_evolve_self_profile,
    save_reflection_to_vault,
    synthesize_self_profile,
)
from .ollama_client import check_ollama, pull_model_stream
from .providers import list_provider_models, provider_chat_stream
from .retrieval import HybridRetriever

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

init_db(DB_PATH)
set_ram_cache_enabled(load_config().enable_ram_cache)
retriever = HybridRetriever(DB_PATH)
_analytics_cache: Optional[Dict[str, Any]] = None


def _on_vault_index_updated() -> None:
    global _analytics_cache
    _analytics_cache = None
    retriever.reload_vectors()


indexer = VaultIndexer(load_config, DB_PATH, on_index_updated=_on_vault_index_updated)
_index_task: Optional[asyncio.Task] = None
_deep_task: Optional[asyncio.Task] = None
_bg_tasks: set[asyncio.Task] = set()
_last_learned_at: Optional[str] = None
_deep_status: dict[str, Any] = {
    "state": "idle",  # idle | reading | reducing | error | stopped
    "notes_processed": 0,
    "total_notes": 0,
    "message": "",
    "updated_at": None,
}


def _track_task(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
    return task


async def _run_deep_synthesis_bg() -> None:
    global _deep_status, _last_learned_at
    cfg = load_config()
    if cfg.provider == "ollama":
        ollama = await check_ollama(cfg.ollama_url)
        if not ollama.get("online"):
            _deep_status = {
                "state": "idle",
                "notes_processed": 0,
                "total_notes": indexer.status.indexed_notes,
                "message": "Vault indexed! Start Ollama (or add an API key) to build your deep Self-Portrait.",
                "updated_at": utc_now_iso(),
            }
            return

    _deep_status = {
        "state": "reading",
        "notes_processed": 0,
        "total_notes": indexer.status.indexed_notes,
        "message": "Starting background note-by-note reading…",
        "updated_at": utc_now_iso(),
    }
    try:
        async for item in deep_synthesize_self_profile_stream(cfg, db_path=DB_PATH):
            stage = item.get("stage", "reading")
            if stage == "done":
                _last_learned_at = utc_now_iso()
                _deep_status = {
                    "state": "idle",
                    "notes_processed": item.get("total_notes", 0),
                    "total_notes": item.get("total_notes", 0),
                    "message": f"Deep Self-Portrait ready ({item.get('total_notes', 0)} notes).",
                    "updated_at": _last_learned_at,
                }
            elif stage == "error":
                _deep_status = {
                    "state": "error",
                    "notes_processed": 0,
                    "total_notes": 0,
                    "message": item.get("error", "Deep synthesis failed."),
                    "updated_at": utc_now_iso(),
                }
            else:
                _deep_status = {
                    "state": stage,
                    "notes_processed": item.get("notes_processed", 0),
                    "total_notes": item.get("total_notes", 0),
                    "message": item.get("message", "Reading vault…"),
                    "updated_at": utc_now_iso(),
                }
    except asyncio.CancelledError:
        _deep_status = {
            "state": "stopped",
            "notes_processed": _deep_status.get("notes_processed", 0),
            "total_notes": _deep_status.get("total_notes", 0),
            "message": "Paused — progress saved and resumable anytime.",
            "updated_at": utc_now_iso(),
        }
        raise
    except Exception as exc:
        _deep_status = {
            "state": "error",
            "notes_processed": _deep_status.get("notes_processed", 0),
            "total_notes": _deep_status.get("total_notes", 0),
            "message": str(exc),
            "updated_at": utc_now_iso(),
        }


def _start_deep_bg() -> bool:
    global _deep_task
    if _deep_task and not _deep_task.done():
        return False
    _deep_task = asyncio.create_task(_run_deep_synthesis_bg())
    return True


async def _run_vault_pipeline(force_reembed: bool = False, run_deep: bool = False) -> None:
    await indexer.index_vault(force_reembed=force_reembed)
    cfg = load_config()
    if run_deep and cfg.auto_learn:
        _start_deep_bg()
    else:
        await maybe_evolve_self_profile(cfg, DB_PATH)


def _detect_candidate_vaults() -> List[str]:
    home = Path.home()
    candidates = [
        home / "Documents" / "Obsidian",
        home / "Obsidian",
        home / "Documents" / "Obsidian Vault",
        home / "OneDrive" / "Documents" / "Obsidian",
    ]
    found: List[str] = []
    for p in candidates:
        if p.exists() and p.is_dir():
            found.append(str(p))
    return found


async def _startup_tasks() -> None:
    cfg = load_config()
    await retriever.backfill_chat_chunks(cfg)
    if cfg.vault_path and Path(cfg.vault_path).exists():
        await indexer.index_vault(force_reembed=False)
        await maybe_evolve_self_profile(cfg, DB_PATH)


async def _background_learn_after_turn(
    conv_id: str,
    assistant_msg_id: int,
    conv_title: str,
    user_text: str,
    assistant_text: str,
    created_at: str,
    model: str,
) -> None:
    global _last_learned_at
    cfg = load_config()
    try:
        await retriever.index_chat_exchange(
            conversation_id=conv_id,
            message_id=assistant_msg_id,
            conv_title=conv_title,
            user_text=user_text,
            assistant_text=assistant_text,
            created_at=created_at,
            cfg=cfg,
        )
        saved = await maybe_auto_distill(
            conv_id, cfg, model=cfg.memory_model or model, db_path=DB_PATH
        )
        if saved:
            _last_learned_at = utc_now_iso()
        profile_res = await maybe_evolve_self_profile(cfg, DB_PATH)
        if profile_res:
            _last_learned_at = utc_now_iso()
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    cfg = load_config()
    if cfg.vault_path and Path(cfg.vault_path).exists():
        indexer.start_watcher(loop)
    global _index_task
    _index_task = asyncio.create_task(_startup_tasks())
    try:
        yield
    finally:
        indexer.stop_watcher()
        if _index_task and not _index_task.done():
            _index_task.cancel()
        if _deep_task and not _deep_task.done():
            _deep_task.cancel()
        for t in list(_bg_tasks):
            t.cancel()


app = FastAPI(title="Apuli — Local-First Obsidian Companion", lifespan=lifespan)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# --- Pydantic Models ---


class SettingsUpdate(BaseModel):
    vault_path: Optional[str] = None
    ollama_url: Optional[str] = None
    provider: Optional[str] = None
    chat_model: Optional[str] = None
    memory_model: Optional[str] = None
    embedding_model: Optional[str] = None
    excluded_folders: Optional[List[str]] = None
    journal_folders: Optional[List[str]] = None
    user_name: Optional[str] = None
    companion_name: Optional[str] = None
    default_mode: Optional[str] = None
    top_k_chunks: Optional[int] = None
    enable_graph_expansion: Optional[bool] = None
    temperature: Optional[float] = None
    auto_learn: Optional[bool] = None
    auto_distill_turns: Optional[int] = None
    enable_cross_chat_recall: Optional[bool] = None
    ui_font: Optional[str] = None
    font_size: Optional[int] = None
    parse_threads: Optional[int] = None
    embed_batch_size: Optional[int] = None
    enable_ram_cache: Optional[bool] = None
    ollama_keep_alive: Optional[str] = None
    ollama_num_thread: Optional[int] = None
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    anthropic_model: Optional[str] = None
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    custom_base_url: Optional[str] = None
    custom_api_key: Optional[str] = None
    custom_model: Optional[str] = None
    clear_openai_key: bool = False
    clear_anthropic_key: bool = False
    clear_gemini_key: bool = False
    clear_custom_key: bool = False


class PullModelReq(BaseModel):
    model: str


class CreateConversationReq(BaseModel):
    title: Optional[str] = None
    mode: str = "companion"
    model: Optional[str] = None
    folder_scope: str = "all"


class ChatReq(BaseModel):
    message: str
    mode: Optional[str] = None
    model: Optional[str] = None
    folder_scope: Optional[str] = None


class MemoryCreateReq(BaseModel):
    category: str = "insight"
    content: str
    pinned: bool = True


class MemoryUpdateReq(BaseModel):
    category: Optional[str] = None
    content: Optional[str] = None
    pinned: Optional[bool] = None


class SelfProfileUpdateReq(BaseModel):
    summary_md: str


class ExportReflectionReq(BaseModel):
    title: str
    content: str
    subfolder: str = "Personal"


class SwitchVaultReq(BaseModel):
    vault_path: str


def _obsidian_uri(vault_path: str, note_rel_path: str) -> str:
    if not vault_path:
        return "#"
    vault_name = Path(vault_path).name
    file_no_ext = note_rel_path[:-3] if note_rel_path.endswith(".md") else note_rel_path
    return (
        f"obsidian://open?vault={urllib.parse.quote(vault_name)}"
        f"&file={urllib.parse.quote(file_no_ext)}"
    )


def _sse_event(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# --- Routes ---


@app.get("/")
async def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_file)


@app.get("/favicon.ico")
async def serve_favicon():
    icon_file = FRONTEND_DIR / "apuli-icon.svg"
    if not icon_file.exists():
        raise HTTPException(status_code=404, detail="Icon not found")
    return FileResponse(icon_file, media_type="image/svg+xml")


def _get_cached_analytics(cfg) -> Dict[str, Any]:
    global _analytics_cache
    if _analytics_cache is None:
        raw = compute_vault_analytics(DB_PATH, cfg.journal_folders)
        for item in raw.get("on_this_day", []):
            item["obsidian_uri"] = _obsidian_uri(cfg.vault_path, item["note_path"])
        for rel in raw.get("relationships", []):
            rel["obsidian_uri"] = _obsidian_uri(cfg.vault_path, rel["note_path"])
        for cl in raw.get("clusters", []):
            for rep in cl.get("representative_notes", []):
                rep["obsidian_uri"] = _obsidian_uri(cfg.vault_path, rep["note_path"])
        _analytics_cache = raw
    return _analytics_cache


@app.get("/api/analytics")
async def api_get_analytics():
    cfg = load_config()
    return _get_cached_analytics(cfg)


@app.get("/api/status")
async def get_status():
    cfg = load_config()
    ollama = await check_ollama(cfg.ollama_url)
    hw = detect_hardware()
    recommendations = recommend_models(ollama.get("models", []), hw)
    provider_models = list_provider_models(cfg, ollama)

    vault_path = Path(cfg.vault_path) if cfg.vault_path else None
    vault_folders = (
        discover_vault_folders(vault_path)
        if vault_path and vault_path.exists()
        else []
    )
    journal_candidates = (
        discover_journal_candidates(vault_path)
        if vault_path and vault_path.exists()
        else []
    )

    with db_session(DB_PATH) as conn:
        folder_counts = [
            dict(r)
            for r in conn.execute(
                "SELECT folder, COUNT(*) AS count FROM notes GROUP BY folder ORDER BY count DESC"
            ).fetchall()
        ]
        recent_notes = [
            dict(r)
            for r in conn.execute(
                """
                SELECT path, title, folder, note_date, word_count
                FROM notes
                ORDER BY COALESCE(note_date, '') DESC, mtime DESC
                LIMIT 12
                """
            ).fetchall()
        ]
        memory_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        chat_chunk_count = conn.execute("SELECT COUNT(*) FROM chat_chunks").fetchone()[0]

    on_this_day = get_on_this_day(DB_PATH, cfg.journal_folders)
    for item in on_this_day:
        item["obsidian_uri"] = _obsidian_uri(cfg.vault_path, item["note_path"])

    return {
        "config": cfg.to_public_dict(),
        "needs_onboarding": not (cfg.vault_path and Path(cfg.vault_path).exists()),
        "candidate_vaults": _detect_candidate_vaults(),
        "ollama": ollama,
        "hardware": hw,
        "model_recommendations": recommendations,
        "provider_models": provider_models,
        "index_status": indexer.status.to_dict(),
        "deep_status": _deep_status,
        "vault_folders": vault_folders,
        "journal_candidates": journal_candidates,
        "indexed_folders": folder_counts,
        "recent_notes": recent_notes,
        "on_this_day": on_this_day,
        "memory_count": memory_count,
        "chat_chunk_count": chat_chunk_count,
        "last_learned_at": _last_learned_at,
    }


def _record_recent_vault(cfg: AppConfig, path_str: str) -> None:
    if not path_str:
        return
    norm = str(Path(path_str))
    existing = [p for p in cfg.recent_vaults if p.lower() != norm.lower()]
    cfg.recent_vaults = [norm, *existing][:6]


def _apply_vault_switch(cfg: AppConfig, old_vault: str) -> None:
    global _index_task, _deep_task, _deep_status
    if _deep_task and not _deep_task.done():
        _deep_task.cancel()
    if _index_task and not _index_task.done():
        _index_task.cancel()
    if old_vault:
        clear_vault_data(DB_PATH)
        _on_vault_index_updated()
        indexer._refresh_stats()
    _deep_status = {
        "state": "idle",
        "notes_processed": 0,
        "total_notes": 0,
        "message": "",
        "updated_at": None,
    }
    _record_recent_vault(cfg, cfg.vault_path)
    save_config(cfg)
    if cfg.vault_path and Path(cfg.vault_path).exists():
        indexer.start_watcher(asyncio.get_running_loop())
        _index_task = asyncio.create_task(
            _run_vault_pipeline(force_reembed=False, run_deep=True)
        )


@app.get("/api/vault/fs")
async def api_browse_fs(path: str = ""):
    """Browse local directories safely so users can pick an Obsidian vault right in the UI."""
    home = Path.home()
    roots: List[Dict[str, str]] = [
        {"label": "Home", "path": str(home)},
    ]
    docs = home / "Documents"
    if docs.exists():
        roots.append({"label": "Documents", "path": str(docs)})
    if os.name == "nt":
        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:\\")
            if drive.exists():
                roots.append({"label": f"{letter}:\\", "path": str(drive)})

    target = Path(path.strip()) if path.strip() else (docs if docs.exists() else home)
    if not target.exists() or not target.is_dir():
        target = home

    parent = str(target.parent) if target.parent != target else ""
    folders: List[Dict[str, Any]] = []
    try:
        entries = sorted(target.iterdir(), key=lambda p: p.name.lower())
    except PermissionError:
        entries = []

    for item in entries:
        try:
            if not item.is_dir() or item.name.startswith(".") or item.name.startswith("$"):
                continue
            is_vault = (item / ".obsidian").is_dir()
            folders.append(
                {
                    "name": item.name,
                    "path": str(item),
                    "is_vault": is_vault,
                }
            )
        except PermissionError:
            continue

    return {
        "current_path": str(target),
        "parent_path": parent,
        "is_current_vault": (target / ".obsidian").is_dir(),
        "roots": roots,
        "folders": folders,
    }


@app.post("/api/vault/switch")
async def api_switch_vault(req: SwitchVaultReq):
    """Switch the active Obsidian vault, clear old vault index, and process in background."""
    target = req.vault_path.strip()
    if not target:
        raise HTTPException(status_code=400, detail="Vault path cannot be empty")
    target_path = Path(target)
    if not target_path.exists() or not target_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Folder not found: {target}")

    cfg = load_config()
    old_vault = cfg.vault_path
    cfg.vault_path = str(target_path)
    if old_vault and Path(old_vault) != target_path:
        _record_recent_vault(cfg, old_vault)
    _apply_vault_switch(cfg, old_vault)
    return {"config": cfg.to_public_dict(), "switched": True}


@app.post("/api/settings")
async def update_settings(payload: SettingsUpdate):
    cfg = load_config()
    old_vault = cfg.vault_path
    old_excluded = list(cfg.excluded_folders)

    updates = payload.model_dump(exclude_none=True)
    if "vault_path" in updates:
        vp = str(updates["vault_path"]).strip()
        if vp and (not Path(vp).exists() or not Path(vp).is_dir()):
            raise HTTPException(status_code=400, detail=f"Vault folder not found: {vp}")
        updates["vault_path"] = vp

    secret_fields = ("openai_api_key", "anthropic_api_key", "gemini_api_key", "custom_api_key")
    for sf in secret_fields:
        if sf in updates:
            val = str(updates[sf]).strip()
            if not val or "••••" in val:
                updates.pop(sf)
            else:
                updates[sf] = val

    if updates.pop("clear_openai_key", False):
        cfg.openai_api_key = ""
    if updates.pop("clear_anthropic_key", False):
        cfg.anthropic_api_key = ""
    if updates.pop("clear_gemini_key", False):
        cfg.gemini_api_key = ""
    if updates.pop("clear_custom_key", False):
        cfg.custom_api_key = ""

    for k, v in updates.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    if cfg.vault_path:
        _record_recent_vault(cfg, cfg.vault_path)
    set_ram_cache_enabled(cfg.enable_ram_cache)
    save_config(cfg)

    global _index_task
    vault_switched = bool(cfg.vault_path and cfg.vault_path != old_vault)
    folders_changed = sorted(cfg.excluded_folders) != sorted(old_excluded)

    if vault_switched:
        if old_vault:
            _record_recent_vault(cfg, old_vault)
        _apply_vault_switch(cfg, old_vault)
    elif folders_changed and cfg.vault_path and Path(cfg.vault_path).exists():
        indexer.start_watcher(asyncio.get_running_loop())
        _index_task = asyncio.create_task(
            _run_vault_pipeline(force_reembed=False, run_deep=False)
        )

    return {"config": cfg.to_public_dict()}


@app.post("/api/index")
async def trigger_index(force_reembed: bool = False, include_deep: bool = False):
    global _index_task
    if _index_task and not _index_task.done():
        return {"started": False, "index_status": indexer.status.to_dict()}

    _index_task = asyncio.create_task(
        _run_vault_pipeline(force_reembed=force_reembed, run_deep=include_deep)
    )
    return {"started": True, "index_status": indexer.status.to_dict()}


@app.post("/api/ollama/pull")
async def pull_any_ollama_model(req: PullModelReq):
    cfg = load_config()
    model = req.model.strip()
    if not model:
        raise HTTPException(status_code=400, detail="Model name is required")

    async def stream():
        try:
            async for item in pull_model_stream(cfg.ollama_url, model):
                yield _sse_event("progress", item)
            if "embed" in model.lower() or model == cfg.embedding_model:
                yield _sse_event("embedding", {"status": "Generating note embeddings..."})
                status = await indexer.embed_missing_only()
                yield _sse_event("done", status.to_dict())
            else:
                yield _sse_event("done", {"model": model, "status": "ready"})
        except Exception as exc:
            yield _sse_event("error", {"error": str(exc)})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/ollama/pull-embed")
async def pull_embedding_model():
    cfg = load_config()
    return await pull_any_ollama_model(PullModelReq(model=cfg.embedding_model))


@app.get("/api/conversations")
async def api_list_conversations():
    return {"conversations": list_conversations(DB_PATH)}


@app.post("/api/conversations")
async def api_create_conversation(req: CreateConversationReq):
    cfg = load_config()
    conv_id = str(uuid.uuid4())[:12]
    now = utc_now_iso()
    title = (req.title or "New Conversation").strip()
    model = req.model or f"{cfg.provider}::{cfg.chat_model}"
    with db_session(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO conversations (id, title, mode, model, folder_scope, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (conv_id, title, req.mode, model, req.folder_scope, now, now),
        )
    return {
        "conversation": {
            "id": conv_id,
            "title": title,
            "mode": req.mode,
            "model": model,
            "folder_scope": req.folder_scope,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }
    }


@app.get("/api/conversations/{conv_id}")
async def api_get_conversation(conv_id: str):
    with db_session(DB_PATH) as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Conversation not found")
    msgs = get_conversation_messages(conv_id, DB_PATH)
    return {"conversation": dict(row), "messages": msgs}


@app.delete("/api/conversations/{conv_id}")
async def api_delete_conversation(conv_id: str):
    with db_session(DB_PATH) as conn:
        conn.execute(
            "DELETE FROM chat_chunks_fts WHERE conversation_id = ?", (conv_id,)
        )
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    retriever.reload_vectors()
    return {"deleted": True}


@app.post("/api/conversations/{conv_id}/chat")
async def api_chat(conv_id: str, req: ChatReq):
    cfg = load_config()
    user_text = req.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Empty message")

    with db_session(DB_PATH) as conn:
        conv = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conv_dict = dict(conv)

    mode = req.mode or conv_dict["mode"] or cfg.default_mode
    model = req.model or conv_dict["model"] or f"{cfg.provider}::{cfg.chat_model}"
    folder_scope = req.folder_scope or conv_dict["folder_scope"] or "all"

    now = utc_now_iso()
    with db_session(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            VALUES (?, 'user', ?, ?)
            """,
            (conv_id, user_text, now),
        )
        msg_count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conv_id,)
        ).fetchone()[0]
        new_title = conv_dict["title"]
        if msg_count == 1 or conv_dict["title"] == "New Conversation":
            new_title = user_text[:48] + ("..." if len(user_text) > 48 else "")
        conn.execute(
            """
            UPDATE conversations
            SET title = ?, mode = ?, model = ?, folder_scope = ?, updated_at = ?
            WHERE id = ?
            """,
            (new_title, mode, model, folder_scope, now, conv_id),
        )

    history = get_conversation_messages(conv_id, DB_PATH)
    temporal = parse_temporal_filter(user_text)
    retrieved, past_chats = await asyncio.gather(
        retriever.retrieve(user_text, cfg, folder_scope=folder_scope),
        retriever.retrieve_past_chats(user_text, cfg, exclude_conv_id=conv_id, top_k=3),
    )
    vault_intel = _get_cached_analytics(cfg)
    system_prompt = build_system_prompt(
        cfg,
        mode,
        retrieved,
        past_chats=past_chats,
        temporal_label=temporal["label"] if temporal else None,
        vault_intel=vault_intel,
    )

    ollama_messages = [{"role": "system", "content": system_prompt}]
    for m in history[-10:]:
        ollama_messages.append({"role": m["role"], "content": m["content"]})

    sources_payload = [
        {
            "source_type": "note",
            "note_path": ch.note_path,
            "title": ch.title,
            "folder": ch.folder,
            "note_date": ch.note_date,
            "heading": ch.heading,
            "excerpt": ch.content[:320],
            "match_reason": ch.match_reason,
            "obsidian_uri": _obsidian_uri(cfg.vault_path, ch.note_path),
        }
        for ch in retrieved
    ] + [
        {
            "source_type": "chat",
            "conversation_id": pc.conversation_id,
            "title": pc.conv_title,
            "folder": "Past Chat",
            "note_date": pc.created_at[:10] if pc.created_at else "",
            "excerpt": pc.content[:320],
            "match_reason": pc.match_reason,
        }
        for pc in past_chats
    ]

    async def event_generator():
        yield _sse_event("sources", {"sources": sources_payload, "title": new_title})

        reply_parts: List[str] = []
        think_parts: List[str] = []
        try:
            async for ev_type, text in provider_chat_stream(
                cfg, model, ollama_messages, temperature=cfg.temperature
            ):
                if ev_type == "thinking":
                    think_parts.append(text)
                    yield _sse_event("thinking", {"text": text})
                else:
                    reply_parts.append(text)
                    yield _sse_event("token", {"text": text})

            full_reply = "".join(reply_parts).strip()
            full_think = "".join(think_parts).strip()
            done_time = utc_now_iso()

            with db_session(DB_PATH) as conn:
                cur = conn.execute(
                    """
                    INSERT INTO messages (
                        conversation_id, role, content, thinking, sources_json, created_at
                    )
                    VALUES (?, 'assistant', ?, ?, ?, ?)
                    """,
                    (
                        conv_id,
                        full_reply,
                        full_think,
                        json.dumps(sources_payload, ensure_ascii=False),
                        done_time,
                    ),
                )
                assistant_msg_id = int(cur.lastrowid)
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (done_time, conv_id),
                )

            _track_task(
                _background_learn_after_turn(
                    conv_id=conv_id,
                    assistant_msg_id=assistant_msg_id,
                    conv_title=new_title,
                    user_text=user_text,
                    assistant_text=full_reply,
                    created_at=done_time,
                    model=model,
                )
            )

            yield _sse_event("done", {"title": new_title})
        except Exception as exc:
            yield _sse_event(
                "error",
                {
                    "error": (
                        f"Could not get response ({model}): {exc}. "
                        "Make sure your local Ollama server is running or check your API settings."
                    )
                },
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/conversations/{conv_id}/distill")
async def api_distill(conv_id: str):
    cfg = load_config()
    saved = await distill_memories_from_conversation(
        conv_id, cfg, model=cfg.memory_model or None
    )
    return {"saved": saved, "memories": get_memories(DB_PATH)}


@app.get("/api/memories")
async def api_list_memories():
    return {"memories": get_memories(DB_PATH)}


@app.post("/api/memories")
async def api_create_memory(req: MemoryCreateReq):
    content = req.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Memory content cannot be empty")
    now = utc_now_iso()
    with db_session(DB_PATH) as conn:
        cur = conn.execute(
            """
            INSERT INTO memories (category, content, source_info, pinned, created_at)
            VALUES (?, ?, 'manual', ?, ?)
            """,
            (req.category.strip().lower(), content, 1 if req.pinned else 0, now),
        )
        mem_id = cur.lastrowid
    return {
        "memory": {
            "id": mem_id,
            "category": req.category,
            "content": content,
            "source_info": "manual",
            "pinned": 1 if req.pinned else 0,
            "created_at": now,
        }
    }


@app.patch("/api/memories/{mem_id}")
async def api_update_memory(mem_id: int, req: MemoryUpdateReq):
    with db_session(DB_PATH) as conn:
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (mem_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Memory not found")
        category = req.category if req.category is not None else row["category"]
        content = req.content.strip() if req.content is not None else row["content"]
        pinned = (1 if req.pinned else 0) if req.pinned is not None else row["pinned"]
        conn.execute(
            "UPDATE memories SET category = ?, content = ?, pinned = ? WHERE id = ?",
            (category, content, pinned, mem_id),
        )
    return {"memories": get_memories(DB_PATH)}


@app.delete("/api/memories/{mem_id}")
async def api_delete_memory(mem_id: int):
    with db_session(DB_PATH) as conn:
        conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
    return {"deleted": True}


@app.get("/api/self-profile")
async def api_get_self_profile():
    with db_session(DB_PATH) as conn:
        digested = conn.execute(
            """
            SELECT COUNT(*) FROM note_digests d
            JOIN notes n ON n.path = d.note_path AND n.content_hash = d.content_hash
            """
        ).fetchone()[0]
        total_notes = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    return {
        "profile": get_self_profile(DB_PATH),
        "digested_notes": digested,
        "total_notes": total_notes,
    }


@app.post("/api/self-profile/synthesize")
async def api_synthesize_self_profile():
    cfg = load_config()
    try:
        profile = await synthesize_self_profile(cfg)
        return {"profile": profile}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {exc}")


@app.post("/api/self-profile/deep-start")
async def api_deep_start():
    started = _start_deep_bg()
    return {"started": started, "deep_status": _deep_status}


@app.post("/api/self-profile/deep-stop")
async def api_deep_stop():
    global _deep_task
    if _deep_task and not _deep_task.done():
        _deep_task.cancel()
        return {"stopped": True, "deep_status": _deep_status}
    return {"stopped": False, "deep_status": _deep_status}


@app.post("/api/self-profile/deep-synthesize")
async def api_deep_synthesize_self_profile():
    cfg = load_config()

    async def stream():
        try:
            async for item in deep_synthesize_self_profile_stream(cfg, db_path=DB_PATH):
                stage = item.get("stage", "reading")
                if stage == "done":
                    yield _sse_event("done", item)
                elif stage == "error":
                    yield _sse_event("error", item)
                else:
                    yield _sse_event("progress", item)
        except Exception as exc:
            yield _sse_event("error", {"error": str(exc)})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.put("/api/self-profile")
async def api_update_self_profile(req: SelfProfileUpdateReq):
    now = utc_now_iso()
    with db_session(DB_PATH) as conn:
        note_count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.execute(
            """
            INSERT INTO self_profile (
                id, summary_md, updated_at, note_count_snapshot, memory_count_snapshot
            )
            VALUES (1, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                summary_md = excluded.summary_md,
                updated_at = excluded.updated_at,
                note_count_snapshot = excluded.note_count_snapshot,
                memory_count_snapshot = excluded.memory_count_snapshot
            """,
            (req.summary_md, now, note_count, mem_count),
        )
    return {"profile": {"summary_md": req.summary_md, "updated_at": now}}


@app.post("/api/vault/export-reflection")
async def api_export_reflection(req: ExportReflectionReq):
    cfg = load_config()
    if not cfg.vault_path or not Path(cfg.vault_path).exists():
        raise HTTPException(status_code=400, detail="Obsidian vault path is not configured.")
    rel_path = save_reflection_to_vault(
        cfg, req.title, req.content, subfolder=req.subfolder
    )
    return {
        "saved_path": rel_path,
        "obsidian_uri": _obsidian_uri(cfg.vault_path, rel_path),
    }
