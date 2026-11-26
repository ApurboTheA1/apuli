"""Personal companion prompts, auto-distilled memory, and note-by-note whole-vault self-portrait."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from .analytics import compute_vault_analytics
from .config import AppConfig, DB_PATH, is_in_folders
from .db import (
    db_session,
    get_conversation_messages,
    get_memories,
    get_self_profile,
    utc_now_iso,
)
from .hardware import compute_synthesis_budget, detect_hardware
from .ollama_client import check_ollama
from .providers import parse_model_spec, provider_chat_once
from .retrieval import RetrievedChatChunk, RetrievedChunk

_profile_lock = asyncio.Lock()
_distill_locks: Dict[str, asyncio.Lock] = {}

MODE_INSTRUCTIONS = {
    "companion": (
        "Mode: Personal Companion.\n"
        "- Speak like a thoughtful, perceptive, grounded friend who has read the user's journals and remembers past conversations.\n"
        "- Help them feel understood, notice emotional or intellectual threads, and connect what they're feeling today with who they have been across their notes.\n"
        "- Be warm and natural, never overly clinical or sycophantic. Ask gentle, clarifying questions when helpful."
    ),
    "recall": (
        "Mode: Memory Recall.\n"
        "- Focus on accurately remembering and surfacing what the user wrote in their Obsidian vault or told you in earlier conversations.\n"
        "- Anchor your answer in dates, note titles, people, and exact details from the retrieved context.\n"
        "- If something isn't in the retrieved context, say clearly that you don't see it rather than guessing."
    ),
    "mirror": (
        "Mode: Pattern & Growth Mirror.\n"
        "- Look across the user's retrieved notes, past conversations, and long-term memories for recurring patterns, tensions, values, blind spots, and personal growth over time.\n"
        "- Gently reflect back how their thinking or feelings evolved between earlier notes and newer ones, citing specific notes."
    ),
    "creative": (
        "Mode: Podcast & Idea Partner.\n"
        "- Help brainstorm, structure thoughts, and connect ideas across their Media/podcast notes, Inbox sparks, College concepts, and Personal reflections.\n"
        "- Keep their authentic voice and past curiosities front and center."
    ),
}


def build_system_prompt(
    cfg: AppConfig,
    mode: str,
    retrieved: List[RetrievedChunk],
    past_chats: Optional[List[RetrievedChatChunk]] = None,
    temporal_label: Optional[str] = None,
    vault_intel: Optional[Dict[str, Any]] = None,
) -> str:
    today_str = datetime.now().strftime("%A, %B %d, %Y")
    mode_block = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["companion"])

    profile = get_self_profile()
    memories = get_memories()

    parts = [
        f"You are {cfg.companion_name}, a completely local personal companion built for {cfg.user_name}.",
        f"Today's date is {today_str}.",
        "Your purpose is to talk with the user, know them deeply through their Obsidian vault and past conversations, and help them remember themselves.",
        mode_block,
        "Grounding rules:\n"
        "- Only claim memories, events, or quotes from the notes, past conversations, or memory entries provided below.\n"
        "- When referencing a note, cite it naturally using its title or path (e.g., [[Note Title]] or `Folder/Note.md`, with its date if relevant).\n"
        "- When recalling an earlier conversation, mention what you talked about naturally without overstating.\n"
        "- Never fabricate details that aren't in the context. If memory is incomplete, say so honestly.",
    ]

    if temporal_label:
        parts.append(
            f"Temporal focus detected in user's question: {temporal_label}. Prioritize notes and context from this timeframe."
        )

    if vault_intel:
        intel_lines = []
        rels = vault_intel.get("relationships") or []
        if rels:
            top_p = ", ".join(
                f"{r['name']} ({r['mentions']} mentions, {r.get('first_seen') or '?'}→{r.get('last_seen') or '?'})"
                for r in rels[:8]
            )
            intel_lines.append(f"- Key People in Vault: {top_p}")
        clusters = vault_intel.get("clusters") or []
        if clusters:
            top_c = "; ".join(
                f"{c['label']} ({c['note_count']} notes)" for c in clusters[:5]
            )
            intel_lines.append(f"- Discovered Life Themes: {top_c}")
        if intel_lines:
            parts.append("=== VAULT GRAPH & PATTERN SIGNALS ===\n" + "\n".join(intel_lines))

    if profile and profile.get("summary_md"):
        parts.append(
            "=== SELF-PORTRAIT (Evolving locally from vault & conversations) ===\n"
            + profile["summary_md"].strip()
        )

    if memories:
        mem_lines = []
        for m in memories[:20]:
            pin = "[Pinned] " if m["pinned"] else ""
            mem_lines.append(f"- {pin}({m['category']}) {m['content']}")
        parts.append("=== LONG-TERM MEMORIES ===\n" + "\n".join(mem_lines))

    if past_chats:
        chat_blocks = []
        for idx, pc in enumerate(past_chats, start=1):
            date_part = pc.created_at[:10] if pc.created_at else "earlier"
            chat_blocks.append(
                f"[Past Chat {idx}] \"{pc.conv_title}\" ({date_part})\n{pc.content}"
            )
        parts.append(
            "=== RELEVANT PAST CONVERSATIONS ===\n" + "\n\n---\n\n".join(chat_blocks)
        )

    if retrieved:
        ctx_blocks = []
        for idx, ch in enumerate(retrieved, start=1):
            date_label = f" | Date: {ch.note_date}" if ch.note_date else ""
            tag_label = f" | Tags: {', '.join(ch.tags[:5])}" if ch.tags else ""
            ctx_blocks.append(
                f"[Note {idx}] {ch.note_path} (Section: {ch.heading}{date_label}{tag_label})\n"
                f"{ch.content}"
            )
        parts.append(
            "=== RETRIEVED OBSIDIAN NOTES ===\n" + "\n\n---\n\n".join(ctx_blocks)
        )
    else:
        parts.append(
            "=== RETRIEVED OBSIDIAN NOTES ===\n(No matching notes were retrieved for this turn.)"
        )

    return "\n\n".join(parts)


async def distill_memories_from_conversation(
    conversation_id: str,
    cfg: AppConfig,
    model: Optional[str] = None,
    only_after_msg_id: int = 0,
    db_path: Path = DB_PATH,
) -> List[dict]:
    """Extract enduring personal facts, values, relationships, or patterns from a chat."""
    lock = _distill_locks.setdefault(conversation_id, asyncio.Lock())
    async with lock:
        msgs = get_conversation_messages(conversation_id, db_path)
        new_msgs = [m for m in msgs if m["id"] > only_after_msg_id]
        if len(new_msgs) < 2:
            return []

        transcript_lines = []
        for m in new_msgs[-16:]:
            speaker = cfg.user_name if m["role"] == "user" else cfg.companion_name
            transcript_lines.append(f"{speaker}: {m['content'][:1200]}")

        existing = get_memories(db_path)
        existing_text = "\n".join(f"- {m['content']}" for m in existing[:30]) or "(none yet)"

        prompt = (
            f"You are maintaining a private local memory bank for {cfg.user_name}.\n"
            "Extract ONLY enduring, meaningful personal facts, values, life milestones, key relationships, "
            "recurring patterns, or ongoing creative/life threads revealed in the conversation below.\n"
            "Skip trivial chatter or things already captured in Existing Memories.\n"
            "Return a valid JSON array of objects, each with keys:\n"
            '  "category": one of ["value", "milestone", "pattern", "relationship", "thread", "preference"]\n'
            f'  "content": concise 1-sentence memory written in third person (e.g., "{cfg.user_name} ...")\n'
            "If there is nothing worth saving, return []. Output ONLY JSON.\n\n"
            f"Existing Memories:\n{existing_text}\n\n"
            "Conversation:\n" + "\n".join(transcript_lines)
        )

        used_model = model or cfg.memory_model or cfg.chat_model
        raw = await provider_chat_once(
            cfg,
            used_model,
            [{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        max_msg_id = max(m["id"] for m in new_msgs)

        match = re.search(r"\[.*\]", raw, re.DOTALL)
        items = []
        if match:
            try:
                loaded = json.loads(match.group(0))
                if isinstance(loaded, list):
                    items = loaded
            except Exception:
                items = []

        existing_lower = {m["content"].strip().lower() for m in existing}
        saved: List[dict] = []
        with db_session(db_path) as conn:
            for item in items[:5]:
                if not isinstance(item, dict):
                    continue
                content = str(item.get("content", "")).strip()
                category = str(item.get("category", "thread")).strip().lower()
                if not content or content.lower() in existing_lower:
                    continue
                if category not in (
                    "value", "milestone", "pattern", "relationship", "thread", "preference"
                ):
                    category = "thread"
                now = utc_now_iso()
                cur = conn.execute(
                    """
                    INSERT INTO memories (category, content, source_info, pinned, created_at)
                    VALUES (?, ?, ?, 0, ?)
                    """,
                    (category, content, f"chat:{conversation_id}", now),
                )
                saved.append(
                    {
                        "id": cur.lastrowid,
                        "category": category,
                        "content": content,
                        "source_info": f"chat:{conversation_id}",
                        "pinned": 0,
                        "created_at": now,
                    }
                )
                existing_lower.add(content.lower())

            conn.execute(
                "UPDATE conversations SET last_distilled_msg_id = ? WHERE id = ?",
                (max_msg_id, conversation_id),
            )
        return saved


async def maybe_auto_distill(
    conversation_id: str,
    cfg: AppConfig,
    model: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> List[dict]:
    """Automatically distill memories every N user turns when auto_learn is enabled."""
    if not cfg.auto_learn:
        return []
    with db_session(db_path) as conn:
        conv = conn.execute(
            "SELECT last_distilled_msg_id FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if not conv:
            return []
        last_id = int(conv["last_distilled_msg_id"] or 0)
        new_user_turns = conn.execute(
            """
            SELECT COUNT(*) FROM messages
            WHERE conversation_id = ? AND id > ? AND role = 'user'
            """,
            (conversation_id, last_id),
        ).fetchone()[0]

    threshold = max(1, int(cfg.auto_distill_turns or 2))
    if new_user_turns < threshold:
        return []

    return await distill_memories_from_conversation(
        conversation_id, cfg, model=model, only_after_msg_id=last_id, db_path=db_path
    )


async def synthesize_self_profile(
    cfg: AppConfig,
    model: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> Dict[str, object]:
    """Quickly synthesize or evolve the 'Who I Am' self-portrait from recent notes & memories."""
    async with _profile_lock:
        existing_profile = get_self_profile(db_path)
        with db_session(db_path) as conn:
            note_count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            all_rows = conn.execute(
                """
                SELECT n.path, n.title, n.folder, n.note_date, c.content
                FROM notes n
                JOIN chunks c ON c.note_path = n.path AND c.chunk_index = 0
                ORDER BY COALESCE(n.note_date, '') DESC, n.mtime DESC
                LIMIT 120
                """
            ).fetchall()

        journal_rows = [r for r in all_rows if is_in_folders(r["path"], cfg.journal_folders)][:16]
        seen_paths = {r["path"] for r in journal_rows}
        other_rows = [
            r for r in all_rows
            if r["path"] not in seen_paths and r["folder"] in ("Personal", "People", "Media", "Inbox")
        ][:12]
        if not journal_rows and not other_rows:
            other_rows = list(all_rows[:20])
        rows = journal_rows + other_rows

        memories = get_memories(db_path)
        mem_count = len(memories)

        samples = []
        for r in rows:
            date_s = f" ({r['note_date']})" if r["note_date"] else ""
            samples.append(f"- {r['path']}{date_s}:\n  {r['content'][:450].strip()}")

        mem_block = "\n".join(
            f"- ({m['category']}) {m['content']}" for m in memories[:20]
        )

        prev_block = ""
        if existing_profile and existing_profile.get("summary_md"):
            prev_block = (
                "Previous Self-Portrait (evolve and refine this rather than discarding enduring truths):\n"
                + existing_profile["summary_md"].strip()
                + "\n\n"
            )

        prompt = (
            f"Based ONLY on these local Obsidian note excerpts, saved memories, and prior portrait for {cfg.user_name}, "
            "write or evolve a warm, grounded, concise Self-Portrait in Markdown (~250-350 words) to help them remember themselves.\n"
            "Include sections:\n"
            "### Core Identity & Values\n"
            "### Active Life Chapters & Projects\n"
            "### Recurring Themes & Inner Patterns\n"
            "### People & Anchors\n\n"
            "Be honest and grounded strictly in the provided text; if something isn't clear yet, leave room for discovery.\n\n"
            f"{prev_block}"
            f"Saved Memories:\n{mem_block or '(none yet)'}\n\n"
            "Recent Note Samples:\n" + ("\n".join(samples) if samples else "(no notes indexed yet)")
        )

        used_model = model or cfg.memory_model or cfg.chat_model
        summary = await provider_chat_once(
            cfg,
            used_model,
            [{"role": "user", "content": prompt}],
            temperature=0.35,
        )
        now = utc_now_iso()
        was_deep = int(existing_profile.get("is_deep") or 0) if existing_profile else 0
        prev_covered = int(existing_profile.get("notes_covered") or 0) if existing_profile else 0
        notes_covered = max(prev_covered, len(rows)) if was_deep else len(rows)

        with db_session(db_path) as conn:
            conn.execute(
                """
                INSERT INTO self_profile (
                    id, summary_md, updated_at, note_count_snapshot,
                    memory_count_snapshot, notes_covered, is_deep
                )
                VALUES (1, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    summary_md = excluded.summary_md,
                    updated_at = excluded.updated_at,
                    note_count_snapshot = excluded.note_count_snapshot,
                    memory_count_snapshot = excluded.memory_count_snapshot,
                    notes_covered = excluded.notes_covered,
                    is_deep = excluded.is_deep
                """,
                (summary, now, note_count, mem_count, notes_covered, was_deep),
            )
        return {
            "summary_md": summary,
            "updated_at": now,
            "note_count_snapshot": note_count,
            "memory_count_snapshot": mem_count,
            "notes_covered": notes_covered,
            "is_deep": was_deep,
        }


def _load_note_segments(
    note_path: str, db_path: Path, max_segment_chars: int = 5000
) -> List[str]:
    """Load the ENTIRE note across all of its chunks, split only if longer than max_segment_chars."""
    with db_session(db_path) as conn:
        rows = conn.execute(
            """
            SELECT heading, content
            FROM chunks
            WHERE note_path = ?
            ORDER BY chunk_index ASC
            """,
            (note_path,),
        ).fetchall()

    if not rows:
        return []

    segments: List[str] = []
    buf: List[str] = []
    cur_len = 0
    for r in rows:
        piece = f"[{r['heading']}]\n{r['content']}".strip() if r["heading"] else r["content"].strip()
        if not piece:
            continue
        if buf and cur_len + len(piece) > max_segment_chars:
            segments.append("\n\n".join(buf))
            buf = []
            cur_len = 0
        buf.append(piece)
        cur_len += len(piece) + 2
    if buf:
        segments.append("\n\n".join(buf))
    return segments


def _save_note_digest(
    note: Dict[str, str],
    digest: str,
    is_personal: int,
    used_model: str,
    db_path: Path,
) -> None:
    now = utc_now_iso()
    with db_session(db_path) as conn:
        conn.execute(
            """
            INSERT INTO note_digests (
                note_path, content_hash, note_date, folder, digest, is_personal, model_used, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(note_path) DO UPDATE SET
                content_hash = excluded.content_hash,
                note_date = excluded.note_date,
                folder = excluded.folder,
                digest = excluded.digest,
                is_personal = excluded.is_personal,
                model_used = excluded.model_used,
                updated_at = excluded.updated_at
            """,
            (
                note["path"],
                note["content_hash"],
                note["note_date"],
                note["folder"],
                digest,
                is_personal,
                used_model,
                now,
            ),
        )


async def _call_chat_once(
    cfg: AppConfig,
    used_model: str,
    prompt: str,
    temperature: float = 0.25,
    num_ctx: int = 8192,
) -> str:
    sig = inspect.signature(provider_chat_once)
    kwargs: Dict[str, Any] = {"temperature": temperature}
    if "num_ctx" in sig.parameters:
        kwargs["num_ctx"] = num_ctx
    if "keep_alive" in sig.parameters:
        kwargs["keep_alive"] = cfg.ollama_keep_alive
    if "num_thread" in sig.parameters:
        kwargs["num_thread"] = cfg.ollama_num_thread
    return await provider_chat_once(
        cfg, used_model, [{"role": "user", "content": prompt}], **kwargs
    )


def _get_cached_chapter(
    db_path: Path, chapter_key: str, source_hash: str
) -> Optional[str]:
    with db_session(db_path) as conn:
        row = conn.execute(
            "SELECT summary, source_hash FROM synthesis_chapters WHERE chapter_key = ?",
            (chapter_key,),
        ).fetchone()
    if row and row["source_hash"] == source_hash and row["summary"]:
        return str(row["summary"])
    return None


def _save_cached_chapter(
    db_path: Path,
    chapter_key: str,
    scope: str,
    source_hash: str,
    model_used: str,
    summary: str,
) -> None:
    with db_session(db_path) as conn:
        conn.execute(
            """
            INSERT INTO synthesis_chapters (chapter_key, scope, source_hash, model_used, summary, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(chapter_key) DO UPDATE SET
                scope = excluded.scope,
                source_hash = excluded.source_hash,
                model_used = excluded.model_used,
                summary = excluded.summary,
                updated_at = excluded.updated_at
            """,
            (chapter_key, scope, source_hash, model_used, summary, utc_now_iso()),
        )


async def _resolve_final_synthesis_model(cfg: AppConfig, reader_model: str) -> str:
    """Use an explicit stronger model if configured, or auto-upgrade final Self-Portrait to 14B when installed."""
    explicit = (getattr(cfg, "final_synthesis_model", "") or "").strip()
    if explicit:
        return explicit
    if cfg.provider != "ollama" or cfg.memory_model or "::" in reader_model:
        return reader_model
    try:
        ollama = await check_ollama(cfg.ollama_url)
    except Exception:
        return reader_model
    if not ollama.get("online"):
        return reader_model
    chat_models: List[str] = list(ollama.get("chat_models") or [])
    if "qwen2.5:14b" in chat_models:
        return "qwen2.5:14b"
    # Otherwise pick any installed non-reasoning 13b–32b model for the single final synthesis pass
    for m in chat_models:
        ml = m.lower()
        if any(tag in ml for tag in ("14b", "13b", "22b", "27b", "32b")) and "r1" not in ml:
            return m
    return reader_model


async def _digest_single_note(
    note: Dict[str, str],
    cfg: AppConfig,
    used_model: str,
    db_path: Path,
    is_journal: bool = False,
    num_ctx: int = 8192,
) -> Tuple[str, int]:
    """Read an entire single note (across all its segments) and store its cached digest."""
    segments = _load_note_segments(note["path"], db_path)
    full_len = sum(len(s) for s in segments)
    if full_len < 25:
        _save_note_digest(note, "REFERENCE_ONLY", 0, used_model, db_path)
        return "REFERENCE_ONLY", 0

    part_summaries: List[str] = []
    for idx, seg in enumerate(segments, start=1):
        part_label = f" (Part {idx}/{len(segments)})" if len(segments) > 1 else ""
        if is_journal:
            prompt = (
                f"Read this personal journal / periodic reflection entry by {cfg.user_name}:\n"
                f"Path: {note['path']} | Date: {note['note_date']}{part_label}\n\n"
                f"{seg}\n\n"
                "Task:\n"
                "- Because this is a personal journal/daily/periodic note, pay close attention to emotional tone, "
                "inner voice, values, turning points, people mentioned, struggles, and recurring patterns.\n"
                "- Write 3–6 concise, specific bullet points capturing what this entry reveals about "
                f"{cfg.user_name}. Stay strictly grounded in the text."
            )
        else:
            prompt = (
                f"Read this Obsidian note written by {cfg.user_name}:\n"
                f"Path: {note['path']} | Date: {note['note_date']} | Folder: {note['folder']}{part_label}\n\n"
                f"{seg}\n\n"
                "Task:\n"
                "- If this note contains anything personal (thoughts, feelings, self-reflection, values, struggles, "
                "relationships, life events, decisions, or creative/podcast ideas), write 2–5 concise, specific bullet points "
                f"capturing what it reveals about {cfg.user_name}.\n"
                "- If it is purely generic reference material (raw lecture notes, textbook definitions, code, or bare links) "
                "with zero personal insight, reply with exactly: REFERENCE_ONLY"
            )
        resp = await _call_chat_once(cfg, used_model, prompt, temperature=0.2, num_ctx=num_ctx)
        cleaned = resp.strip()
        if cleaned and (is_journal or "REFERENCE_ONLY" not in cleaned.upper()):
            part_summaries.append(cleaned)

    if not part_summaries:
        digest, is_personal = "REFERENCE_ONLY", 0
    elif len(part_summaries) == 1:
        digest, is_personal = part_summaries[0], 1
    else:
        merge_prompt = (
            f"Combine these section summaries of a single note ({note['path']}, {note['note_date']}) "
            f"into 3–6 concise bullet points capturing what it reveals about {cfg.user_name}:\n\n"
            + "\n\n".join(part_summaries)
        )
        digest = await _call_chat_once(cfg, used_model, merge_prompt, temperature=0.2, num_ctx=num_ctx)
        is_personal = 1

    _save_note_digest(note, digest, is_personal, used_model, db_path)
    return digest, is_personal


async def _digest_note_batch(
    batch: List[Tuple[Dict[str, str], bool, str]],
    cfg: AppConfig,
    used_model: str,
    num_ctx: int,
    db_path: Path,
) -> None:
    """Process a batch of notes in a single model call and persist each note's digest."""
    if len(batch) == 1:
        note, is_j, _ = batch[0]
        await _digest_single_note(note, cfg, used_model, db_path, is_journal=is_j, num_ctx=num_ctx)
        return

    entries = []
    for idx, (note, is_j, text) in enumerate(batch, start=1):
        kind = "JOURNAL" if is_j else note["folder"]
        entries.append(f"[{idx}] ({kind} · {note['note_date']}) {note['path']}\n{text}")

    prompt = (
        f"Analyze these {len(batch)} chronological Obsidian notes by {cfg.user_name}.\n"
        "For EACH numbered note [1]..[{n}], write 1–3 concise, specific sentences capturing what it reveals "
        f"about {cfg.user_name} (emotional tone, values, struggles, relationships, decisions, or creative projects).\n"
        "Rules:\n"
        "- For any note marked JOURNAL, NEVER skip it — capture its personal meaning.\n"
        "- For non-journal notes that are purely generic reference/textbook/code with zero personal insight, set their value to \"REFERENCE_ONLY\".\n"
        "- Return ONLY a valid JSON object mapping each string index \"1\"..\"{n}\" to its summary string.\n\n"
        + "\n\n---\n\n".join(entries)
    ).replace("{n}", str(len(batch)))

    raw = await _call_chat_once(cfg, used_model, prompt, temperature=0.2, num_ctx=num_ctx)
    parsed: Dict[str, Any] = {}
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            loaded = json.loads(m.group(0))
            if isinstance(loaded, dict):
                parsed = loaded
        except Exception:
            parsed = {}

    if parsed:
        for idx, (note, is_j, _) in enumerate(batch, start=1):
            val = str(parsed.get(str(idx), "")).strip()
            if not val or (not is_j and "REFERENCE_ONLY" in val.upper()):
                _save_note_digest(note, "REFERENCE_ONLY", 0, used_model, db_path)
            else:
                _save_note_digest(note, val, 1, used_model, db_path)
    else:
        # Fallback if a small model answered in plain prose instead of JSON
        cleaned = raw.strip()
        for idx, (note, is_j, _) in enumerate(batch):
            if idx == 0 and cleaned:
                _save_note_digest(note, cleaned, 1, used_model, db_path)
            elif is_j and cleaned:
                _save_note_digest(note, f"(Grouped reflection) {cleaned[:350]}", 1, used_model, db_path)
            else:
                _save_note_digest(note, "REFERENCE_ONLY", 0, used_model, db_path)


async def deep_synthesize_self_profile_stream(
    cfg: AppConfig,
    model: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Walk through EVERY indexed note in the vault chronologically in adaptive batches
    sized to the user's hardware and AI model capabilities, caching per-note digests in
    SQLite so progress is resumable and future updates only read new/edited notes.
    """
    async with _profile_lock:
        t_start = time.monotonic()
        used_model = model or cfg.memory_model or cfg.chat_model
        prov, mod_name = parse_model_spec(used_model, cfg.provider or "ollama")
        hw = detect_hardware()
        budget = compute_synthesis_budget(prov, mod_name, hw=hw)
        concurrency = 2 if (
            prov in ("openai", "anthropic", "gemini") or float(hw.get("vram_gb") or 0) >= 7.5
        ) else 1

        with db_session(db_path) as conn:
            note_rows = conn.execute(
                """
                SELECT path, title, folder, note_date, content_hash
                FROM notes
                ORDER BY COALESCE(note_date, '9999-99-99') ASC, mtime ASC
                """
            ).fetchall()
            cached_map = {
                r["note_path"]: dict(r)
                for r in conn.execute("SELECT * FROM note_digests").fetchall()
            }

        notes_list = [
            {
                "path": r["path"],
                "title": r["title"],
                "folder": r["folder"],
                "note_date": r["note_date"] or "undated",
                "content_hash": r["content_hash"],
            }
            for r in note_rows
        ]
        total_notes = len(notes_list)

        if total_notes == 0:
            yield {"stage": "error", "error": "No indexed notes found in the vault yet."}
            return

        to_digest: List[Tuple[Dict[str, str], bool]] = []
        reused = 0
        for n in notes_list:
            is_j = is_in_folders(n["path"], cfg.journal_folders)
            cached = cached_map.get(n["path"])
            if cached and cached["content_hash"] == n["content_hash"] and not (is_j and cached["is_personal"] == 0):
                reused += 1
            else:
                to_digest.append((n, is_j))

        if reused > 0:
            yield {
                "stage": "reading",
                "notes_processed": reused,
                "total_notes": total_notes,
                "cached_notes": reused,
                "message": f"Resuming — {reused}/{total_notes} cached; processing {len(to_digest)} remaining ({budget['label']})…",
            }

        # Pack remaining notes into adaptive batches matched to hardware + model
        batches: List[List[Tuple[Dict[str, str], bool, str]]] = []
        cur_batch: List[Tuple[Dict[str, str], bool, str]] = []
        cur_chars = 0
        per_note_cap = int(budget["per_note_cap"])
        batch_max_chars = int(budget["batch_max_chars"])
        batch_max_notes = int(budget["batch_max_notes"])

        for n, is_j in to_digest:
            segs = _load_note_segments(n["path"], db_path, max_segment_chars=per_note_cap)
            full_text = "\n\n".join(segs).strip()
            if len(full_text) < 25:
                _save_note_digest(n, "REFERENCE_ONLY", 0, used_model, db_path)
                reused += 1
                continue

            # Very long journal notes get dedicated full-note digestion so nothing is truncated
            if is_j and len(full_text) > per_note_cap * 1.4:
                if cur_batch:
                    batches.append(cur_batch)
                    cur_batch = []
                    cur_chars = 0
                batches.append([(n, is_j, full_text)])
                continue

            clipped = full_text[:per_note_cap]
            entry_len = len(clipped) + len(n["path"]) + 60
            if cur_batch and (
                len(cur_batch) >= batch_max_notes or cur_chars + entry_len > batch_max_chars
            ):
                batches.append(cur_batch)
                cur_batch = []
                cur_chars = 0
            cur_batch.append((n, is_j, clipped))
            cur_chars += entry_len

        if cur_batch:
            batches.append(cur_batch)

        processed = reused
        total_batches = len(batches)
        b_idx = 0
        while b_idx < total_batches:
            wave = batches[b_idx : b_idx + concurrency]
            wave_end = min(total_batches, b_idx + len(wave))
            first_date = wave[0][0][0]["note_date"]
            last_date = wave[-1][-1][0]["note_date"]
            processed += sum(len(b) for b in wave)
            yield {
                "stage": "reading",
                "notes_processed": processed,
                "total_notes": total_notes,
                "cached_notes": reused,
                "batch": wave_end,
                "total_batches": total_batches,
                "message": (
                    f"Reading {processed}/{total_notes} ({first_date} → {last_date}, "
                    f"batch {wave_end}/{total_batches} · {budget['label']})"
                ),
            }
            if len(wave) == 1:
                await _digest_note_batch(wave[0], cfg, used_model, int(budget["num_ctx"]), db_path)
            else:
                await asyncio.gather(
                    *(
                        _digest_note_batch(b, cfg, used_model, int(budget["num_ctx"]), db_path)
                        for b in wave
                    )
                )
            b_idx += len(wave)

        # Gather all digested insights chronologically with content_hash for monthly chapter caching
        with db_session(db_path) as conn:
            digest_rows = conn.execute(
                """
                SELECT d.note_path, d.note_date, d.folder, d.digest, d.is_personal,
                       n.content_hash
                FROM note_digests d
                JOIN notes n ON n.path = d.note_path
                ORDER BY COALESCE(n.note_date, '9999-99-99') ASC, n.mtime ASC
                """
            ).fetchall()

        journal_rows = [
            dict(r)
            for r in digest_rows
            if is_in_folders(r["note_path"], cfg.journal_folders)
            and r["digest"]
            and r["digest"] != "REFERENCE_ONLY"
        ]
        other_rows = [
            dict(r)
            for r in digest_rows
            if not is_in_folders(r["note_path"], cfg.journal_folders)
            and r["is_personal"] == 1
            and r["digest"]
            and r["digest"] != "REFERENCE_ONLY"
        ]
        if not journal_rows and not other_rows:
            other_rows = [
                {
                    "note_path": r["note_path"],
                    "note_date": r["note_date"],
                    "folder": r["folder"],
                    "content_hash": r["content_hash"],
                    "digest": f"{r['note_path']} ({r['folder']})",
                }
                for r in digest_rows[:25]
            ]

        analytics = compute_vault_analytics(db_path, cfg.journal_folders)
        anchor_paths = {a["note_path"]: a["in_links"] for a in analytics.get("anchor_notes", [])}
        num_ctx = int(budget["num_ctx"])
        group_char_cap = max(8000, int(budget["batch_max_chars"]))
        journal_final_budget = min(16000, max(7500, int(num_ctx * 1.1)))

        async def _fold_bounded(
            entries: List[str],
            label: str,
            target_max_chars: int,
            max_rounds: int = 2,
        ) -> AsyncIterator[Dict[str, Any]]:
            working = [e for e in entries if e and e.strip()]
            round_num = 1
            while working and round_num <= max_rounds and (
                len(working) > 10 or sum(len(x) for x in working) > target_max_chars
            ):
                if len(working) == 1:
                    working = [working[0][:target_max_chars]]
                    break

                groups: List[List[str]] = []
                cur_g: List[str] = []
                cur_len = 0
                for item in working:
                    if cur_g and (len(cur_g) >= 16 or cur_len + len(item) > group_char_cap):
                        groups.append(cur_g)
                        cur_g = []
                        cur_len = 0
                    cur_g.append(item)
                    cur_len += len(item) + 2
                if cur_g:
                    groups.append(cur_g)

                if len(groups) >= len(working):
                    groups = [working[i : i + 2] for i in range(0, len(working), 2)]

                next_level: List[str] = []
                total_g = len(groups)
                for g_idx in range(0, total_g, concurrency):
                    wave_g = groups[g_idx : g_idx + concurrency]
                    yield {
                        "type": "progress",
                        "message": (
                            f"Synthesizing {label} · Round {round_num}, "
                            f"group {min(total_g, g_idx + len(wave_g))}/{total_g} ({len(entries)} entries)…"
                        ),
                    }
                    prompts: List[Tuple[str, str]] = []
                    for g in wave_g:
                        each_cap = max(600, group_char_cap // max(1, len(g)))
                        trimmed_g = [item[:each_cap] for item in g]
                        fold_prompt = (
                            f"Consolidate these chronological {label} insights about {cfg.user_name} into one compact chapter summary "
                            "(5–8 concise bullets preserving specific people, dates, turning points, struggles, and values; keep under ~1,000 characters):\n\n"
                            + "\n\n---\n\n".join(trimmed_g)
                        )
                        prompts.append((fold_prompt, trimmed_g[0][:600]))
                    if len(prompts) == 1:
                        res_txt = await _call_chat_once(
                            cfg, used_model, prompts[0][0], temperature=0.25, num_ctx=num_ctx
                        )
                        next_level.append(res_txt.strip() or prompts[0][1])
                    else:
                        outs = await asyncio.gather(
                            *(
                                _call_chat_once(
                                    cfg, used_model, prm, temperature=0.25, num_ctx=num_ctx
                                )
                                for prm, _ in prompts
                            )
                        )
                        for idx_o, res_txt in enumerate(outs):
                            next_level.append(res_txt.strip() or prompts[idx_o][1])
                working = next_level
                round_num += 1

            if working and sum(len(x) for x in working) > target_max_chars:
                per_item = max(350, target_max_chars // len(working))
                working = [x[:per_item] for x in working]
            yield {"type": "result", "items": working}

        # --- Stage 2A: Monthly Cached Journal Chapters ---
        folded_journals: List[str] = []
        chapters_reused = 0
        chapters_generated = 0
        if journal_rows:
            month_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            undated_buf: List[Dict[str, Any]] = []
            for r in journal_rows:
                nd = (r.get("note_date") or "").strip()
                ym = nd[:7] if len(nd) >= 7 and re.match(r"^20\d{2}-\d{2}$", nd[:7]) else ""
                if ym:
                    month_buckets[ym].append(r)
                else:
                    undated_buf.append(r)

            bucket_items: List[Tuple[str, List[Dict[str, Any]]]] = [
                (ym, month_buckets[ym]) for ym in sorted(month_buckets.keys())
            ]
            for u_idx in range(0, len(undated_buf), 12):
                bucket_items.append((f"undated-{u_idx // 12 + 1:02d}", undated_buf[u_idx : u_idx + 12]))

            total_months = len(bucket_items)
            month_summaries: Dict[str, str] = {}
            uncached_months: List[Tuple[str, str, List[str]]] = []

            for b_key, mrows in bucket_items:
                sig_parts = [used_model] + [
                    f"{mr['note_path']}:{mr.get('content_hash') or ''}" for mr in mrows
                ]
                src_hash = hashlib.sha256("|".join(sig_parts).encode("utf-8")).hexdigest()[:24]
                cached_ch = _get_cached_chapter(db_path, f"journal:{b_key}", src_hash)
                if cached_ch:
                    month_summaries[b_key] = cached_ch
                    chapters_reused += 1
                    continue

                raw_entries = [
                    f"[{mr['note_date'] or 'undated'}] {mr['note_path']}:\n{mr['digest']}"
                    for mr in mrows
                ]
                if len(mrows) <= 3 and sum(len(x) for x in raw_entries) <= 1300:
                    compact = f"### {b_key} ({len(mrows)} entries)\n" + "\n".join(raw_entries)
                    _save_cached_chapter(
                        db_path, f"journal:{b_key}", "journal", src_hash, used_model, compact
                    )
                    month_summaries[b_key] = compact
                    chapters_reused += 1
                else:
                    uncached_months.append((b_key, src_hash, raw_entries))

            if uncached_months:
                done_months = chapters_reused
                sem = asyncio.Semaphore(concurrency)

                async def _build_month_chapter(
                    b_key: str, src_hash: str, raw_entries: List[str]
                ) -> Tuple[str, str]:
                    async with sem:
                        combined = "\n\n---\n\n".join(raw_entries)
                        # If an extraordinarily dense month exceeds one prompt, pre-chunk once cleanly
                        if len(combined) > group_char_cap:
                            mid = len(raw_entries) // 2
                            parts_m = [raw_entries[:mid], raw_entries[mid:]]
                            part_outs = []
                            for prt in parts_m:
                                sub_txt = "\n\n---\n\n".join(prt)[:group_char_cap]
                                sub_p = (
                                    f"Summarize this portion of {cfg.user_name}'s {b_key} journal entries into 4–6 concise bullets "
                                    "(preserving people, dates, feelings, struggles, and turning points):\n\n"
                                    + sub_txt
                                )
                                part_outs.append(
                                    await _call_chat_once(
                                        cfg, used_model, sub_p, temperature=0.25, num_ctx=num_ctx
                                    )
                                )
                            combined = "\n\n".join(part_outs)

                        m_prompt = (
                            f"Distill {cfg.user_name}'s journal entries for **{b_key}** ({len(raw_entries)} entries) "
                            "into a single crisp monthly chapter (4–7 bullets, ~400–850 characters max) preserving "
                            "key people, concrete events, emotional shifts, struggles, and values:\n\n"
                            + combined[:group_char_cap]
                        )
                        ans = await _call_chat_once(
                            cfg, used_model, m_prompt, temperature=0.25, num_ctx=num_ctx
                        )
                        chapter_md = f"### {b_key} ({len(raw_entries)} entries)\n{(ans.strip() or raw_entries[0][:400])}"
                        _save_cached_chapter(
                            db_path, f"journal:{b_key}", "journal", src_hash, used_model, chapter_md
                        )
                        return b_key, chapter_md

                for i_m in range(0, len(uncached_months), concurrency):
                    wave_m = uncached_months[i_m : i_m + concurrency]
                    labels_w = ", ".join(x[0] for x in wave_m)
                    done_months += len(wave_m)
                    yield {
                        "stage": "reducing",
                        "notes_processed": total_notes,
                        "total_notes": total_notes,
                        "message": (
                            f"Synthesizing monthly journal chapters · Round 1, "
                            f"{done_months}/{total_months} months ({labels_w})…"
                        ),
                    }
                    res_wave = await asyncio.gather(
                        *(_build_month_chapter(k, sh, re_l) for k, sh, re_l in wave_m)
                    )
                    for k, ch_md in res_wave:
                        month_summaries[k] = ch_md
                        chapters_generated += 1

            ordered_chapters = [month_summaries[k] for k, _ in bucket_items if k in month_summaries]
            if sum(len(c) for c in ordered_chapters) > journal_final_budget:
                async for ev in _fold_bounded(
                    ordered_chapters,
                    "monthly journal chapters",
                    target_max_chars=journal_final_budget,
                    max_rounds=2,
                ):
                    if ev["type"] == "progress":
                        yield {
                            "stage": "reducing",
                            "notes_processed": total_notes,
                            "total_notes": total_notes,
                            "message": ev["message"],
                        }
                    else:
                        folded_journals = ev["items"]
            else:
                folded_journals = ordered_chapters
                if chapters_reused and not chapters_generated:
                    yield {
                        "stage": "reducing",
                        "notes_processed": total_notes,
                        "total_notes": total_notes,
                        "message": f"Reused {chapters_reused} cached monthly journal chapters (Round 1)…",
                    }

        # --- Stage 2B: Smart High-Signal Supporting Vault Context ---
        folded_others: List[str] = []
        if other_rows:
            def _support_score(r: Dict[str, Any]) -> Tuple[int, int, str]:
                is_people = 1 if r.get("folder") == "People" else 0
                links = int(anchor_paths.get(r["note_path"], 0))
                nd = str(r.get("note_date") or "")
                return (is_people, links, nd)

            ranked_others = sorted(other_rows, key=_support_score, reverse=True)
            top_direct = ranked_others[:14]
            direct_entries = [
                f"[{r['note_date'] or 'undated'}] {r['note_path']}:\n{r['digest'][:320]}"
                for r in top_direct
            ]
            folded_others = direct_entries

            # Summarize broader secondary notes via cached supporting digest if large (>18)
            tail_others = ranked_others[14:46]
            if len(tail_others) >= 6:
                tail_sig = "|".join(
                    [used_model] + [f"{r['note_path']}:{r.get('content_hash') or ''}" for r in tail_others]
                )
                tail_hash = hashlib.sha256(tail_sig.encode("utf-8")).hexdigest()[:24]
                cached_sup = _get_cached_chapter(db_path, "supporting:vault-tail", tail_hash)
                if cached_sup:
                    folded_others.append(cached_sup)
                    chapters_reused += 1
                else:
                    yield {
                        "stage": "reducing",
                        "notes_processed": total_notes,
                        "total_notes": total_notes,
                        "message": f"Integrating supporting vault themes ({len(other_rows)} notes)…",
                    }
                    tail_txt = "\n\n".join(
                        f"[{r['folder']}] {r['note_path']}: {r['digest'][:240]}" for r in tail_others
                    )[:group_char_cap]
                    sup_prompt = (
                        f"Summarize the secondary interests, projects, studies, and media/creative threads across these {len(tail_others)} "
                        f"supporting notes for {cfg.user_name} in 4–6 concise bullets (~600 chars):\n\n{tail_txt}"
                    )
                    sup_ans = await _call_chat_once(
                        cfg, used_model, sup_prompt, temperature=0.25, num_ctx=num_ctx
                    )
                    sup_md = "### Broader Supporting Notes & Interests\n" + (sup_ans.strip() or "")
                    _save_cached_chapter(
                        db_path, "supporting:vault-tail", "supporting", tail_hash, used_model, sup_md
                    )
                    folded_others.append(sup_md)
                    chapters_generated += 1

        final_model = await _resolve_final_synthesis_model(cfg, used_model)
        yield {
            "stage": "reducing",
            "notes_processed": total_notes,
            "total_notes": total_notes,
            "message": f"Writing final Self-Portrait from {total_notes} notes using {final_model}…",
        }

        memories = get_memories(db_path)
        mem_block = "\n".join(
            f"- ({m['category']}) {m['content']}" for m in memories[:25]
        )

        # Classic ML cluster + relationship signals (analytics already computed above)
        ml_lines = []
        if analytics.get("clusters"):
            ml_lines.append(
                "Discovered Semantic Life Clusters: "
                + "; ".join(f"{c['label']} ({c['note_count']} notes)" for c in analytics["clusters"][:6])
            )
        if analytics.get("relationships"):
            ml_lines.append(
                "Most Mentioned People Across Vault: "
                + ", ".join(
                    f"{r['name']} ({r['mentions']}x, {r.get('first_seen') or '?'}→{r.get('last_seen') or '?'})"
                    for r in analytics["relationships"][:10]
                )
            )
        ml_block = ("=== STATISTICAL & GRAPH SIGNALS ===\n" + "\n".join(ml_lines)) if ml_lines else ""

        synthesis_sections = []
        if ml_block:
            synthesis_sections.append(ml_block)
        if folded_journals:
            synthesis_sections.append(
                "=== CORE JOURNAL & PERIODIC REFLECTIONS (Primary Grounding) ===\n"
                + "\n\n---\n\n".join(folded_journals)
            )
        if folded_others:
            synthesis_sections.append(
                "=== SUPPORTING VAULT CONTEXT (People, Projects, Media & Study) ===\n"
                + "\n\n---\n\n".join(folded_others)
            )

        final_prompt = (
            f"Based on this complete note-by-note chronological synthesis of all {total_notes} Obsidian notes and saved memories for {cfg.user_name}, "
            "write a deeply personal, honest, grounded Self-Portrait in Markdown (~450–650 words) to help them remember themselves.\n"
            "Give highest weight to their Core Journal & Periodic Reflections as the truest window into their inner life, "
            "while weaving in their people, projects, and broader interests from the rest of the vault.\n"
            "Use these exact sections:\n"
            "### Core Identity & Values\n"
            "### Your Story & How You've Evolved\n"
            "### Active Threads, Projects & Curiosities\n"
            "### Recurring Patterns & Inner Tensions\n"
            "### People & Anchors\n\n"
            "Be warm, specific, and faithful to the evidence across their timeline — never invent facts.\n\n"
            f"Saved Memories:\n{mem_block or '(none yet)'}\n\n"
            + "\n\n".join(synthesis_sections)
        )

        final_ctx = max(num_ctx, 12288)
        summary = await _call_chat_once(
            cfg, final_model, final_prompt, temperature=0.35, num_ctx=final_ctx
        )
        elapsed_s = round(time.monotonic() - t_start, 1)
        now = utc_now_iso()
        with db_session(db_path) as conn:
            note_count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            mem_count = len(memories)
            conn.execute(
                """
                INSERT INTO self_profile (
                    id, summary_md, updated_at, note_count_snapshot,
                    memory_count_snapshot, notes_covered, is_deep
                )
                VALUES (1, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(id) DO UPDATE SET
                    summary_md = excluded.summary_md,
                    updated_at = excluded.updated_at,
                    note_count_snapshot = excluded.note_count_snapshot,
                    memory_count_snapshot = excluded.memory_count_snapshot,
                    notes_covered = excluded.notes_covered,
                    is_deep = 1
                """,
                (summary, now, note_count, mem_count, total_notes),
            )

        yield {
            "stage": "done",
            "notes_processed": total_notes,
            "total_notes": total_notes,
            "elapsed_seconds": elapsed_s,
            "chapters_reused": chapters_reused,
            "chapters_generated": chapters_generated,
            "final_model": final_model,
            "profile": {
                "summary_md": summary,
                "updated_at": now,
                "note_count_snapshot": note_count,
                "memory_count_snapshot": mem_count,
                "notes_covered": total_notes,
                "is_deep": 1,
            },
        }


async def maybe_evolve_self_profile(
    cfg: AppConfig,
    db_path: Path = DB_PATH,
    min_cooldown_seconds: int = 600,
) -> Optional[Dict[str, object]]:
    """Automatically create or evolve the Self-Portrait when meaningful new notes/memories accrue."""
    if not cfg.auto_learn or _profile_lock.locked():
        return None

    if cfg.provider == "ollama":
        ollama = await check_ollama(cfg.ollama_url)
        if not ollama.get("online"):
            return None

    with db_session(db_path) as conn:
        note_count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
        mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    if note_count < 5 and mem_count < 3:
        return None

    profile = get_self_profile(db_path)
    if not profile or not profile.get("summary_md"):
        return await synthesize_self_profile(cfg, db_path=db_path)

    try:
        prev_dt = datetime.fromisoformat(profile["updated_at"])
        elapsed = (datetime.now(timezone.utc) - prev_dt).total_seconds()
        if elapsed < min_cooldown_seconds:
            return None
    except Exception:
        pass

    with db_session(db_path) as conn:
        edited_since = conn.execute(
            "SELECT COUNT(*) FROM notes WHERE indexed_at > ?",
            (profile["updated_at"],),
        ).fetchone()[0]

    prev_notes = int(profile.get("note_count_snapshot") or 0)
    prev_mems = int(profile.get("memory_count_snapshot") or 0)
    if (mem_count - prev_mems) >= 4 or abs(note_count - prev_notes) >= 10 or edited_since >= 8:
        if int(profile.get("is_deep") or 0) == 1:
            latest_profile = None
            async for ev in deep_synthesize_self_profile_stream(cfg, db_path=db_path):
                if ev.get("stage") == "done":
                    latest_profile = ev.get("profile")
            return latest_profile
        return await synthesize_self_profile(cfg, db_path=db_path)

    return None


def save_reflection_to_vault(
    cfg: AppConfig, title: str, content: str, subfolder: str = "Personal"
) -> str:
    """Explicitly save a reflection note into the user's Obsidian vault when requested."""
    vault_root = Path(cfg.vault_path)
    target_dir = vault_root / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_title = re.sub(r'[\\/:*?"<>|]+', "-", title).strip() or "Reflection"
    date_prefix = datetime.now().strftime("%Y-%m-%d")
    file_path = target_dir / f"{date_prefix} - {safe_title}.md"
    counter = 1
    while file_path.exists():
        counter += 1
        file_path = target_dir / f"{date_prefix} - {safe_title} ({counter}).md"

    frontmatter = (
        f"---\n"
        f"date: {date_prefix}\n"
        f"tags: [reflection, {cfg.companion_name.lower()}]\n"
        f"---\n\n"
        f"# {title}\n\n"
        f"{content.strip()}\n"
    )
    file_path.write_text(frontmatter, encoding="utf-8")
    return file_path.relative_to(vault_root).as_posix()
