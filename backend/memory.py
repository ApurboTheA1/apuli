"""Personal companion prompts, auto-distilled memory, and evolving self-portrait."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .config import AppConfig, DB_PATH
from .db import (
    db_session,
    get_conversation_messages,
    get_memories,
    get_self_profile,
    utc_now_iso,
)
from .ollama_client import check_ollama
from .providers import provider_chat_once
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

        used_model = model or cfg.chat_model
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
    """Synthesize or evolve the 'Who I Am' self-portrait from Personal/People notes & memories."""
    async with _profile_lock:
        existing_profile = get_self_profile(db_path)
        with db_session(db_path) as conn:
            note_count = conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            rows = conn.execute(
                """
                SELECT n.path, n.title, n.folder, n.note_date, c.content
                FROM notes n
                JOIN chunks c ON c.note_path = n.path AND c.chunk_index = 0
                WHERE n.folder IN ('Personal', 'People', 'Media', 'Inbox')
                ORDER BY COALESCE(n.note_date, '') DESC, n.mtime DESC
                LIMIT 22
                """
            ).fetchall()
            if not rows:
                rows = conn.execute(
                    """
                    SELECT n.path, n.title, n.folder, n.note_date, c.content
                    FROM notes n
                    JOIN chunks c ON c.note_path = n.path AND c.chunk_index = 0
                    ORDER BY COALESCE(n.note_date, '') DESC, n.mtime DESC
                    LIMIT 20
                    """
                ).fetchall()

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
                "Previous Self-Portrait (evolve and refine this rather than discarding important enduring truths):\n"
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
            "Note Samples:\n" + ("\n".join(samples) if samples else "(no notes indexed yet)")
        )

        used_model = model or cfg.chat_model
        summary = await provider_chat_once(
            cfg,
            used_model,
            [{"role": "user", "content": prompt}],
            temperature=0.35,
        )
        now = utc_now_iso()
        with db_session(db_path) as conn:
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
                (summary, now, note_count, mem_count),
            )
        return {
            "summary_md": summary,
            "updated_at": now,
            "note_count_snapshot": note_count,
            "memory_count_snapshot": mem_count,
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

    # Check cooldown
    try:
        prev_dt = datetime.fromisoformat(profile["updated_at"])
        elapsed = (datetime.now(timezone.utc) - prev_dt).total_seconds()
        if elapsed < min_cooldown_seconds:
            return None
    except Exception:
        pass

    prev_notes = int(profile.get("note_count_snapshot") or 0)
    prev_mems = int(profile.get("memory_count_snapshot") or 0)
    if (mem_count - prev_mems) >= 4 or abs(note_count - prev_notes) >= 10:
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
