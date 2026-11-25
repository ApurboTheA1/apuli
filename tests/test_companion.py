"""Unit and API tests for Apuli — Local Obsidian Companion."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from backend.config import AppConfig
from backend.db import init_db
from backend.indexer import VaultIndexer, is_excluded
from backend.obsidian_parser import parse_obsidian_note
from backend.retrieval import HybridRetriever


def test_parse_obsidian_note_extracts_metadata_and_chunks(tmp_path: Path):
    vault = tmp_path / "Obsidian"
    personal = vault / "Personal"
    personal.mkdir(parents=True)
    note = personal / "2026-02-10 Reflection.md"
    note.write_text(
        "---\n"
        "date: 2026-02-10\n"
        "tags: [journal, growth]\n"
        "---\n\n"
        "# Morning Thoughts\n"
        "Talked with [[Rafi|my friend]] about starting the #podcast.\n\n"
        "## Evening\n"
        "Feeling calmer about the semester ahead.\n",
        encoding="utf-8",
    )

    parsed = parse_obsidian_note(note, vault)
    assert parsed.rel_path == "Personal/2026-02-10 Reflection.md"
    assert parsed.folder == "Personal"
    assert parsed.note_date == "2026-02-10"
    assert "journal" in parsed.tags
    assert "growth" in parsed.tags
    assert "podcast" in parsed.tags
    assert parsed.links == ["Rafi"]
    assert len(parsed.chunks) == 2
    assert "Morning Thoughts" in parsed.chunks[0].heading
    assert "Evening" in parsed.chunks[1].heading


def test_exclusion_rules():
    assert is_excluded(Path(".obsidian/workspace.json"), [])
    assert is_excluded(Path(".trash/old.md"), [])
    assert is_excluded(Path("CloudPass/creds.md"), ["CloudPass"])
    assert not is_excluded(Path("Personal/today.md"), ["CloudPass"])


def test_indexer_and_hybrid_retrieval(tmp_path: Path):
    vault = tmp_path / "Obsidian"
    (vault / "Personal").mkdir(parents=True)
    (vault / "People").mkdir(parents=True)
    (vault / "CloudPass").mkdir(parents=True)
    (vault / ".trash").mkdir(parents=True)

    (vault / "Personal" / "2026-01-15.md").write_text(
        "# Podcast Dream\nI want to record a heartfelt podcast episode with [[Tarek]] about identity.",
        encoding="utf-8",
    )
    (vault / "People" / "Tarek.md").write_text(
        "# Tarek\nOld friend from university who always encourages my creative projects.",
        encoding="utf-8",
    )
    (vault / "CloudPass" / "secret.md").write_text(
        "supersecretpassword123", encoding="utf-8"
    )
    (vault / ".trash" / "deleted.md").write_text(
        "deleted podcast note", encoding="utf-8"
    )

    db_file = tmp_path / "test.db"
    init_db(db_file)

    cfg = AppConfig(
        vault_path=str(vault),
        excluded_folders=["CloudPass"],
        ollama_url="http://127.0.0.1:59999",  # offline fallback test
    )
    indexer = VaultIndexer(lambda: cfg, db_path=db_file)
    status = asyncio.run(indexer.index_vault())
    assert status.indexed_notes == 2

    retriever = HybridRetriever(db_path=db_file)
    hits = asyncio.run(retriever.retrieve("podcast identity", cfg, folder_scope="all"))
    paths = [h.note_path for h in hits]
    assert "Personal/2026-01-15.md" in paths
    # Graph expansion should pull in linked note People/Tarek.md
    assert "People/Tarek.md" in paths
    assert all("CloudPass" not in p and ".trash" not in p for p in paths)


def test_api_endpoints_serve_ui_and_memories():
    from backend.app import app

    with TestClient(app) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert "Apuli" in root.text

        status = client.get("/api/status")
        assert status.status_code == 200
        data = status.json()
        assert "config" in data
        assert "index_status" in data

        conv = client.post("/api/conversations", json={"mode": "companion"})
        assert conv.status_code == 200
        conv_id = conv.json()["conversation"]["id"]

        mem = client.post(
            "/api/memories",
            json={"category": "value", "content": "Values honest self-reflection", "pinned": True},
        )
        assert mem.status_code == 200
        mem_id = mem.json()["memory"]["id"]

        client.delete(f"/api/memories/{mem_id}")
        client.delete(f"/api/conversations/{conv_id}")


def test_cross_chat_recall_and_auto_learning(tmp_path: Path, monkeypatch):
    from backend import memory as memory_mod
    from backend.db import db_session, get_memories, get_self_profile

    db_file = tmp_path / "learn.db"
    init_db(db_file)
    cfg = AppConfig(
        vault_path=str(tmp_path),
        ollama_url="http://127.0.0.1:59999",
        auto_learn=True,
        auto_distill_turns=2,
        enable_cross_chat_recall=True,
    )

    with db_session(db_file) as conn:
        conn.execute(
            """
            INSERT INTO conversations (id, title, mode, model, folder_scope, created_at, updated_at)
            VALUES ('c1', 'Late night thoughts', 'companion', 'qwen2.5:7b', 'all', '2026-11-20T10:00:00Z', '2026-11-20T10:00:00Z')
            """
        )
        conn.execute(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            VALUES ('c1', 'user', 'My grandfather used to call me Apuli and I miss those evenings.', '2026-11-20T10:00:00Z')
            """
        )
        m2 = conn.execute(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            VALUES ('c1', 'assistant', 'That name carries real warmth.', '2026-11-20T10:00:05Z')
            """
        ).lastrowid
        conn.execute(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            VALUES ('c1', 'user', 'I want to start recording honest podcast episodes again.', '2026-11-20T10:01:00Z')
            """
        )

    retriever = HybridRetriever(db_path=db_file)
    asyncio.run(
        retriever.index_chat_exchange(
            conversation_id="c1",
            message_id=int(m2),
            conv_title="Late night thoughts",
            user_text="My grandfather used to call me Apuli and I miss those evenings.",
            assistant_text="That name carries real warmth.",
            created_at="2026-11-20T10:00:05Z",
            cfg=cfg,
        )
    )

    # Searching from a DIFFERENT conversation ('c2') finds 'c1', while excluding 'c1' skips it
    hits_from_c2 = asyncio.run(
        retriever.retrieve_past_chats("grandfather Apuli", cfg, exclude_conv_id="c2")
    )
    assert len(hits_from_c2) == 1
    assert hits_from_c2[0].conversation_id == "c1"

    hits_from_c1 = asyncio.run(
        retriever.retrieve_past_chats("grandfather Apuli", cfg, exclude_conv_id="c1")
    )
    assert hits_from_c1 == []

    # Mock Ollama calls for auto-distillation & self-portrait evolution
    async def fake_chat_once(url, model, messages, temperature=0.3):
        prompt = messages[0]["content"]
        if "Return a valid JSON array" in prompt:
            return '[{"category": "relationship", "content": "Apurbo\'s grandfather used to call him Apuli."}]'
        return "### Core Identity & Values\nApurbo values honesty and warmth."

    async def fake_check_ollama(url):
        return {"online": True, "models": ["qwen2.5:7b"], "chat_models": ["qwen2.5:7b"], "embed_models": []}

    monkeypatch.setattr(memory_mod, "provider_chat_once", fake_chat_once)
    monkeypatch.setattr(memory_mod, "check_ollama", fake_check_ollama)

    saved = asyncio.run(memory_mod.maybe_auto_distill("c1", cfg, db_path=db_file))
    assert len(saved) == 1
    assert "Apuli" in get_memories(db_file)[0]["content"]

    # Second call immediately after should not re-distill because < 2 new user turns
    saved_again = asyncio.run(memory_mod.maybe_auto_distill("c1", cfg, db_path=db_file))
    assert saved_again == []

    # Add 2 more memories so mem_count >= 3 and test initial auto-synthesis of self_profile
    with db_session(db_file) as conn:
        conn.execute(
            "INSERT INTO memories (category, content, source_info, pinned, created_at) VALUES ('value', 'Values curiosity', 'manual', 0, '2026-11-20T10:02:00Z')"
        )
        conn.execute(
            "INSERT INTO memories (category, content, source_info, pinned, created_at) VALUES ('thread', 'Building podcast', 'manual', 0, '2026-11-20T10:03:00Z')"
        )

    prof = asyncio.run(memory_mod.maybe_evolve_self_profile(cfg, db_path=db_file))
    assert prof is not None
    stored = get_self_profile(db_file)
    assert stored is not None
    assert "Apurbo values honesty" in stored["summary_md"]


def test_hardware_recommendations_and_provider_masking():
    from backend.hardware import recommend_models
    from backend.providers import list_provider_models

    light = recommend_models([], {"ram_gb": 8.0, "vram_gb": 0.0, "gpu_kind": "cpu"})
    assert light["tier"] == "lightweight"
    assert any(m["name"] == "llama3.2:3b" and m["recommended"] for m in light["chat_recommendations"])

    perf = recommend_models(
        ["qwen2.5:7b", "nomic-embed-text:latest"],
        {"ram_gb": 60.0, "vram_gb": 8.0, "gpu_kind": "nvidia"},
    )
    assert perf["tier"] == "performance"
    assert perf["embedding_recommendation"]["installed"] is True
    qwen = next(m for m in perf["chat_recommendations"] if m["name"] == "qwen2.5:7b")
    assert qwen["installed"] is True

    cfg = AppConfig(
        openai_api_key="sk-secret-test-key-1234",
        openai_model="gpt-4o-mini",
    )
    pub = cfg.to_public_dict()
    assert pub["has_openai_key"] is True
    assert "sk-secret" not in pub["openai_api_key"]

    models = list_provider_models(cfg, {"chat_models": ["qwen2.5:7b"]})
    assert models[0]["is_local"] is True
    assert any(m["provider"] == "openai" and not m["is_local"] for m in models)
