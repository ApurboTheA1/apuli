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
    assert any(m["name"] == "dolphin-llama3:8b" for m in perf["chat_recommendations"])

    cfg = AppConfig(
        chat_model="dolphin-llama3:8b",
        memory_model="qwen2.5:7b",
        openai_api_key="sk-secret-test-key-1234",
        openai_model="gpt-4o-mini",
    )
    pub = cfg.to_public_dict()
    assert pub["has_openai_key"] is True
    assert pub["memory_model"] == "qwen2.5:7b"
    assert "sk-secret" not in pub["openai_api_key"]

    models = list_provider_models(cfg, {"chat_models": ["qwen2.5:7b", "dolphin-llama3:8b"]})
    assert models[0]["is_local"] is True
    assert any(m["provider"] == "openai" and not m["is_local"] for m in models)


def test_memory_model_override_used_for_distillation(tmp_path: Path, monkeypatch):
    from backend import memory as memory_mod
    from backend.db import db_session

    db_file = tmp_path / "mem_override.db"
    init_db(db_file)
    with db_session(db_file) as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, mode, model, folder_scope, created_at, updated_at) VALUES ('c1', 'Chat', 'companion', 'dolphin-llama3:8b', 'all', '2026-11-20', '2026-11-20')"
        )
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES ('c1', 'user', 'Hello', '2026-11-20')"
        )
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES ('c1', 'assistant', 'Hi there', '2026-11-20')"
        )

    seen_models = []

    async def fake_provider_chat_once(cfg, model_spec, messages, temperature=0.3):
        seen_models.append(model_spec)
        return "[]"

    monkeypatch.setattr(memory_mod, "provider_chat_once", fake_provider_chat_once)
    cfg = AppConfig(chat_model="dolphin-llama3:8b", memory_model="qwen2.5:7b")
    asyncio.run(memory_mod.distill_memories_from_conversation("c1", cfg, db_path=db_file))
    assert seen_models == ["qwen2.5:7b"]


def test_deep_whole_vault_synthesis(tmp_path: Path, monkeypatch):
    from backend import memory as memory_mod
    from backend.db import get_self_profile

    vault = tmp_path / "Obsidian"
    (vault / "Personal").mkdir(parents=True)
    (vault / "College").mkdir(parents=True)

    # Create chronological personal notes + 1 reference note + 1 long multi-chunk note
    (vault / "Personal" / "note_1.md").write_text(
        "---\ndate: 2024-01-10\n---\n# Entry 1\nEarly reflection on purpose.",
        encoding="utf-8",
    )
    (vault / "College" / "ref.md").write_text(
        "---\ndate: 2024-06-15\n---\n# OS Notes\nProcess scheduling algorithm definitions.",
        encoding="utf-8",
    )
    (vault / "Personal" / "note_3.md").write_text(
        "---\ndate: 2026-02-20\n---\n# Entry 3\n" + ("Deep personal insight. " * 30),
        encoding="utf-8",
    )

    db_file = tmp_path / "deep.db"
    init_db(db_file)
    cfg = AppConfig(vault_path=str(vault), ollama_url="http://127.0.0.1:59999")
    indexer = VaultIndexer(lambda: cfg, db_path=db_file)
    asyncio.run(indexer.index_vault())

    prompts_seen = []

    async def fake_provider_chat_once(cfg_arg, model_spec, messages, temperature=0.3):
        p = messages[0]["content"]
        prompts_seen.append(p)
        if "Chronological Vault Synthesis" in p or "CORE JOURNAL" in p or "SUPPORTING VAULT" in p:
            return "### Core Identity & Values\nConsistent growth across 2024–2026."
        if "Analyze these" in p:
            return '{"1": "Early reflection on purpose.", "2": "REFERENCE_ONLY", "3": "Deep personal insight."}'
        if "Process scheduling algorithm" in p:
            return "REFERENCE_ONLY"
        return "- Personal growth and reflection noted."

    monkeypatch.setattr(memory_mod, "provider_chat_once", fake_provider_chat_once)

    async def run_stream():
        events = []
        async for ev in memory_mod.deep_synthesize_self_profile_stream(cfg, db_path=db_file):
            events.append(ev)
        return events

    events = asyncio.run(run_stream())
    assert events[-1]["stage"] == "done"
    assert events[-1]["total_notes"] == 3

    # Verify adaptive batching packed all 3 notes chronologically into 1 batch call + 1 final synthesis call
    batch_prompts = [p for p in prompts_seen if "Analyze these 3 chronological" in p]
    assert len(batch_prompts) == 1
    assert batch_prompts[0].index("2024-01-10") < batch_prompts[0].index("2024-06-15") < batch_prompts[0].index("2026-02-20")

    stored = get_self_profile(db_file)
    assert stored is not None
    assert stored["is_deep"] == 1
    assert stored["notes_covered"] == 3

    # Second run with no note changes should reuse all 3 cached digests (0 new batch calls)
    prompts_seen.clear()
    events2 = asyncio.run(run_stream())
    assert events2[-1]["stage"] == "done"
    assert [p for p in prompts_seen if "Analyze these" in p or "Read this" in p] == []

    # Switching vaults via clear_vault_data removes old notes, digests, and self_profile
    from backend.db import clear_vault_data, db_session

    clear_vault_data(db_file)
    assert get_self_profile(db_file) is None
    with db_session(db_file) as conn:
        assert conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM note_digests").fetchone()[0] == 0


def test_status_exposes_deep_status():
    from backend.app import app

    with TestClient(app) as client:
        res = client.get("/api/status")
        assert res.status_code == 200
        body = res.json()
        assert "deep_status" in body
        assert "state" in body["deep_status"]

        prof = client.get("/api/self-profile")
        assert prof.status_code == 200
        assert "digested_notes" in prof.json()


def test_journal_folders_discovery_scope_and_deep_priority(tmp_path: Path, monkeypatch):
    from backend import memory as memory_mod
    from backend.indexer import discover_journal_candidates

    vault = tmp_path / "Obsidian"
    (vault / "Personal" / "Journal").mkdir(parents=True)
    (vault / "College").mkdir(parents=True)

    (vault / "Personal" / "Journal" / "2026-01-01.md").write_text(
        "# New Year\nThinking deeply about creativity and family.", encoding="utf-8"
    )
    (vault / "College" / "Lecture1.md").write_text(
        "# Lecture 1\nCreativity and family sociology lecture definitions.", encoding="utf-8"
    )

    cands = discover_journal_candidates(vault)
    journal_cand = next(c for c in cands if c["path"] == "Personal/Journal")
    assert journal_cand["suggested"] is True
    assert journal_cand["note_count"] == 1

    db_file = tmp_path / "journal.db"
    init_db(db_file)
    cfg = AppConfig(
        vault_path=str(vault),
        journal_folders=["Personal/Journal"],
        ollama_url="http://127.0.0.1:59999",
    )
    indexer = VaultIndexer(lambda: cfg, db_path=db_file)
    asyncio.run(indexer.index_vault())

    retriever = HybridRetriever(db_path=db_file)
    journal_only = asyncio.run(
        retriever.retrieve("creativity family", cfg, folder_scope="journal")
    )
    assert [h.note_path for h in journal_only] == ["Personal/Journal/2026-01-01.md"]

    seen = []

    async def fake_chat_once(cfg_arg, model_spec, messages, temperature=0.3):
        p = messages[0]["content"]
        seen.append(p)
        if "CORE JOURNAL & PERIODIC REFLECTIONS" in p:
            return "### Core Identity & Values\nDeeply grounded in journals."
        if "Analyze these" in p:
            return '{"1": "Values creativity and family.", "2": "REFERENCE_ONLY"}'
        return "- Values creativity and family."

    monkeypatch.setattr(memory_mod, "provider_chat_once", fake_chat_once)

    async def run_deep():
        out = []
        async for ev in memory_mod.deep_synthesize_self_profile_stream(cfg, db_path=db_file):
            out.append(ev)
        return out

    events = asyncio.run(run_deep())
    assert events[-1]["stage"] == "done"
    assert any("JOURNAL" in p for p in seen)
    assert any("CORE JOURNAL & PERIODIC REFLECTIONS" in p for p in seen)


def test_adaptive_synthesis_budget_scales_with_hardware_and_model():
    from backend.hardware import compute_synthesis_budget

    cloud = compute_synthesis_budget("gemini", "gemini-2.5-flash")
    assert cloud["batch_max_notes"] == 30

    rtx4060 = compute_synthesis_budget("ollama", "qwen2.5:7b", {"ram_gb": 60.0, "vram_gb": 8.0, "gpu_kind": "nvidia"})
    assert rtx4060["batch_max_notes"] == 10

    small = compute_synthesis_budget("ollama", "llama3.2:3b", {"ram_gb": 8.0, "vram_gb": 0.0, "gpu_kind": "cpu"})
    assert small["batch_max_notes"] == 4


def test_vault_switch_endpoint_validation_and_recent_vaults(tmp_path: Path, monkeypatch):
    from backend import app as app_mod

    vault_a = tmp_path / "VaultA"
    vault_b = tmp_path / "VaultB"
    vault_a.mkdir()
    vault_b.mkdir()
    (vault_b / "hello.md").write_text("# Hello B", encoding="utf-8")

    db_file = tmp_path / "switch.db"
    init_db(db_file)
    state_cfg = AppConfig(vault_path=str(vault_a), ollama_url="http://127.0.0.1:59999")

    def fake_load():
        return state_cfg

    def fake_save(new_cfg):
        nonlocal state_cfg
        state_cfg = new_cfg
        return new_cfg

    monkeypatch.setattr(app_mod, "DB_PATH", db_file)
    monkeypatch.setattr(app_mod, "load_config", fake_load)
    monkeypatch.setattr(app_mod, "save_config", fake_save)
    monkeypatch.setattr(app_mod.indexer, "_db_path", db_file)
    monkeypatch.setattr(app_mod.retriever, "_db_path", db_file)

    with TestClient(app_mod.app) as client:
        bad = client.post("/api/vault/switch", json={"vault_path": str(tmp_path / "Missing")})
        assert bad.status_code == 400

        ok = client.post("/api/vault/switch", json={"vault_path": str(vault_b)})
        assert ok.status_code == 200
        cfg_data = ok.json()["config"]
        assert cfg_data["vault_path"] == str(vault_b)
        assert str(vault_a) in cfg_data["recent_vaults"]
        assert str(vault_b) in cfg_data["recent_vaults"]


def test_fs_folder_browser_detects_obsidian_vaults(tmp_path: Path):
    from backend.app import app

    vault_dir = tmp_path / "MyVault"
    (vault_dir / ".obsidian").mkdir(parents=True)
    (tmp_path / "OtherFolder").mkdir()

    with TestClient(app) as client:
        res = client.get("/api/vault/fs", params={"path": str(tmp_path)})
        assert res.status_code == 200
        data = res.json()
        assert data["current_path"] == str(tmp_path)
        by_name = {f["name"]: f for f in data["folders"]}
        assert by_name["MyVault"]["is_vault"] is True
        assert by_name["OtherFolder"]["is_vault"] is False


def test_configurable_performance_settings_and_ram_cache(tmp_path: Path):
    from backend.db import get_connection, set_ram_cache_enabled

    db_file = tmp_path / "perf.db"
    init_db(db_file)

    set_ram_cache_enabled(True)
    conn_on = get_connection(db_file)
    mmap_on = conn_on.execute("PRAGMA mmap_size;").fetchone()[0]
    cache_on = conn_on.execute("PRAGMA cache_size;").fetchone()[0]
    conn_on.close()
    assert mmap_on >= 268435456
    assert cache_on == -65536

    set_ram_cache_enabled(False)
    conn_off = get_connection(db_file)
    mmap_off = conn_off.execute("PRAGMA mmap_size;").fetchone()[0]
    cache_off = conn_off.execute("PRAGMA cache_size;").fetchone()[0]
    conn_off.close()
    assert mmap_off == 0
    assert cache_off == -2000
    set_ram_cache_enabled(True)

    cfg = AppConfig(
        parse_threads=0,
        embed_batch_size=64,
        enable_ram_cache=False,
        ollama_keep_alive="60m",
        ollama_num_thread=12,
    )
    pub = cfg.to_public_dict()
    assert pub["parse_threads"] == 0
    assert pub["embed_batch_size"] == 64
    assert pub["ollama_keep_alive"] == "60m"
    assert pub["ollama_num_thread"] == 12


def test_vault_analytics_temporal_and_on_this_day(tmp_path: Path, monkeypatch):
    from datetime import date
    from backend import app as app_mod
    from backend.analytics import (
        compute_vault_analytics,
        get_on_this_day,
        parse_temporal_filter,
    )

    # 1. Deterministic temporal query parser
    april = parse_temporal_filter("What happened in April 2023?", today=date(2026, 11, 26))
    assert april is not None
    assert april["start_date"] == "2023-04-01"
    assert april["end_date"].startswith("2023-04-")

    last_m = parse_temporal_filter("Show me notes from last month", today=date(2026, 3, 15))
    assert last_m is not None
    assert last_m["start_date"] == "2026-02-01"
    assert last_m["end_date"].startswith("2026-02-")

    yr = parse_temporal_filter("How was my mindset in 2024?", today=date(2026, 11, 26))
    assert yr is not None
    assert yr["start_date"] == "2024-01-01"
    assert yr["end_date"] == "2024-12-31"

    # 2. Index a small multi-year vault
    vault = tmp_path / "Vault"
    (vault / "Personal" / "Journal").mkdir(parents=True)
    (vault / "People").mkdir(parents=True)

    (vault / "People" / "Rafi.md").write_text(
        "# Rafi\nClose university friend and collaborator.",
        encoding="utf-8",
    )
    (vault / "Personal" / "Journal" / "2023-04-12.md").write_text(
        "---\ndate: 2023-04-12\n---\n"
        "Felt calm and grateful talking with [[Rafi]] about architecture and music on a peaceful evening.",
        encoding="utf-8",
    )
    (vault / "Personal" / "Journal" / "2024-04-13.md").write_text(
        "---\ndate: 2024-04-13\n---\n"
        "Feeling overwhelmed and anxious about difficult exams today, though Rafi checked in by text message.",
        encoding="utf-8",
    )
    (vault / "Personal" / "Journal" / "2024-11-26.md").write_text(
        "---\ndate: 2024-11-26\n---\n"
        "Looking back two years ago on gratitude, clarity, and quiet creative progress today.",
        encoding="utf-8",
    )
    (vault / "Personal" / "Journal" / "2025-11-25.md").write_text(
        "---\ndate: 2025-11-25\n---\n"
        "One year ago feeling peaceful after finishing the software prototype together with [[Rafi]].",
        encoding="utf-8",
    )

    db_file = tmp_path / "analytics.db"
    init_db(db_file)
    cfg = AppConfig(
        vault_path=str(vault),
        journal_folders=["Personal/Journal"],
        ollama_url="http://127.0.0.1:59999",
    )
    indexer = VaultIndexer(lambda: cfg, db_path=db_file)
    asyncio.run(indexer.index_vault())

    # 3. On This Day (±3 calendar days from prior years)
    otd = get_on_this_day(
        db_path=db_file,
        journal_folders=["Personal/Journal"],
        today=date(2026, 11, 26),
        limit=5,
    )
    otd_dates = {item["note_date"] for item in otd}
    assert "2024-11-26" in otd_dates
    assert "2025-11-25" in otd_dates
    assert "2023-04-12" not in otd_dates

    # 4. Graph PageRank, People relationship tracking (linked + unlinked), timeline & TF-IDF
    analytics = compute_vault_analytics(db_path=db_file, journal_folders=["Personal/Journal"])
    assert isinstance(analytics["clusters"], list)
    assert len(analytics["anchor_notes"]) >= 1
    assert any(a["title"] == "Rafi" or a["note_path"].endswith("Rafi.md") for a in analytics["anchor_notes"])

    rel_map = {r["name"]: r for r in analytics["relationships"]}
    assert "Rafi" in rel_map
    assert rel_map["Rafi"]["linked_mentions"] == 2
    assert rel_map["Rafi"]["mentions"] == 3  # includes 1 unlinked mention in 2024-04-13

    months = {m["month"]: m for m in analytics["timeline"]}
    assert "2023-04" in months
    assert "2024-04" in months
    assert months["2023-04"]["sentiment"] > months["2024-04"]["sentiment"]
    assert len(months["2023-04"]["top_terms"]) >= 1

    # 5. Temporal filtering inside HybridRetriever (direct hits obey the window; graph expansion may attach linked People notes)
    retriever = HybridRetriever(db_path=db_file)
    hits = asyncio.run(retriever.retrieve("April 2023", cfg, folder_scope="all"))
    direct_hits = [h for h in hits if h.match_reason != "linked note"]
    assert any(h.note_date == "2023-04-12" for h in direct_hits)
    assert all((h.note_date or "").startswith("2023-04") for h in direct_hits)

    # 6. /api/analytics HTTP endpoint
    monkeypatch.setattr(app_mod, "DB_PATH", db_file)
    monkeypatch.setattr(app_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(app_mod, "_analytics_cache", None)
    with TestClient(app_mod.app) as client:
        resp = client.get("/api/analytics")
        assert resp.status_code == 200
        payload = resp.json()
        assert "relationships" in payload
        assert "timeline" in payload
        assert "clusters" in payload


def test_network_lan_pin_authentication_and_lockout(tmp_path: Path, monkeypatch):
    from backend import app as app_mod
    from backend.config import hash_pin, verify_pin

    # 1. PIN hashing & constant-time verification
    h = hash_pin("2468")
    assert h.startswith("pbkdf2_sha256$")
    assert verify_pin("2468", h) is True
    assert verify_pin("9999", h) is False

    db_file = tmp_path / "pin_test.db"
    init_db(db_file)
    state_cfg = AppConfig(ollama_url="http://127.0.0.1:59999")

    def fake_load():
        return state_cfg

    def fake_save(new_cfg):
        nonlocal state_cfg
        state_cfg = new_cfg
        return new_cfg

    monkeypatch.setattr(app_mod, "DB_PATH", db_file)
    monkeypatch.setattr(app_mod, "load_config", fake_load)
    monkeypatch.setattr(app_mod, "save_config", fake_save)
    monkeypatch.setattr(app_mod.indexer, "_db_path", db_file)
    monkeypatch.setattr(app_mod.retriever, "_db_path", db_file)
    app_mod._auth_sessions.clear()
    app_mod._auth_failures.clear()

    # 2. Remote client blocked (403) when no PIN has been configured yet
    with TestClient(app_mod.app, client=("192.168.1.50", 51000)) as remote_client:
        assert remote_client.get("/").status_code == 200
        st = remote_client.get("/api/auth/status").json()
        assert st["is_remote"] is True
        assert st["locked"] is True
        assert st["blocked_no_pin"] is True
        assert remote_client.get("/api/status").status_code == 403

    # 3. Localhost stays unlocked and sets the Network PIN via /api/settings
    with TestClient(app_mod.app, client=("127.0.0.1", 51001)) as local_client:
        res = local_client.post(
            "/api/settings",
            json={"allow_lan": True, "network_pin": "2468"},
        )
        assert res.status_code == 200
        pub = res.json()["config"]
        assert "network_pin_hash" not in pub
        assert pub["has_network_pin"] is True
        assert pub["allow_lan"] is True
        assert local_client.get("/api/status").status_code == 200

    # 4. Remote client requires PIN, enforces lockout after 5 bad tries, unlocks cleanly with right PIN
    with TestClient(app_mod.app, client=("192.168.1.51", 51002)) as bad_remote:
        assert bad_remote.get("/api/status").status_code == 401
        for _ in range(4):
            assert bad_remote.post("/api/auth/unlock", json={"pin": "0000"}).status_code == 401
        assert bad_remote.post("/api/auth/unlock", json={"pin": "0000"}).status_code == 429
        assert bad_remote.get("/api/auth/status").json()["retry_after_seconds"] > 0

    with TestClient(app_mod.app, client=("192.168.1.52", 51003)) as good_remote:
        assert good_remote.get("/api/status").status_code == 401
        unlock = good_remote.post("/api/auth/unlock", json={"pin": "2468"})
        assert unlock.status_code == 200
        assert good_remote.get("/api/status").status_code == 200

        # Lock session invalidates cookie immediately
        assert good_remote.post("/api/auth/lock").status_code == 200
        assert good_remote.get("/api/status").status_code == 401


def test_deep_synthesis_reduce_terminates_on_long_digests(tmp_path: Path, monkeypatch):
    from backend import memory as memory_mod

    vault = tmp_path / "Vault"
    (vault / "Journal").mkdir(parents=True)
    for i in range(18):
        (vault / "Journal" / f"2024-01-{i+1:02d}.md").write_text(
            f"# Day {i+1}\nLong personal journal entry about growth, creativity, and reflection number {i+1}.",
            encoding="utf-8",
        )

    db_file = tmp_path / "reduce_loop.db"
    init_db(db_file)
    cfg = AppConfig(
        vault_path=str(vault),
        journal_folders=["Journal"],
        ollama_url="http://127.0.0.1:59999",
    )
    indexer = VaultIndexer(lambda: cfg, db_path=db_file)
    asyncio.run(indexer.index_vault())

    calls = 0

    async def verbose_chat_once(cfg_arg, model_spec, messages, temperature=0.3, **kwargs):
        nonlocal calls
        calls += 1
        p = messages[0]["content"]
        if "CORE JOURNAL & PERIODIC REFLECTIONS" in p:
            assert len(p) < 16000
            return "### Core Identity & Values\nTerminated cleanly."
        if "Analyze these" in p:
            import json as _json
            return _json.dumps({str(k): "Detailed personal insight " * 30 for k in range(1, 20)})
        # Intentionally return an oversized summary (>5500 chars) that previously triggered an infinite loop
        return ("Oversized chapter bullet point about identity and growth.\n" * 120)

    monkeypatch.setattr(memory_mod, "provider_chat_once", verbose_chat_once)

    async def run_deep():
        events = []
        async for ev in memory_mod.deep_synthesize_self_profile_stream(cfg, db_path=db_file):
            events.append(ev)
        return events

    events = asyncio.run(run_deep())
    assert events[-1]["stage"] == "done"
    assert calls < 30
    assert any(
        ev["stage"] == "reducing" and "Round " in ev.get("message", "")
        for ev in events
    )



