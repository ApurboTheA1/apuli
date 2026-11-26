"""Configuration management for Apuli — Local Obsidian Companion."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_PATH = DATA_DIR / "config.json"
DB_PATH = DATA_DIR / "companion.db"

SYSTEM_EXCLUDED_DIRS = [".obsidian", ".trash", ".git", ".stfolder", "node_modules"]


def hash_pin(pin: str) -> str:
    """Hash a network access PIN with a random per-PIN salt using PBKDF2-SHA256."""
    clean = (pin or "").strip()
    if not clean:
        return ""
    salt_hex = secrets.token_hex(16)
    digest_hex = hashlib.pbkdf2_hmac(
        "sha256", clean.encode("utf-8"), bytes.fromhex(salt_hex), 200_000
    ).hex()
    return f"pbkdf2_sha256${salt_hex}${digest_hex}"


def verify_pin(pin: str, stored_hash: str) -> bool:
    """Constant-time verification of a PIN against a stored PBKDF2-SHA256 hash."""
    clean = (pin or "").strip()
    if not clean or not stored_hash:
        return False
    parts = stored_hash.split("$")
    if len(parts) != 3 or parts[0] != "pbkdf2_sha256":
        return False
    try:
        salt = bytes.fromhex(parts[1])
        expected = bytes.fromhex(parts[2])
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", clean.encode("utf-8"), salt, 200_000)
    return hmac.compare_digest(candidate, expected)


def _mask_key(val: str) -> str:
    if not val:
        return ""
    if len(val) <= 8:
        return "••••"
    return f"{val[:4]}••••{val[-4:]}"


@dataclass
class AppConfig:
    vault_path: str = ""
    ollama_url: str = "http://127.0.0.1:11434"
    provider: str = "ollama"  # ollama | openai | anthropic | gemini | openai_compatible
    chat_model: str = "qwen2.5:7b"
    memory_model: str = ""  # Optional dedicated model for memory distillation & self-portrait
    final_synthesis_model: str = ""  # Optional stronger model for final Self-Portrait pass ("" = Auto 14B if installed)
    embedding_model: str = "nomic-embed-text"
    excluded_folders: List[str] = field(default_factory=list)
    journal_folders: List[str] = field(default_factory=list)
    recent_vaults: List[str] = field(default_factory=list)
    user_name: str = "Friend"
    companion_name: str = "Apuli"
    default_mode: str = "companion"
    top_k_chunks: int = 8
    enable_graph_expansion: bool = True
    temperature: float = 0.65
    auto_learn: bool = True
    auto_distill_turns: int = 2
    enable_cross_chat_recall: bool = True
    ui_font: str = "system"
    font_size: int = 15
    parse_threads: int = 0          # 0 = Auto (uses detected CPU threads)
    embed_batch_size: int = 48      # Chunks per /api/embed batch (1..128)
    enable_ram_cache: bool = True   # SQLite 256MB mmap + 64MB page cache
    ollama_keep_alive: str = "20m"  # "0", "5m", "20m", "60m", "-1"
    ollama_num_thread: int = 0      # 0 = Ollama auto; >0 caps CPU threads for Ollama
    allow_lan: bool = False         # Bind 0.0.0.0 for local Wi-Fi / LAN access
    network_pin_hash: str = ""      # PBKDF2-SHA256 hash required for remote LAN access
    require_pin_on_localhost: bool = False  # Optionally lock localhost too

    # Optional Frontier / External API keys (kept local in data/config.json)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    custom_base_url: str = ""
    custom_api_key: str = ""
    custom_model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_public_dict(self) -> dict:
        data = asdict(self)
        data.pop("network_pin_hash", None)
        data["has_network_pin"] = bool(self.network_pin_hash)
        data["has_openai_key"] = bool(self.openai_api_key)
        data["has_anthropic_key"] = bool(self.anthropic_api_key)
        data["has_gemini_key"] = bool(self.gemini_api_key)
        data["has_custom_key"] = bool(self.custom_api_key)
        data["openai_api_key"] = _mask_key(self.openai_api_key)
        data["anthropic_api_key"] = _mask_key(self.anthropic_api_key)
        data["gemini_api_key"] = _mask_key(self.gemini_api_key)
        data["custom_api_key"] = _mask_key(self.custom_api_key)
        return data


def load_config() -> AppConfig:
    """Load configuration from disk, creating defaults if absent."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        cfg = AppConfig()
        save_config(cfg)
        return cfg

    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        defaults = AppConfig().to_dict()
        merged = {**defaults, **{k: v for k, v in raw.items() if k in defaults}}
        return AppConfig(**merged)
    except Exception:
        return AppConfig()


def save_config(cfg: AppConfig) -> AppConfig:
    """Persist configuration to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return cfg


def is_in_folders(note_rel_path: str, folders: List[str]) -> bool:
    """Return True if a relative note path lives inside any listed folder/subfolder."""
    if not folders:
        return False
    norm = note_rel_path.replace("\\", "/").strip("/").lower()
    for f in folders:
        clean = f.replace("\\", "/").strip("/").lower()
        if clean and (norm == clean or norm.startswith(clean + "/")):
            return True
    return False
