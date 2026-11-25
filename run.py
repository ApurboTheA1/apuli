"""One-command launcher for Apuli — Local Obsidian Companion."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import httpx
import uvicorn

from backend.config import load_config


def ensure_ollama_running(ollama_url: str) -> None:
    try:
        r = httpx.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=1.5)
        if r.status_code == 200:
            return
    except Exception:
        pass

    ollama_exe = shutil.which("ollama")
    if not ollama_exe:
        fallback = Path.home() / "AppData/Local/Programs/Ollama/ollama.exe"
        if fallback.exists():
            ollama_exe = str(fallback)

    if ollama_exe:
        print("Starting local Ollama server in background...")
        subprocess.Popen(
            [ollama_exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        for _ in range(15):
            time.sleep(0.5)
            try:
                r = httpx.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=1.0)
                if r.status_code == 200:
                    print("Ollama is online.")
                    return
            except Exception:
                continue


def main() -> None:
    cfg = load_config()
    ensure_ollama_running(cfg.ollama_url)
    host = "127.0.0.1"
    port = 8420
    url = f"http://{host}:{port}"
    print(f"Apuli — Local Obsidian Companion running at {url}")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    uvicorn.run("backend.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
