"""One-command launcher for Apuli — Local Obsidian Companion."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import List

import httpx
import uvicorn

from backend.config import load_config


def _detect_lan_ips() -> List[str]:
    ips: List[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            primary = s.getsockname()[0]
            if primary and not primary.startswith("127."):
                ips.append(primary)
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    return ips


def _parse_arg_value(flag: str, default: str = "") -> str:
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return default


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

    cli_host = _parse_arg_value("--host")
    port_str = _parse_arg_value("--port", "8420")
    try:
        port = int(port_str)
    except ValueError:
        port = 8420

    if cli_host:
        host = cli_host
    elif "--lan" in sys.argv:
        host = "0.0.0.0"
    elif "--local" in sys.argv:
        host = "127.0.0.1"
    else:
        host = "0.0.0.0" if cfg.allow_lan else "127.0.0.1"

    os.environ["APULI_BIND_HOST"] = host
    os.environ["APULI_PORT"] = str(port)

    local_url = f"http://127.0.0.1:{port}"
    print(f"Apuli — Local Obsidian Companion running at {local_url}")
    if host == "0.0.0.0":
        lan_ips = _detect_lan_ips()
        for ip in lan_ips:
            print(f"  LAN URL: http://{ip}:{port}")
        if not cfg.network_pin_hash:
            print(
                "  WARNING: No Network PIN is set yet. Remote devices will be blocked "
                "until you set a PIN in Settings -> Network Access & PIN Lock."
            )
        else:
            print("  Network PIN lock: ENABLED for remote devices.")

    if "--no-browser" not in sys.argv:
        webbrowser.open(local_url)
    uvicorn.run("backend.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
