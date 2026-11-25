"""Cross-platform hardware detection & local Ollama model recommendation engine."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from functools import lru_cache
from typing import Any, Dict, List, Optional


def _get_ram_gb() -> float:
    system = platform.system()
    try:
        if system == "Windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return round(stat.ullTotalPhys / (1024 ** 3), 1)
        elif system == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return round(int(out) / (1024 ** 3), 1)
        elif system == "Linux":
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 ** 2), 1)
    except Exception:
        pass
    return 8.0


def _get_gpu_info(ram_gb: float) -> Dict[str, Any]:
    system = platform.system()
    arch = platform.machine().lower()

    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=3,
            ).strip()
            if out:
                first = out.splitlines()[0]
                parts = [p.strip() for p in first.split(",")]
                name = parts[0]
                vram_gb = round(float(parts[1]) / 1024.0, 1) if len(parts) > 1 else 0.0
                return {"name": name, "vram_gb": vram_gb, "kind": "nvidia"}
        except Exception:
            pass

    if system == "Darwin" and ("arm" in arch or "aarch64" in arch):
        return {
            "name": "Apple Silicon Unified GPU",
            "vram_gb": round(ram_gb * 0.7, 1),
            "kind": "apple_silicon",
        }

    return {"name": "CPU / Integrated Graphics", "vram_gb": 0.0, "kind": "cpu"}


@lru_cache(maxsize=1)
def detect_hardware() -> Dict[str, Any]:
    """Detect basic CPU, RAM, and GPU specifications safely and locally."""
    ram_gb = _get_ram_gb()
    gpu = _get_gpu_info(ram_gb)
    return {
        "os": platform.system(),
        "arch": platform.machine(),
        "cpu_cores": os.cpu_count() or 4,
        "ram_gb": ram_gb,
        "gpu_name": gpu["name"],
        "vram_gb": gpu["vram_gb"],
        "gpu_kind": gpu["kind"],
    }


def recommend_models(
    installed_models: Optional[List[str]] = None,
    hw: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Recommend local Ollama models tailored to the detected hardware."""
    info = hw or detect_hardware()
    installed = installed_models or []
    installed_lower = {m.lower() for m in installed}

    def is_installed(name: str) -> bool:
        nl = name.lower()
        return nl in installed_lower or any(
            m.split(":")[0] == nl.split(":")[0] and nl == m for m in installed_lower
        )

    ram = float(info.get("ram_gb") or 8.0)
    vram = float(info.get("vram_gb") or 0.0)
    effective_mem = max(vram, ram * 0.65) if vram > 0 else ram * 0.5

    if vram >= 12 or (info.get("gpu_kind") == "apple_silicon" and ram >= 24) or (vram >= 8 and ram >= 32):
        tier = "performance"
        if vram >= 12 or (info.get("gpu_kind") == "apple_silicon" and ram >= 24):
            summary = (
                f"Strong local setup ({ram:.0f} GB RAM, {vram:.1f} GB GPU memory): "
                "you can run 14B models smoothly for rich nuance, or 7B–8B models at ultra-fast speeds."
            )
            candidates = [
                {
                    "name": "qwen2.5:14b",
                    "role": "Best Overall Nuance",
                    "size": "~9.0 GB",
                    "reason": "Deep reflection, faithful note citation, and nuanced emotional memory.",
                    "recommended": True,
                },
                {
                    "name": "qwen2.5:7b",
                    "role": "Fast Everyday Companion",
                    "size": "~4.7 GB",
                    "reason": "Snappy responses with strong grounding in your Obsidian notes.",
                    "recommended": False,
                },
                {
                    "name": "deepseek-r1:14b",
                    "role": "Deep Pattern Mirror",
                    "size": "~9.0 GB",
                    "reason": "Step-by-step reasoning across timelines and life patterns.",
                    "recommended": False,
                },
            ]
        else:
            # 8 GB VRAM + 32+ GB RAM (e.g., RTX 4060 8GB + 64GB RAM)
            summary = (
                f"Balanced GPU + high RAM ({ram:.0f} GB RAM, {vram:.1f} GB VRAM): "
                "7B–8B models fit entirely in VRAM for instant streaming; 14B models also work via GPU+RAM offload."
            )
            candidates = [
                {
                    "name": "qwen2.5:7b",
                    "role": "Best Speed & Grounding",
                    "size": "~4.7 GB",
                    "reason": "Fits completely in 8 GB VRAM for fast streaming and accurate vault recall.",
                    "recommended": True,
                },
                {
                    "name": "llama3.1:8b",
                    "role": "Warm Conversationalist",
                    "size": "~4.9 GB",
                    "reason": "Natural, empathetic tone for personal check-ins and journaling.",
                    "recommended": False,
                },
                {
                    "name": "deepseek-r1:7b",
                    "role": "Reflective Reasoner",
                    "size": "~4.7 GB",
                    "reason": "Shows thoughtful reasoning chains when tracing patterns in your vault.",
                    "recommended": False,
                },
                {
                    "name": "qwen2.5:14b",
                    "role": "Deeper Context (Offloaded)",
                    "size": "~9.0 GB",
                    "reason": "Uses your high system RAM + GPU for deeper synthesis when speed is less critical.",
                    "recommended": False,
                },
            ]
    elif effective_mem >= 7.5 or ram >= 14:
        tier = "balanced"
        summary = (
            f"Mid-range local setup ({ram:.0f} GB RAM"
            + (f", {vram:.1f} GB VRAM" if vram else "")
            + "): 7B–8B models offer the best balance of memory quality and speed."
        )
        candidates = [
            {
                "name": "qwen2.5:7b",
                "role": "Best Balanced Companion",
                "size": "~4.7 GB",
                "reason": "Strong instruction following and clean citations on 16 GB systems.",
                "recommended": True,
            },
            {
                "name": "llama3.2:3b",
                "role": "Ultra-Fast Light Option",
                "size": "~2.0 GB",
                "reason": "Instant replies if you want minimal battery or RAM usage.",
                "recommended": False,
            },
            {
                "name": "gemma2:9b",
                "role": "Thoughtful Prose",
                "size": "~5.4 GB",
                "reason": "Warm, articulate writing style for self-reflection.",
                "recommended": False,
            },
        ]
    else:
        tier = "lightweight"
        summary = (
            f"Compact hardware ({ram:.0f} GB RAM): 3B–4B models stay fast and responsive without swapping."
        )
        candidates = [
            {
                "name": "llama3.2:3b",
                "role": "Best Lightweight Companion",
                "size": "~2.0 GB",
                "reason": "Runs smoothly on 8 GB RAM laptops with warm conversational style.",
                "recommended": True,
            },
            {
                "name": "qwen2.5:3b",
                "role": "Sharp Compact Recall",
                "size": "~1.9 GB",
                "reason": "Great structured memory extraction on modest hardware.",
                "recommended": False,
            },
        ]

    for c in candidates:
        c["installed"] = is_installed(c["name"])

    embed_rec = {
        "name": "nomic-embed-text",
        "role": "Local Semantic Memory",
        "size": "~274 MB",
        "reason": "Powers offline meaning search across your notes and past chats.",
        "installed": any("nomic-embed-text" in m.lower() for m in installed),
    }

    return {
        "hardware": info,
        "tier": tier,
        "summary": summary,
        "chat_recommendations": candidates,
        "embedding_recommendation": embed_rec,
    }
