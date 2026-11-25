"""Async client for talking to the local Ollama instance."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

EMBED_HINTS = (
    "embed",
    "nomic",
    "mxbai",
    "bge-",
    "minilm",
    "arctic-embed",
)


def is_embedding_model(name: str) -> bool:
    lower = name.lower()
    return any(hint in lower for hint in EMBED_HINTS)


async def check_ollama(ollama_url: str) -> Dict[str, Any]:
    """Check if Ollama is reachable and list available models."""
    url = ollama_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(f"{url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
            embed_models = [m for m in models if is_embedding_model(m)]
            chat_models = [m for m in models if m not in embed_models]
            return {
                "online": True,
                "models": models,
                "chat_models": chat_models,
                "embed_models": embed_models,
                "error": None,
            }
    except Exception as exc:
        return {
            "online": False,
            "models": [],
            "chat_models": [],
            "embed_models": [],
            "error": str(exc),
        }


async def has_model(ollama_url: str, model_name: str) -> bool:
    info = await check_ollama(ollama_url)
    if not info["online"]:
        return False
    target = model_name.lower()
    for m in info["models"]:
        ml = m.lower()
        if ml == target or ml.split(":")[0] == target.split(":")[0] and ":" not in target:
            return True
    return False


async def embed_texts(
    ollama_url: str, model: str, texts: List[str]
) -> Optional[List[List[float]]]:
    """Generate embeddings via Ollama /api/embed. Returns None if unavailable."""
    if not texts:
        return []
    url = ollama_url.rstrip("/")
    safe_texts = [t[:3500] if len(t) > 3500 else t for t in texts]
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{url}/api/embed",
                json={"model": model, "input": safe_texts},
            )
            if resp.status_code == 404:
                # Fallback for older Ollama versions supporting /api/embeddings
                vectors: List[List[float]] = []
                for item in safe_texts:
                    r2 = await client.post(
                        f"{url}/api/embeddings",
                        json={"model": model, "prompt": item},
                    )
                    r2.raise_for_status()
                    vectors.append(r2.json()["embedding"])
                return vectors
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings")
            if isinstance(embeddings, list) and len(embeddings) == len(safe_texts):
                return embeddings
            return None
    except Exception:
        return None


async def pull_model_stream(
    ollama_url: str, model: str
) -> AsyncIterator[Dict[str, Any]]:
    """Stream model pull progress from Ollama."""
    url = ollama_url.rstrip("/")
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", f"{url}/api/pull", json={"model": model, "stream": True}
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


async def chat_stream(
    ollama_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.65,
) -> AsyncIterator[Tuple[str, str]]:
    """
    Stream chat responses from Ollama /api/chat.
    Yields tuples of (event_type, text) where event_type is 'thinking' or 'token'.
    Handles inline <think>...</think> blocks cleanly.
    """
    url = ollama_url.rstrip("/")
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_ctx": 8192,
        },
    }

    in_think = False
    buffer = ""

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{url}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg = data.get("message") or {}
                if msg.get("thinking"):
                    yield ("thinking", msg["thinking"])

                chunk = msg.get("content") or ""
                if chunk:
                    buffer += chunk
                    while buffer:
                        if in_think:
                            end_idx = buffer.find("</think>")
                            if end_idx != -1:
                                part = buffer[:end_idx]
                                if part:
                                    yield ("thinking", part)
                                buffer = buffer[end_idx + len("</think>"):]
                                in_think = False
                            else:
                                # Keep trailing chars in case '</think>' is split across chunks
                                if len(buffer) > 8:
                                    yield ("thinking", buffer[:-8])
                                    buffer = buffer[-8:]
                                break
                        else:
                            start_idx = buffer.find("<think>")
                            if start_idx != -1:
                                part = buffer[:start_idx]
                                if part:
                                    yield ("token", part)
                                buffer = buffer[start_idx + len("<think>"):]
                                in_think = True
                            else:
                                # Keep up to 7 chars in case '<think>' is split across chunks
                                if len(buffer) > 7 and "<" in buffer[-7:]:
                                    cut = buffer.rfind("<")
                                    if cut > 0:
                                        yield ("token", buffer[:cut])
                                        buffer = buffer[cut:]
                                    break
                                yield ("token", buffer)
                                buffer = ""

                if data.get("done"):
                    if buffer:
                        yield ("thinking" if in_think else "token", buffer)
                        buffer = ""
                    break


async def chat_once(
    ollama_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
) -> str:
    """Run a non-streaming chat completion and strip <think> blocks."""
    url = ollama_url.rstrip("/")
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": 8192},
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(f"{url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = (data.get("message") or {}).get("content") or ""
        # Remove <think>...</think> if present
        while "<think>" in content and "</think>" in content:
            s = content.find("<think>")
            e = content.find("</think>", s)
            if e == -1:
                break
            content = content[:s] + content[e + len("</think>"):]
        return content.strip()
