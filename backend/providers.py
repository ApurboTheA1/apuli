"""Unified Local-First AI provider layer (Ollama primary + optional Frontier APIs)."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Tuple

import httpx

from .config import AppConfig
from .ollama_client import chat_once as ollama_chat_once
from .ollama_client import chat_stream as ollama_chat_stream


def list_provider_models(cfg: AppConfig, ollama_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return selectable models across local Ollama (first) and any configured APIs."""
    choices: List[Dict[str, Any]] = []

    ollama_models = ollama_info.get("chat_models") or []
    if not ollama_models and cfg.chat_model and cfg.provider == "ollama":
        ollama_models = [cfg.chat_model]
    for m in ollama_models:
        choices.append(
            {
                "id": f"ollama::{m}",
                "provider": "ollama",
                "model": m,
                "label": f"Local · {m}",
                "is_local": True,
            }
        )

    if cfg.openai_api_key:
        m = cfg.openai_model or "gpt-4o-mini"
        choices.append(
            {
                "id": f"openai::{m}",
                "provider": "openai",
                "model": m,
                "label": f"Cloud (OpenAI) · {m}",
                "is_local": False,
            }
        )

    if cfg.anthropic_api_key:
        m = cfg.anthropic_model or "claude-sonnet-4-20250514"
        choices.append(
            {
                "id": f"anthropic::{m}",
                "provider": "anthropic",
                "model": m,
                "label": f"Cloud (Anthropic) · {m}",
                "is_local": False,
            }
        )

    if cfg.gemini_api_key:
        m = cfg.gemini_model or "gemini-2.5-flash"
        choices.append(
            {
                "id": f"gemini::{m}",
                "provider": "gemini",
                "model": m,
                "label": f"Cloud (Gemini) · {m}",
                "is_local": False,
            }
        )

    if cfg.custom_base_url and cfg.custom_model:
        is_local_url = any(
            h in cfg.custom_base_url.lower()
            for h in ("127.0.0.1", "localhost", "0.0.0.0")
        )
        prefix = "Local Custom" if is_local_url else "Cloud Custom"
        choices.append(
            {
                "id": f"openai_compatible::{cfg.custom_model}",
                "provider": "openai_compatible",
                "model": cfg.custom_model,
                "label": f"{prefix} · {cfg.custom_model}",
                "is_local": is_local_url,
            }
        )

    return choices


def parse_model_spec(
    model_spec: str, default_provider: str = "ollama"
) -> Tuple[str, str]:
    if "::" in model_spec:
        prov, mod = model_spec.split("::", 1)
        return prov.strip(), mod.strip()
    return default_provider, model_spec.strip()


async def _openai_stream(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
) -> AsyncIterator[Tuple[str, str]]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0].get("delta") or {}
                    txt = delta.get("content")
                    if txt:
                        yield ("token", txt)
                except Exception:
                    continue


async def _anthropic_stream(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
) -> AsyncIterator[Tuple[str, str]]:
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    chat_msgs = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] != "system"
    ]
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 2048,
        "temperature": temperature,
        "system": "\n\n".join(system_parts),
        "messages": chat_msgs,
        "stream": True,
    }
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", "https://api.anthropic.com/v1/messages", headers=headers, json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                try:
                    data = json.loads(line[5:].strip())
                    if data.get("type") == "content_block_delta":
                        txt = (data.get("delta") or {}).get("text")
                        if txt:
                            yield ("token", txt)
                except Exception:
                    continue


async def _gemini_stream(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
) -> AsyncIterator[Tuple[str, str]]:
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"temperature": temperature},
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:streamGenerateContent?alt=sse&key={api_key}"
    )
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                try:
                    data = json.loads(line[5:].strip())
                    cands = data.get("candidates") or []
                    if cands:
                        parts = ((cands[0].get("content") or {}).get("parts")) or []
                        for p in parts:
                            if p.get("text"):
                                yield ("token", p["text"])
                except Exception:
                    continue


async def provider_chat_stream(
    cfg: AppConfig,
    model_spec: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.65,
) -> AsyncIterator[Tuple[str, str]]:
    """Stream chat tokens from local Ollama (default) or an optional external API."""
    provider, model = parse_model_spec(model_spec, cfg.provider or "ollama")

    if provider == "ollama":
        async for ev in ollama_chat_stream(cfg.ollama_url, model, messages, temperature):
            yield ev
        return

    if provider == "openai":
        if not cfg.openai_api_key:
            raise ValueError("OpenAI API key is not configured.")
        async for ev in _openai_stream(
            "https://api.openai.com/v1", cfg.openai_api_key, model, messages, temperature
        ):
            yield ev
        return

    if provider == "openai_compatible":
        if not cfg.custom_base_url:
            raise ValueError("Custom OpenAI-compatible base URL is not configured.")
        async for ev in _openai_stream(
            cfg.custom_base_url, cfg.custom_api_key, model, messages, temperature
        ):
            yield ev
        return

    if provider == "anthropic":
        if not cfg.anthropic_api_key:
            raise ValueError("Anthropic API key is not configured.")
        async for ev in _anthropic_stream(
            cfg.anthropic_api_key, model, messages, temperature
        ):
            yield ev
        return

    if provider == "gemini":
        if not cfg.gemini_api_key:
            raise ValueError("Gemini API key is not configured.")
        async for ev in _gemini_stream(
            cfg.gemini_api_key, model, messages, temperature
        ):
            yield ev
        return

    raise ValueError(f"Unsupported provider: {provider}")


async def provider_chat_once(
    cfg: AppConfig,
    model_spec: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    num_ctx: int = 8192,
) -> str:
    """Non-streaming completion via local Ollama (preferred) or active provider."""
    provider, model = parse_model_spec(model_spec, cfg.provider or "ollama")
    if provider == "ollama":
        return await ollama_chat_once(cfg.ollama_url, model, messages, temperature, num_ctx=num_ctx)

    parts: List[str] = []
    async for ev_type, text in provider_chat_stream(cfg, model_spec, messages, temperature):
        if ev_type == "token":
            parts.append(text)
    return "".join(parts).strip()
