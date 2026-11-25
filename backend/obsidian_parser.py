"""Obsidian Markdown parser with frontmatter, wikilink, tag, date, and heading-aware chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

FRONTMATTER_RE = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|\Z)", re.DOTALL)
WIKILINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
TAG_RE = re.compile(r"(?<![\w/])#([A-Za-z][A-Za-z0-9_\-/]*)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
DATE_FILENAME_RE = re.compile(r"(20\d{2}[-_]\d{2}[-_]\d{2})")
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)


@dataclass
class ParsedChunk:
    chunk_index: int
    heading: str
    content: str


@dataclass
class ParsedNote:
    rel_path: str
    title: str
    folder: str
    note_date: Optional[str]
    tags: List[str]
    links: List[str]
    frontmatter: Dict[str, Any]
    mtime: float
    content_hash: str
    word_count: int
    preview: str
    chunks: List[ParsedChunk]


def _extract_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    raw_yaml = match.group(1)
    body = text[match.end():]
    try:
        loaded = yaml.safe_load(raw_yaml)
        if isinstance(loaded, dict):
            return _sanitize_yaml(loaded), body
    except Exception:
        pass
    return {}, body


def _sanitize_yaml(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_yaml(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_yaml(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _extract_tags(frontmatter: Dict[str, Any], body: str) -> List[str]:
    tags = set()
    fm_tags = frontmatter.get("tags") or frontmatter.get("tag")
    if isinstance(fm_tags, str):
        for t in re.split(r"[,\s]+", fm_tags.strip()):
            t = t.lstrip("#").strip()
            if t:
                tags.add(t)
    elif isinstance(fm_tags, list):
        for item in fm_tags:
            if isinstance(item, str):
                t = item.lstrip("#").strip()
                if t:
                    tags.add(t)

    stripped_body = CODE_BLOCK_RE.sub("", body)
    for match in TAG_RE.finditer(stripped_body):
        tags.add(match.group(1))
    return sorted(tags)


def _extract_links(body: str) -> List[str]:
    links = []
    seen = set()
    for match in WIKILINK_RE.finditer(body):
        raw = match.group(1).strip()
        target = raw.split("|", 1)[0].split("#", 1)[0].strip()
        if target and target not in seen:
            seen.add(target)
            links.append(target)
    return links


def _extract_date(
    frontmatter: Dict[str, Any], rel_path: str, mtime: float
) -> Optional[str]:
    for key in ("date", "created", "created_at", "day"):
        val = frontmatter.get(key)
        if isinstance(val, str):
            m = DATE_FILENAME_RE.search(val.replace("/", "-"))
            if m:
                return m.group(1).replace("_", "-")
            if re.match(r"^\d{4}-\d{2}-\d{2}", val.strip()):
                return val.strip()[:10]

    m = DATE_FILENAME_RE.search(rel_path)
    if m:
        return m.group(1).replace("_", "-")

    try:
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
    except Exception:
        return None


def _chunk_markdown(
    body: str, title: str, max_chars: int = 950, overlap_chars: int = 160
) -> List[ParsedChunk]:
    lines = body.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    heading_stack: List[Tuple[int, str]] = []
    current_lines: List[str] = []
    current_heading = title

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            if current_lines:
                text_block = "\n".join(current_lines).strip()
                if text_block:
                    sections.append((current_heading, current_lines))
                current_lines = []
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            current_heading = " > ".join(h[1] for h in heading_stack)
        else:
            current_lines.append(line)

    if current_lines:
        text_block = "\n".join(current_lines).strip()
        if text_block:
            sections.append((current_heading, current_lines))

    if not sections:
        clean_body = body.strip()
        if not clean_body:
            return [ParsedChunk(chunk_index=0, heading=title, content=title)]
        sections = [(title, clean_body.splitlines())]

    chunks: List[ParsedChunk] = []
    idx = 0

    for heading, sec_lines in sections:
        sec_text = "\n".join(sec_lines).strip()
        if not sec_text:
            continue
        if len(sec_text) <= max_chars:
            chunks.append(ParsedChunk(chunk_index=idx, heading=heading, content=sec_text))
            idx += 1
            continue

        paragraphs = re.split(r"\n\s*\n", sec_text)
        buf = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) > max_chars:
                if buf:
                    chunks.append(ParsedChunk(chunk_index=idx, heading=heading, content=buf.strip()))
                    idx += 1
                    buf = ""
                start = 0
                while start < len(para):
                    end = min(start + max_chars, len(para))
                    piece = para[start:end].strip()
                    if piece:
                        chunks.append(ParsedChunk(chunk_index=idx, heading=heading, content=piece))
                        idx += 1
                    if end >= len(para):
                        break
                    start = max(0, end - overlap_chars)
                continue

            candidate = f"{buf}\n\n{para}".strip() if buf else para
            if len(candidate) <= max_chars:
                buf = candidate
            else:
                chunks.append(ParsedChunk(chunk_index=idx, heading=heading, content=buf.strip()))
                idx += 1
                tail = buf[-overlap_chars:] if len(buf) > overlap_chars else buf
                buf = f"{tail}\n\n{para}".strip()

        if buf.strip():
            chunks.append(ParsedChunk(chunk_index=idx, heading=heading, content=buf.strip()))
            idx += 1

    if not chunks:
        chunks.append(ParsedChunk(chunk_index=0, heading=title, content=title))

    return chunks


def parse_obsidian_note(file_path: Path, vault_root: Path) -> ParsedNote:
    """Read and parse a single Markdown note relative to the vault root."""
    rel_path = file_path.relative_to(vault_root).as_posix()
    folder = rel_path.split("/", 1)[0] if "/" in rel_path else "Root"
    title = file_path.stem
    stat = file_path.stat()
    mtime = stat.st_mtime

    raw_bytes = file_path.read_bytes()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    text = raw_bytes.decode("utf-8", errors="replace")

    frontmatter, body = _extract_frontmatter(text)
    tags = _extract_tags(frontmatter, body)
    links = _extract_links(body)
    note_date = _extract_date(frontmatter, rel_path, mtime)

    clean_for_preview = re.sub(r"\s+", " ", body).strip()
    preview = clean_for_preview[:240]
    word_count = len(body.split())
    chunks = _chunk_markdown(body, title)

    return ParsedNote(
        rel_path=rel_path,
        title=title,
        folder=folder,
        note_date=note_date,
        tags=tags,
        links=links,
        frontmatter=frontmatter,
        mtime=mtime,
        content_hash=content_hash,
        word_count=word_count,
        preview=preview,
        chunks=chunks,
    )
