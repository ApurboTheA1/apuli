"""Classic ML, Graph Theory, TF-IDF, Sentiment Pulse, On This Day, and Temporal Query Parser.

Runs purely on CPU + NumPy + SQLite (zero LLM calls, millisecond latency).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import DB_PATH, is_in_folders
from .db import db_session

WORD_RE = re.compile(r"[A-Za-z][A-Za-z']{2,}")

ANALYTICS_STOPWORDS = {
    "the", "and", "that", "have", "for", "not", "with", "you", "this", "but",
    "his", "from", "they", "she", "her", "say", "will", "one", "all", "would",
    "there", "their", "what", "out", "about", "who", "get", "which", "when",
    "make", "can", "like", "time", "just", "him", "know", "take", "into",
    "year", "your", "good", "some", "could", "them", "see", "other", "than",
    "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well",
    "way", "even", "new", "want", "because", "any", "these", "give", "day",
    "most", "was", "were", "been", "being", "had", "has", "are", "did",
    "doing", "done", "got", "getting", "feel", "feeling", "felt", "really",
    "very", "much", "more", "still", "too", "here", "where", "why", "while",
    "today", "yesterday", "tomorrow", "around", "things", "thing", "something",
    "anything", "everything", "nothing", "going", "went", "came", "made",
    "said", "told", "though", "although", "maybe", "probably", "actually",
    "literally", "kinda", "sorta", "don't", "didn't", "can't", "won't",
    "i'm", "i've", "i'll", "it's", "that's", "there's", "wasn't", "isn't",
    "aren't", "couldn't", "wouldn't", "shouldn't", "note", "notes", "journal",
}

POS_LEXICON = {
    "happy", "grateful", "thankful", "calm", "peace", "peaceful", "excited",
    "hopeful", "hope", "love", "loved", "loving", "proud", "joy", "joyful",
    "good", "great", "better", "best", "clear", "clarity", "confident",
    "motivated", "inspired", "creative", "relief", "relieved", "warm",
    "kind", "caring", "strong", "growing", "growth", "progress", "smile",
    "laugh", "laughed", "fun", "enjoy", "enjoyed", "blessed", "content",
    "energized", "focused", "determined", "free", "safe", "honest",
    "meaningful", "worth", "accomplished", "success", "win", "chill",
    "relaxed", "alhamdulillah", "shukriya", "awesome", "amazing", "bright",
}

NEG_LEXICON = {
    "sad", "angry", "anxious", "anxiety", "stressed", "stress", "tired",
    "exhausted", "lonely", "alone", "empty", "worthless", "guilty", "guilt",
    "shame", "ashamed", "fear", "afraid", "scared", "worried", "worry",
    "overwhelmed", "confused", "lost", "stuck", "frustrated", "mad", "hate",
    "hated", "hurt", "pain", "painful", "cry", "cried", "crying", "depressed",
    "hopeless", "numb", "burnout", "burned", "drained", "regret", "regretted",
    "fail", "failed", "failure", "insecure", "doubt", "doubting", "helpless",
    "broken", "miserable", "upset", "terrible", "awful", "bad", "worse",
    "worst", "dark", "heavy", "panic", "miss", "missing", "toxic", "heated",
    "conflict", "argument", "fight", "fought", "struggle", "struggling",
}

NEGATORS = {"not", "no", "never", "don't", "didn't", "wasn't", "isn't", "cant", "can't", "hardly"}

MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


# ---------------------------------------------------------------------------
# 5. Temporal Query Parser (Deterministic Date Window Extraction)
# ---------------------------------------------------------------------------

def parse_temporal_filter(
    query: str, today: Optional[date] = None
) -> Optional[Dict[str, str]]:
    """
    Extract an explicit or relative date window (YYYY-MM-DD..YYYY-MM-DD) from
    natural language queries like 'April 2023', 'in 2024', 'last month', '2 years ago'.
    """
    ref = today or date.today()
    q = query.lower()

    # 1. Explicit Month + Year: e.g. "april 2023", "oct, 2024"
    for m_name, m_num in MONTH_MAP.items():
        m = re.search(rf"\b{m_name}\b[\s,]*(20\d{{2}})\b", q)
        if m:
            yr = int(m.group(1))
            return {
                "start_date": f"{yr:04d}-{m_num:02d}-01",
                "end_date": f"{yr:04d}-{m_num:02d}-31",
                "label": f"{m_name.title()} {yr}",
            }

    # 2. Relative N years/months/weeks/days ago
    m_ago = re.search(r"\b(a|an|one|two|three|\d+)\s+(year|years|month|months|week|weeks)\s+ago\b", q)
    if m_ago:
        num_word = m_ago.group(1)
        word_nums = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3}
        n = word_nums.get(num_word) or int(num_word)
        unit = m_ago.group(2)
        if unit.startswith("year"):
            center = date(max(2000, ref.year - n), ref.month, min(ref.day, 28))
            start = center - timedelta(days=45)
            end = center + timedelta(days=45)
            return {
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "label": f"Around {center.strftime('%b %Y')}",
            }
        if unit.startswith("month"):
            total_m = ref.year * 12 + (ref.month - 1) - n
            yr, mo = divmod(total_m, 12)
            mo += 1
            return {
                "start_date": f"{yr:04d}-{mo:02d}-01",
                "end_date": f"{yr:04d}-{mo:02d}-31",
                "label": f"{date(yr, mo, 1).strftime('%B %Y')}",
            }
        if unit.startswith("week"):
            center = ref - timedelta(days=7 * n)
            return {
                "start_date": (center - timedelta(days=4)).isoformat(),
                "end_date": (center + timedelta(days=4)).isoformat(),
                "label": f"Week of {center.isoformat()}",
            }

    # 3. Relative named windows
    if "yesterday" in q:
        d = (ref - timedelta(days=1)).isoformat()
        return {"start_date": d, "end_date": d, "label": "Yesterday"}
    if "this week" in q or "last few days" in q:
        return {
            "start_date": (ref - timedelta(days=7)).isoformat(),
            "end_date": ref.isoformat(),
            "label": "Past 7 days",
        }
    if "last week" in q:
        return {
            "start_date": (ref - timedelta(days=14)).isoformat(),
            "end_date": (ref - timedelta(days=5)).isoformat(),
            "label": "Last week",
        }
    if "this month" in q:
        return {
            "start_date": f"{ref.year:04d}-{ref.month:02d}-01",
            "end_date": f"{ref.year:04d}-{ref.month:02d}-31",
            "label": ref.strftime("%B %Y"),
        }
    if "last month" in q:
        total_m = ref.year * 12 + (ref.month - 1) - 1
        yr, mo = divmod(total_m, 12)
        mo += 1
        return {
            "start_date": f"{yr:04d}-{mo:02d}-01",
            "end_date": f"{yr:04d}-{mo:02d}-31",
            "label": date(yr, mo, 1).strftime("%B %Y"),
        }
    if "last year" in q:
        yr = ref.year - 1
        return {
            "start_date": f"{yr:04d}-01-01",
            "end_date": f"{yr:04d}-12-31",
            "label": str(yr),
        }
    if "this year" in q:
        return {
            "start_date": f"{ref.year:04d}-01-01",
            "end_date": f"{ref.year:04d}-12-31",
            "label": str(ref.year),
        }

    # 4. Explicit standalone year: e.g. "in 2023", "during 2024", "back in 2023"
    m_yr = re.search(r"\b(?:in|during|from|around|back in)\s+(20\d{2})\b", q)
    if m_yr:
        yr = int(m_yr.group(1))
        return {
            "start_date": f"{yr:04d}-01-01",
            "end_date": f"{yr:04d}-12-31",
            "label": str(yr),
        }

    # 5. Standalone month name without year -> assume most recent occurrence of that month
    for m_name, m_num in MONTH_MAP.items():
        if re.search(rf"\bin\s+{m_name}\b", q):
            yr = ref.year if m_num <= ref.month else ref.year - 1
            return {
                "start_date": f"{yr:04d}-{m_num:02d}-01",
                "end_date": f"{yr:04d}-{m_num:02d}-31",
                "label": f"{m_name.title()} {yr}",
            }

    return None


# ---------------------------------------------------------------------------
# 4. "On This Day" Memory Resurfacer (Zero LLM)
# ---------------------------------------------------------------------------

def get_on_this_day(
    db_path: Path = DB_PATH,
    journal_folders: Optional[List[str]] = None,
    today: Optional[date] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Find vault notes written on this calendar day (or ±3 days) in previous years."""
    ref = today or date.today()
    jfolders = journal_folders or []

    with db_session(db_path) as conn:
        rows = conn.execute(
            """
            SELECT n.path, n.title, n.folder, n.note_date, n.preview, n.word_count
            FROM notes n
            WHERE n.note_date IS NOT NULL
              AND length(n.note_date) >= 10
              AND substr(n.note_date, 1, 4) < ?
            ORDER BY n.note_date DESC
            """,
            (f"{ref.year:04d}",),
        ).fetchall()

    candidates: List[Tuple[int, int, Dict[str, Any]]] = []
    for r in rows:
        try:
            nd = date.fromisoformat(r["note_date"][:10])
        except ValueError:
            continue
        # Compare month/day distance safely across leap years
        try:
            anniv = date(ref.year, nd.month, nd.day)
        except ValueError:
            anniv = date(ref.year, 2, 28)
        diff = abs((anniv - ref).days)
        if diff > 3:
            continue
        if (r["word_count"] or 0) < 10:
            continue

        is_j = is_in_folders(r["path"], jfolders)
        priority = 0 if is_j else (1 if r["folder"] in ("Personal", "People") else 2)
        years_ago = ref.year - nd.year
        candidates.append(
            (
                diff,
                priority,
                {
                    "note_path": r["path"],
                    "title": r["title"],
                    "folder": r["folder"],
                    "note_date": r["note_date"],
                    "years_ago": years_ago,
                    "exact_day": diff == 0,
                    "preview": r["preview"],
                    "word_count": r["word_count"],
                },
            )
        )

    candidates.sort(key=lambda x: (x[0], x[1], -x[2]["years_ago"]))
    return [c[2] for c in candidates[:limit]]


# ---------------------------------------------------------------------------
# 3. Sentiment / Valence & Monthly TF-IDF Topic Pulse
# ---------------------------------------------------------------------------

def score_valence(text: str) -> float:
    """Compute lexicon-based emotional valence in [-1.0, +1.0] with negation handling."""
    tokens = [w.lower() for w in WORD_RE.findall(text)]
    if not tokens:
        return 0.0
    pos = 0.0
    neg = 0.0
    for i, tok in enumerate(tokens):
        negated = any(tokens[j] in NEGATORS for j in range(max(0, i - 3), i))
        if tok in POS_LEXICON:
            if negated:
                neg += 0.8
            else:
                pos += 1.0
        elif tok in NEG_LEXICON:
            if negated:
                pos += 0.5
            else:
                neg += 1.0
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / (total + 1.5), 3)


def _extract_terms(text: str) -> List[str]:
    return [
        w.lower()
        for w in WORD_RE.findall(text)
        if w.lower() not in ANALYTICS_STOPWORDS and len(w) >= 3
    ]


def compute_timeline_pulse(
    db_path: Path = DB_PATH,
    journal_folders: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Compute monthly writing frequency, word count, sentiment score, and distinctive TF-IDF keywords."""
    jfolders = journal_folders or []

    with db_session(db_path) as conn:
        rows = conn.execute(
            """
            SELECT n.path, n.folder, n.note_date, n.word_count,
                   GROUP_CONCAT(c.content, ' ') AS full_text
            FROM notes n
            JOIN chunks c ON c.note_path = n.path
            WHERE n.note_date IS NOT NULL AND length(n.note_date) >= 7
            GROUP BY n.path
            ORDER BY n.note_date ASC
            """
        ).fetchall()

    # Prefer journal folders if configured & populated, else Personal/People, else all
    journal_rows = [r for r in rows if is_in_folders(r["path"], jfolders)]
    if len(journal_rows) >= 5:
        target_rows = journal_rows
    else:
        personal_rows = [r for r in rows if r["folder"] in ("Personal", "People")]
        target_rows = personal_rows if len(personal_rows) >= 5 else rows

    by_month: Dict[str, List[sqlite3.Row]] = defaultdict(list)  # type: ignore[name-defined]
    for r in target_rows:
        ym = r["note_date"][:7]
        if re.match(r"^20\d{2}-\d{2}$", ym):
            by_month[ym].append(r)

    if not by_month:
        return []

    month_tf: Dict[str, Counter] = {}
    doc_freq: Counter = Counter()
    for ym, mrows in by_month.items():
        tf: Counter = Counter()
        for r in mrows:
            tf.update(_extract_terms(r["full_text"] or ""))
        month_tf[ym] = tf
        for term in tf.keys():
            doc_freq[term] += 1

    num_months = max(1, len(by_month))
    timeline: List[Dict[str, Any]] = []
    for ym in sorted(by_month.keys()):
        mrows = by_month[ym]
        words_total = sum(int(r["word_count"] or 0) for r in mrows)
        valences = [score_valence(r["full_text"] or "") for r in mrows]
        avg_val = round(sum(valences) / len(valences), 3) if valences else 0.0

        tf = month_tf[ym]
        total_terms = max(1, sum(tf.values()))
        scored_terms = []
        for term, cnt in tf.items():
            if cnt < 2 and len(mrows) > 1:
                continue
            idf = math.log((num_months + 1) / (doc_freq[term] + 0.5)) + 1.0
            scored_terms.append((term, (cnt / total_terms) * idf))
        scored_terms.sort(key=lambda x: x[1], reverse=True)

        timeline.append(
            {
                "month": ym,
                "note_count": len(mrows),
                "word_count": words_total,
                "sentiment": avg_val,
                "top_terms": [t[0] for t in scored_terms[:5]],
            }
        )

    return timeline


# ---------------------------------------------------------------------------
# 2. Vault Graph PageRank & Unlinked People Relationship Tracker
# ---------------------------------------------------------------------------

def compute_graph_and_relationships(
    db_path: Path = DB_PATH,
    journal_folders: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute wikilink PageRank centrality and track People/ mentions across all notes."""
    jfolders = journal_folders or []
    with db_session(db_path) as conn:
        notes = conn.execute(
            """
            SELECT n.path, n.title, n.folder, n.note_date, n.links_json,
                   GROUP_CONCAT(c.content, ' ') AS full_text
            FROM notes n
            LEFT JOIN chunks c ON c.note_path = n.path
            GROUP BY n.path
            """
        ).fetchall()

    if not notes:
        return {"anchor_notes": [], "relationships": [], "pagerank": {}}

    title_to_path: Dict[str, str] = {}
    all_paths: List[str] = []
    for r in notes:
        all_paths.append(r["path"])
        title_to_path[r["title"].lower()] = r["path"]

    path_idx = {p: i for i, p in enumerate(all_paths)}
    n_nodes = len(all_paths)
    out_edges: List[List[int]] = [[] for _ in range(n_nodes)]
    in_degree: List[int] = [0] * n_nodes

    for r in notes:
        src = path_idx[r["path"]]
        try:
            links = json.loads(r["links_json"] or "[]")
        except Exception:
            links = []
        seen_dst = set()
        for link in links:
            dst_path = title_to_path.get(str(link).lower())
            if dst_path and dst_path != r["path"]:
                dst = path_idx[dst_path]
                if dst not in seen_dst:
                    seen_dst.add(dst)
                    out_edges[src].append(dst)
                    in_degree[dst] += 1

    # Power-iteration PageRank (20 iterations)
    pr = np.full(n_nodes, 1.0 / n_nodes, dtype=np.float64)
    damping = 0.85
    for _ in range(20):
        next_pr = np.full(n_nodes, (1.0 - damping) / n_nodes, dtype=np.float64)
        dangling_sum = sum(pr[i] for i in range(n_nodes) if not out_edges[i])
        next_pr += damping * (dangling_sum / n_nodes)
        for i in range(n_nodes):
            if out_edges[i]:
                share = damping * pr[i] / len(out_edges[i])
                for dst in out_edges[i]:
                    next_pr[dst] += share
        pr = next_pr

    pagerank_map = {all_paths[i]: float(pr[i]) for i in range(n_nodes)}
    ranked_indices = sorted(range(n_nodes), key=lambda i: (in_degree[i], pr[i]), reverse=True)
    anchor_notes = []
    for idx in ranked_indices[:10]:
        if in_degree[idx] == 0:
            break
        r = notes[idx]
        anchor_notes.append(
            {
                "note_path": r["path"],
                "title": r["title"],
                "folder": r["folder"],
                "in_links": in_degree[idx],
                "pagerank": round(float(pr[idx]), 5),
            }
        )

    # People relationship timeline (both [[wikilink]] and unlinked text mentions)
    people_notes = [r for r in notes if r["folder"] == "People"]
    people_patterns: List[Tuple[dict, re.Pattern]] = []
    for pn in people_notes:
        name = pn["title"].strip()
        if len(name) < 3:
            continue
        first = name.split()[0] if " " in name else ""
        # Match full name, or distinctive first name (>= 4 chars)
        if first and len(first) >= 4:
            pat = re.compile(
                rf"\b(?:{re.escape(name)}|{re.escape(first)})\b", re.IGNORECASE
            )
        else:
            pat = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
        people_patterns.append(({"path": pn["path"], "title": name}, pat))

    rel_stats: Dict[str, Dict[str, Any]] = {}
    for pinfo, _ in people_patterns:
        rel_stats[pinfo["path"]] = {
            "name": pinfo["title"],
            "note_path": pinfo["path"],
            "mentions": 0,
            "journal_mentions": 0,
            "linked_mentions": 0,
            "first_seen": None,
            "last_seen": None,
            "valences": [],
        }

    for r in notes:
        if r["folder"] == "People":
            continue
        text = r["full_text"] or ""
        if not text:
            continue
        try:
            links_lower = {str(l).lower() for l in json.loads(r["links_json"] or "[]")}
        except Exception:
            links_lower = set()
        is_j = is_in_folders(r["path"], jfolders) or r["folder"] == "Personal"
        ndate = r["note_date"]
        val = None

        for pinfo, pat in people_patterns:
            linked = pinfo["title"].lower() in links_lower
            matched = linked or bool(pat.search(text))
            if not matched:
                continue
            st = rel_stats[pinfo["path"]]
            st["mentions"] += 1
            if is_j:
                st["journal_mentions"] += 1
            if linked:
                st["linked_mentions"] += 1
            if ndate:
                if not st["first_seen"] or ndate < st["first_seen"]:
                    st["first_seen"] = ndate
                if not st["last_seen"] or ndate > st["last_seen"]:
                    st["last_seen"] = ndate
            if val is None:
                val = score_valence(text)
            st["valences"].append(val)

    relationships = []
    for st in rel_stats.values():
        if st["mentions"] == 0:
            continue
        vals = st.pop("valences")
        st["avg_sentiment"] = round(sum(vals) / len(vals), 3) if vals else 0.0
        relationships.append(st)

    relationships.sort(
        key=lambda x: (x["journal_mentions"], x["mentions"], x["linked_mentions"]),
        reverse=True,
    )

    return {
        "anchor_notes": anchor_notes,
        "relationships": relationships[:20],
        "pagerank": pagerank_map,
    }


# ---------------------------------------------------------------------------
# 1. Spherical K-Means Clustering on Existing Note Embeddings (Pure NumPy)
# ---------------------------------------------------------------------------

def cluster_vault_themes(
    db_path: Path = DB_PATH,
    journal_folders: Optional[List[str]] = None,
    max_clusters: int = 7,
) -> List[Dict[str, Any]]:
    """Cluster chunk embeddings with spherical K-Means and label each life theme with TF-IDF."""
    jfolders = journal_folders or []
    with db_session(db_path) as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.note_path, c.content, c.embedding, c.embedding_dim,
                   n.title, n.folder, n.note_date
            FROM chunks c
            JOIN notes n ON n.path = c.note_path
            WHERE c.embedding IS NOT NULL AND c.embedding_dim > 0
            ORDER BY c.id ASC
            """
        ).fetchall()

    if len(rows) < 6:
        return []

    # Focus clustering on personal/journal/people/media/inbox chunks if plentiful
    personal_rows = [
        r for r in rows
        if is_in_folders(r["note_path"], jfolders)
        or r["folder"] in ("Personal", "People", "Media", "Inbox", "DUHABP")
    ]
    pool = personal_rows if len(personal_rows) >= 12 else rows

    dim = pool[-1]["embedding_dim"]
    valid_rows = []
    vecs: List[np.ndarray] = []
    for r in pool:
        if r["embedding_dim"] != dim:
            continue
        arr = np.frombuffer(r["embedding"], dtype=np.float32)
        if arr.shape[0] == dim:
            norm = np.linalg.norm(arr)
            if norm > 0:
                vecs.append(arr / norm)
                valid_rows.append(r)

    n = len(vecs)
    if n < 6:
        return []

    k = min(max_clusters, max(2, n // 8))
    mat = np.vstack(vecs)  # (n, dim), already unit-normalized

    # Deterministic k-means++ initialization on cosine distance
    rng = np.random.default_rng(42)
    centroids = np.zeros((k, dim), dtype=np.float32)
    centroids[0] = mat[0]
    for c_idx in range(1, k):
        sims = mat @ centroids[:c_idx].T
        best_sim = sims.max(axis=1)
        dists = np.maximum(0.0, 1.0 - best_sim)
        total = float(dists.sum())
        if total <= 1e-8:
            centroids[c_idx] = mat[c_idx % n]
        else:
            probs = dists / total
            choice = int(rng.choice(n, p=probs))
            centroids[c_idx] = mat[choice]

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(15):
        sims = mat @ centroids.T
        new_labels = sims.argmax(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c_idx in range(k):
            mask = labels == c_idx
            if mask.any():
                mean_vec = mat[mask].mean(axis=0)
                norm = np.linalg.norm(mean_vec)
                if norm > 0:
                    centroids[c_idx] = mean_vec / norm

    # TF-IDF labeling per cluster
    cluster_tf: Dict[int, Counter] = defaultdict(Counter)
    cluster_df: Counter = Counter()
    cluster_indices: Dict[int, List[int]] = defaultdict(list)

    for idx, c_idx in enumerate(labels):
        ci = int(c_idx)
        cluster_indices[ci].append(idx)
        terms = _extract_terms(valid_rows[idx]["content"])
        cluster_tf[ci].update(terms)

    for ci, tf in cluster_tf.items():
        for t in tf.keys():
            cluster_df[t] += 1

    clusters_out: List[Dict[str, Any]] = []
    final_sims = mat @ centroids.T

    for ci, idxs in cluster_indices.items():
        if len(idxs) < 2:
            continue
        tf = cluster_tf[ci]
        total_terms = max(1, sum(tf.values()))
        scored = []
        for t, cnt in tf.items():
            if cnt < 2:
                continue
            idf = math.log((k + 1) / (cluster_df[t] + 0.5)) + 1.0
            scored.append((t, (cnt / total_terms) * idf))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_terms = [t[0] for t in scored[:5]]

        # Pick representative unique notes closest to centroid
        ordered = sorted(idxs, key=lambda i: float(final_sims[i, ci]), reverse=True)
        reps = []
        seen_paths = set()
        dates = []
        for i in idxs:
            if valid_rows[i]["note_date"]:
                dates.append(valid_rows[i]["note_date"])
        for i in ordered:
            r = valid_rows[i]
            if r["note_path"] in seen_paths:
                continue
            seen_paths.add(r["note_path"])
            reps.append(
                {
                    "title": r["title"],
                    "note_path": r["note_path"],
                    "folder": r["folder"],
                    "note_date": r["note_date"],
                }
            )
            if len(reps) >= 4:
                break

        unique_notes = len({valid_rows[i]["note_path"] for i in idxs})
        clusters_out.append(
            {
                "cluster_id": ci,
                "label": ", ".join(t.title() for t in top_terms[:3]) or f"Cluster {ci + 1}",
                "keywords": top_terms,
                "chunk_count": len(idxs),
                "note_count": unique_notes,
                "date_span": f"{min(dates)} → {max(dates)}" if dates else "",
                "representative_notes": reps,
            }
        )

    clusters_out.sort(key=lambda c: c["note_count"], reverse=True)
    return clusters_out


def compute_vault_analytics(
    db_path: Path = DB_PATH,
    journal_folders: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run all classic ML, graph, timeline, and On This Day analytics in one pass."""
    graph_rel = compute_graph_and_relationships(db_path, journal_folders)
    timeline = compute_timeline_pulse(db_path, journal_folders)
    clusters = cluster_vault_themes(db_path, journal_folders)
    on_this_day = get_on_this_day(db_path, journal_folders)

    return {
        "clusters": clusters,
        "relationships": graph_rel["relationships"],
        "anchor_notes": graph_rel["anchor_notes"],
        "timeline": timeline,
        "on_this_day": on_this_day,
    }
