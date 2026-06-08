"""Regenerate citation_key on existing 01_extraction outputs deterministically (no LLM).

Format: <author_or_title_prefix>_<year>_<title_keyword>
- Strips diacritics (Roldán → roldan, Álvarez → alvarez).
- Falls back to title keywords if first author is missing/Unknown.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

EXT = Path(__file__).resolve().parent.parent / "outputs" / "01_extraction"

STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "for", "on", "in", "at", "to", "from",
    "with", "by", "is", "are", "be", "as", "into", "via", "about", "across", "after",
    "before", "between", "among", "than", "this", "these", "those", "that", "it",
    "its", "their", "his", "her", "our", "your", "we", "i", "they", "he", "she",
    "evidence", "study", "studies", "paper", "preprint", "case", "report", "analysis",
    "approach", "approaches", "investigation", "investigations", "examination",
    "impact", "impacts", "effect", "effects",
    "using", "use", "uses", "based", "new", "novel", "experimental",
}


def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _last_name(author: str) -> str:
    a = author.strip()
    if not a:
        return ""
    if "," in a:
        return _slug(a.split(",")[0])
    parts = a.split()
    if not parts:
        return ""
    return _slug(parts[-1])


def _title_keywords(title: str, n: int = 1) -> list[str]:
    toks = [t for t in re.split(r"[^A-Za-z0-9]+", title or "") if t]
    out = []
    for t in toks:
        sl = _slug(t)
        if not sl or sl in STOPWORDS:
            continue
        if len(sl) < 3:
            continue
        out.append(sl)
        if len(out) >= n:
            break
    return out


def _make_key(authors: list[str] | None, year: int | None, title: str) -> str:
    year_s = str(year) if year else "nd"
    title_tail = "_".join(_title_keywords(title, n=1))
    first_author = (authors or [""])[0] or ""
    ln = _last_name(first_author) if first_author and not first_author.lower().startswith("unknown") else ""
    if ln:
        prefix = ln
    else:
        kws = _title_keywords(title, n=2)
        prefix = "_".join(kws) if kws else "untitled"
    key = f"{prefix}_{year_s}"
    if title_tail and title_tail not in prefix:
        key = f"{key}_{title_tail}"
    if len(key) > 35:
        key = key[:35].rstrip("_")
    return key


def regenerate_all() -> dict[str, tuple[str, str]]:
    changes: dict[str, tuple[str, str]] = {}
    for p in sorted(EXT.glob("*.json")):
        if p.stem.endswith(".error"):
            continue
        d = json.loads(p.read_text())
        old = d.get("citation_key", "")
        new = _make_key(d.get("authors"), d.get("year"), d.get("title") or "")
        if new != old:
            d["citation_key"] = new
            with open(p, "w") as f:
                json.dump(d, f, indent=2, default=str)
            changes[p.stem] = (old, new)
    return changes


def main() -> None:
    changes = regenerate_all()
    print(f"Updated {len(changes)} citation_keys")
    for pid, (old, new) in sorted(changes.items()):
        print(f"  [{pid}]")
        print(f"    {old!r} → {new!r}")


if __name__ == "__main__":
    main()
