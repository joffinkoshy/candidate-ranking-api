"""Split a resume into passage-level chunks for retrieval.

Chunking matters: retrieving whole resumes buries the relevant lines in noise.
We pack sentence-ish segments into a character budget with a small overlap so a
skill mentioned at a chunk boundary is still findable.
"""

import re

from app.config import settings

# Split on sentence terminators or newlines.
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _hard_split(segment: str, chunk_size: int) -> list[str]:
    """Word-window fallback for a single segment longer than the budget."""
    out, cur = [], ""
    for word in segment.split():
        if cur and len(cur) + 1 + len(word) > chunk_size:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, chunk_size: int | None = None,
               overlap: int | None = None) -> list[str]:
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap if overlap is not None else settings.CHUNK_OVERLAP

    text = (text or "").strip()
    if not text:
        return []

    segments = [s.strip() for s in _SPLIT_RE.split(text) if s.strip()]

    chunks: list[str] = []
    current = ""
    for seg in segments:
        if len(seg) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(seg, chunk_size))
            continue

        if not current:
            current = seg
        elif len(current) + 1 + len(seg) <= chunk_size:
            current = f"{current} {seg}"
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail} {seg}".strip() if tail else seg

    if current:
        chunks.append(current)
    return chunks
