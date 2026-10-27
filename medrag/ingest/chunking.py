"""Section-aware chunking with a contextual header on every chunk.

Each chunk starts with a short header (who / when / what study) so it can be understood and
retrieved on its own ("contextual retrieval"). Bodies are split on section boundaries first,
then on sentences, and packed up to `max_words` with a small sentence overlap.
"""
import re

SECTION_RE = re.compile(r"^(TECHNIQUE|COMPARISON|FINDINGS|IMPRESSION|RECOMMENDATION|HISTORY|INDICATION):", re.M)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sections(text: str) -> list[str]:
    starts = [m.start() for m in SECTION_RE.finditer(text)]
    if not starts:
        return [text.strip()] if text.strip() else []
    if starts[0] != 0:
        starts.insert(0, 0)
    bounds = starts + [len(text)]
    return [text[a:b].strip() for a, b in zip(bounds, bounds[1:]) if text[a:b].strip()]


def _words(s: str) -> int:
    return len(s.split())


def chunk_text(body: str, max_words: int = 220, overlap_sentences: int = 1) -> list[str]:
    """Pack sections (or sentences of an oversized section) into chunks of at most max_words."""
    units: list[str] = []
    for sec in split_sections(body):
        if _words(sec) <= max_words:
            units.append(sec)
        else:
            units.extend(SENTENCE_RE.split(sec))

    chunks, cur = [], []
    for u in units:
        if cur and _words(" ".join(cur + [u])) > max_words:
            chunks.append("\n".join(cur))
            cur = cur[-overlap_sentences:] if overlap_sentences and _words(cur[-1]) < max_words // 3 else []
        cur.append(u)
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def chunk_document(header: str, body: str, max_words: int = 220) -> list[str]:
    budget = max(60, max_words - _words(header))
    return [f"{header}\n{c}" for c in chunk_text(body, budget)]
