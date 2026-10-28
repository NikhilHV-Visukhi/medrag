"""Section-aware chunking with a contextual header on every chunk.

Each chunk starts with a short header (who / when / what study) so it can be understood and
retrieved on its own ("contextual retrieval"). Bodies are split on section boundaries first,
then on sentences, and packed up to `max_words` with a small sentence overlap.
"""
import re

# Any upper-case heading at the start of a line ("FINDINGS:", "HISTORY OF PRESENT ILLNESS:", "PLAN:").
SECTION_RE = re.compile(r"^([A-Z][A-Z0-9 /&(),'-]{2,60}):", re.M)
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


RECURSIVE_SEPARATORS = ("\n\n", "\n", ". ", "; ", ", ", " ")


def recursive_split(text: str, max_words: int = 220, overlap_words: int = 40, seps: tuple[str, ...] = RECURSIVE_SEPARATORS) -> list[str]:
    """Generic recursive splitting (as in LangChain's RecursiveCharacterTextSplitter, counted in words):
    split on the coarsest separator, split oversized pieces on the next one, then pack the pieces up to
    max_words, starting each new chunk with the last `overlap_words` words of the previous one.
    It ignores section headings, so it is the baseline for the section-aware splitter."""
    def pieces(t: str, level: int) -> list[str]:
        if _words(t) <= max_words or level == len(seps):
            return [t]
        parts = [p for p in t.split(seps[level]) if p.strip()]
        if seps[level].strip():  # keep the punctuation on the piece it ended
            parts = [p + seps[level].strip() for p in parts[:-1]] + parts[-1:]
        return [q for p in parts for q in pieces(p, level + 1)]

    chunks, cur = [], []
    for p in pieces(text.strip(), 0):
        w = p.split()
        if cur and len(cur) + len(w) > max_words:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_words:] if overlap_words else []
        cur += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_document(header: str, body: str, max_words: int = 220, strategy: str = "section", overlap_words: int = 40) -> list[str]:
    """strategy: 'section' (section-aware with sentence overlap, the default) or 'recursive' (generic, word overlap)."""
    budget = max(60, max_words - _words(header))
    parts = recursive_split(body, budget, overlap_words) if strategy == "recursive" else chunk_text(body, budget)
    return [f"{header}\n{c}" for c in parts]
