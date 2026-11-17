import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RawChunk:
    text: str
    page: int | None
    start: int
    end: int


def chunk_text(text: str, page: int | None = None, max_chars: int = 700) -> list[RawChunk]:
    """Paragraph-aware chunking that keeps character offsets so excerpts stay verifiable."""
    chunks: list[RawChunk] = []
    cur_start: int | None = None
    cur_end = 0
    for m in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", text):
        para_start, para_end = m.start(), m.end()
        is_heading = text[para_start] == "#"
        if cur_start is not None and is_heading:  # headings always start a new chunk
            chunks.append(RawChunk(text[cur_start:cur_end].strip(), page, cur_start, cur_end))
            cur_start = None
        if cur_start is None:
            cur_start, cur_end = para_start, para_end
        elif para_end - cur_start <= max_chars:
            cur_end = para_end
        else:
            chunks.append(RawChunk(text[cur_start:cur_end].strip(), page, cur_start, cur_end))
            cur_start, cur_end = para_start, para_end
        # a single over-long paragraph is split on sentence boundaries
        while cur_end - cur_start > max_chars * 1.5:
            cut = text.rfind(". ", cur_start, cur_start + max_chars)
            cut = cut + 1 if cut > cur_start else cur_start + max_chars
            chunks.append(RawChunk(text[cur_start:cut].strip(), page, cur_start, cut))
            cur_start = cut
    if cur_start is not None and text[cur_start:cur_end].strip():
        chunks.append(RawChunk(text[cur_start:cur_end].strip(), page, cur_start, cur_end))
    return [c for c in chunks if c.text]
