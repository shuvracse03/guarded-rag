"""Post-generation citation verification. The model's citations are claims; code decides which are real."""
import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class VerifiedQuote:
    source_id: str
    excerpt: str  # a real slice of the source text that the model was shown


def parse_model_json(raw: str) -> dict | None:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{[\s\S]*\}", raw or "")
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def find_quote(quote: str, text: str) -> str | None:
    words = quote.strip().strip("\"'`“”").split()
    if len(" ".join(words)) < 8:
        return None
    pattern = r"\s+".join(re.escape(w) for w in words)
    m = re.search(pattern, text, flags=re.I)
    return m.group(0) if m else None


def verify_citations(raw_citations, evidence_texts: dict[str, str]) -> list[VerifiedQuote]:
    """Keep only citations whose source was actually provided AND whose quote really occurs in it."""
    verified: list[VerifiedQuote] = []
    seen: set[tuple[str, str]] = set()
    if not isinstance(raw_citations, list):
        return verified
    for c in raw_citations:
        if not isinstance(c, dict):
            continue
        sid, quote = c.get("source_id"), c.get("quote")
        if not isinstance(sid, str) or not isinstance(quote, str) or sid not in evidence_texts:
            continue
        hit = find_quote(quote, evidence_texts[sid])
        if hit and (sid, hit) not in seen:
            seen.add((sid, hit))
            verified.append(VerifiedQuote(sid, hit))
    return verified
