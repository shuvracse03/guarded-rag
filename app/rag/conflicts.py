"""Rule-based claim extraction used to detect conflicting evidence without trusting the model."""
import re
from dataclasses import dataclass

_UNIT_DAYS = {"day": 1, "month": 30, "year": 365}
_CLAIM = re.compile(
    r"(?:retain(?:ed|s)?|retention|kept|keep|stored?|deleted? after|purged? after)[^.\n]{0,80}?"
    r"\b(\d{1,4})[\s-]*(day|month|year)s?\b",
    re.I,
)


@dataclass(frozen=True)
class Claim:
    days: int
    sentence: str


def extract_retention_claim(text: str) -> Claim | None:
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        m = _CLAIM.search(sentence)
        if m:
            return Claim(days=int(m.group(1)) * _UNIT_DAYS[m.group(2).lower()], sentence=sentence.strip())
    return None
