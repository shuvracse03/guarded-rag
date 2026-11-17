"""Tenant and permission enforcement. Runs in application code, never delegated to the prompt."""
import re
from dataclasses import dataclass

from app.auth import Principal

SHARED_TENANT = "shared"


@dataclass(frozen=True)
class ChunkMeta:
    tenant: str
    classification: str


def can_read(principal: Principal, meta: dict) -> bool:
    tenant = meta.get("tenant")
    if tenant != principal.tenant and tenant != SHARED_TENANT:
        return False
    return meta.get("classification") in principal.clearance


def allowed_tenants(principal: Principal) -> list[str]:
    return [principal.tenant, SHARED_TENANT]


def filter_readable(principal: Principal, hits: list) -> list:
    """Second, independent check after the store-level tenant filter (defence in depth)."""
    return [h for h in hits if can_read(principal, h.meta)]


_WORD = re.compile(r"[a-z0-9\-]+")


def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def contains_foreign_content(answer: str, foreign_texts: list[str], allowed_text: str = "") -> bool:
    """Output guard: does the model answer reproduce text from documents this principal may not read?

    Shingles that also occur in readable evidence are ignored, so shared boilerplate cannot cause false alarms.
    """
    ans = _shingles(answer, 8) - _shingles(allowed_text, 8)
    if not ans:
        return False
    return any(ans & _shingles(t, 8) for t in foreign_texts)


# Distinctive tokens (e.g. canary markers) that must never be echoed.
CANARY = re.compile(r"[A-Z0-9]+-CANARY-[A-Za-z0-9]+")


def contains_canary(text: str, allowed_text: str = "") -> bool:
    return any(m not in allowed_text for m in CANARY.findall(text))
