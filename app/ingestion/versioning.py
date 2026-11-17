"""Stale/conflicting version handling.

Rules (enforced in code, never by the model):
  * Only documents registered with authority=official can be `current` or `superseded`.
  * Within a doc_family, the official document with the latest (effective_date, version) is `current`;
    older official ones are `superseded` (stale).
  * A document with authority=unverified is never allowed to supersede anything, however new it is
    or whatever it claims about itself.
"""
from datetime import date

from app.schemas import SourceStatus


def _key(meta: dict) -> tuple[date, int]:
    eff = meta.get("effective_date") or ""
    return (date.fromisoformat(eff) if eff else date.min, int(meta.get("version", 1)))


def resolve_statuses(docs: list[dict]) -> dict[str, SourceStatus]:
    statuses: dict[str, SourceStatus] = {}
    families: dict[str, list[dict]] = {}
    for d in docs:
        if d["authority"] != "official":
            statuses[d["doc_id"]] = SourceStatus.unverified
        elif d.get("doc_family"):
            families.setdefault(d["doc_family"], []).append(d)
        else:
            statuses[d["doc_id"]] = SourceStatus.current
    for members in families.values():
        newest = max(members, key=_key)
        for d in members:
            statuses[d["doc_id"]] = SourceStatus.current if d is newest else SourceStatus.superseded
    return statuses
