import logging
import re
import uuid
from dataclasses import dataclass

from app.auth import Principal
from app.rag.citations import parse_model_json, verify_citations
from app.rag.conflicts import Claim, extract_retention_claim
from app.rag.prompt import SYSTEM_PROMPT, PromptSource, build_user_prompt, shown_text
from app.retrieval.retriever import Retriever
from app.retrieval.store import Hit
from app.schemas import AskResponse, Citation, ConflictInfo, SourceNote, SourceStatus
from app.security import policy
from app.security.injection_detector import detect, redact

log = logging.getLogger(__name__)
MAX_EVIDENCE = 5


@dataclass
class Evidence:
    hit: Hit
    status: SourceStatus
    shown: str  # exactly what the model sees; citations are verified against this
    claim: Claim | None

    @property
    def topic(self) -> str:
        return self.hit.meta.get("topic", "")


def _note(hit: Hit, reason: str, excerpt: str | None = None) -> SourceNote:
    m = hit.meta
    return SourceNote(
        source_id=hit.chunk_id, doc_id=m["doc_id"], title=m["title"], status=SourceStatus(m["status"]),
        version=m["version"], reason=reason, excerpt=(excerpt or hit.text)[:300],
    )


def _citation(hit: Hit, status: SourceStatus, excerpt: str) -> Citation:
    m = hit.meta
    return Citation(
        source_id=hit.chunk_id, doc_id=m["doc_id"], title=m["title"], page=m["page"] if m["page"] > 0 else None,
        excerpt=excerpt, status=status, version=m["version"], sha256=m["sha256"],
    )


class AnswerService:
    def __init__(self, retriever: Retriever, llm, model_name: str, min_relevance: float, default_top_k: int):
        self.retriever = retriever
        self.llm = llm
        self.model_name = model_name
        self.min_relevance = min_relevance
        self.default_top_k = default_top_k

    def _abstain(self, rid: str, reason: str, **extra) -> AskResponse:
        return AskResponse(request_id=rid, answer=None, abstained=True, abstain_reason=reason, model=self.model_name, **extra)

    def ask(self, question: str, principal: Principal, top_k: int | None = None) -> AskResponse:
        rid = uuid.uuid4().hex[:12]
        hits = [h for h in self.retriever.retrieve(question, principal, top_k or self.default_top_k)
                if h.relevance >= self.min_relevance]
        if not hits:
            return self._abstain(rid, "insufficient_evidence")

        stale: list[SourceNote] = []
        flagged: list[SourceNote] = []
        candidates: list[Evidence] = []
        for h in hits:
            status = SourceStatus(h.meta["status"])
            det = detect(h.text)
            if status == SourceStatus.superseded:
                claim = extract_retention_claim(h.text)
                stale.append(_note(h, "superseded by a newer official version", claim.sentence if claim else None))
                continue
            if det.flagged and status == SourceStatus.unverified:
                flagged.append(_note(h, f"quarantined: prompt-injection indicators ({', '.join(det.flags)})",
                                     " | ".join(det.flagged_lines)))
                continue
            text = h.text
            if det.flagged:  # official source with instruction-like lines: keep it, redact those lines
                text = redact(text)
                flagged.append(_note(h, f"instruction-like lines redacted ({', '.join(det.flags)})",
                                     " | ".join(det.flagged_lines)))
            candidates.append(Evidence(h, status, shown_text(text), extract_retention_claim(text)))

        # Unverified claims never override official ones: a *differing* unverified claim is reported as a conflict.
        official_claims: dict[str, set[int]] = {}
        for e in candidates:
            if e.status == SourceStatus.current and e.topic and e.claim:
                official_claims.setdefault(e.topic, set()).add(e.claim.days)
        conflicts: list[ConflictInfo] = []
        evidence: list[Evidence] = []
        for e in candidates:
            if (e.status == SourceStatus.unverified and e.claim and e.topic in official_claims
                    and e.claim.days not in official_claims[e.topic]):
                official = [o for o in candidates if o.status == SourceStatus.current and o.topic == e.topic and o.claim]
                conflicts.append(ConflictInfo(
                    topic=e.topic,
                    values=[
                        {"source_id": e.hit.chunk_id, "status": "unverified", "claim": e.claim.sentence},
                        *[{"source_id": o.hit.chunk_id, "status": "current", "claim": o.claim.sentence} for o in official],
                    ],
                    resolution="ignored: unverified source cannot override an official one",
                ))
            else:
                evidence.append(e)
        evidence = evidence[:MAX_EVIDENCE]
        common = dict(stale_sources=stale, flagged_sources=flagged, conflicts=conflicts)
        if not evidence:
            return self._abstain(rid, "no_trusted_evidence", **common)

        # Equal-authority disagreement: refuse to pick a winner.
        by_topic: dict[str, list[Evidence]] = {}
        for e in evidence:
            if e.topic and e.claim and e.status == SourceStatus.current:
                by_topic.setdefault(e.topic, []).append(e)
        for topic, group in by_topic.items():
            if len({e.claim.days for e in group}) > 1:
                conflict = ConflictInfo(
                    topic=topic,
                    values=[{"source_id": e.hit.chunk_id, "status": e.status.value, "claim": e.claim.sentence} for e in group],
                    resolution="unresolved: official sources contradict each other",
                )
                return AskResponse(
                    request_id=rid, answer=None, abstained=True, abstain_reason="conflicting_evidence", conflict=True,
                    citations=[_citation(e.hit, e.status, e.claim.sentence) for e in group],
                    stale_sources=stale, flagged_sources=flagged, conflicts=[*conflicts, conflict], model=self.model_name,
                )

        sources = [PromptSource(e.hit.chunk_id, e.status.value, "official" if e.status == SourceStatus.current else "unverified", e.shown)
                   for e in evidence]
        parsed = None
        for _ in range(2):  # one retry on malformed output
            parsed = parse_model_json(self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, sources)))
            if parsed is not None:
                break
        if parsed is None:
            return self._abstain(rid, "invalid_model_output", **common)
        answer = parsed.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return self._abstain(rid, "insufficient_evidence", **common)

        by_id = {e.hit.chunk_id: e for e in evidence}
        verified = verify_citations(parsed.get("citations"), {k: v.shown for k, v in by_id.items()})
        if not verified:
            return self._abstain(rid, "no_verifiable_citation", **common)

        allowed = " ".join(e.shown for e in evidence)
        foreign = [t for t, m in self.retriever.store.all_chunks() if not policy.can_read(principal, m)]
        if policy.contains_foreign_content(answer, foreign, allowed) or policy.contains_canary(answer, allowed):
            log.warning("output policy violation blocked (request %s)", rid)
            return self._abstain(rid, "output_policy_violation", **common)
        # An answer may only state numbers that appear in its own verified citations (or in the question).
        quotes = " ".join(v.excerpt for v in verified)
        if set(re.findall(r"\d+", answer)) - set(re.findall(r"\d+", quotes + " " + question)):
            return self._abstain(rid, "answer_not_supported_by_citations", **common)

        return AskResponse(
            request_id=rid, answer=answer.strip(), abstained=False,
            citations=[_citation(by_id[v.source_id].hit, by_id[v.source_id].status, v.excerpt) for v in verified],
            model=self.model_name, **common,
        )
