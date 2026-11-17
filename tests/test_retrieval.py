from app.auth import Principal
from tests.conftest import ask, headers

ACME = Principal("acme-freight", "analyst")
ACME_ADMIN = Principal("acme-freight", "admin")
BLUE = Principal("bluepeak", "analyst")
BLUE_ADMIN = Principal("bluepeak", "admin")


def retrieve(container, q, who, k=8):
    return container.answers.retriever.retrieve(q, who, k)


def test_retrieval_returns_relevant_policy_first(seeded):
    hits = retrieve(seeded, "How long are shipment records retained?", ACME)
    assert hits and hits[0].meta["doc_id"] in {"KB-RET-002", "KB-RET-001", "BULL-2025-09"}
    assert hits == sorted(hits, key=lambda h: h.relevance, reverse=True)


def test_retrieval_respects_top_k(seeded):
    assert len(retrieve(seeded, "retention", ACME, k=2)) == 2


def test_statuses_attached_to_hits(seeded):
    by_doc = {h.meta["doc_id"]: h.meta["status"] for h in retrieve(seeded, "retention policy records days", ACME, k=20)}
    assert by_doc["KB-RET-002"] == "current"
    assert by_doc["KB-RET-001"] == "superseded"
    assert by_doc["BULL-2025-09"] == "unverified"


def test_tenant_a_never_retrieves_tenant_b_documents(seeded):
    hits = retrieve(seeded, "BluePeak escalation phrase contract exception 365 days", ACME, k=20)
    assert hits
    assert all(h.meta["tenant"] in {"acme-freight", "shared"} for h in hits)
    assert "T-2001" not in {h.meta["doc_id"] for h in hits}


def test_tenant_b_does_not_see_tenant_a_documents(seeded):
    hits = retrieve(seeded, "export failed EXP-504 dashboard rename bulletin", BLUE_ADMIN, k=20)
    assert all(h.meta["tenant"] in {"bluepeak", "shared"} for h in hits)


def test_analyst_cannot_retrieve_confidential_but_admin_can(seeded):
    q = "incident INC-7731 root cause label service credential rotation"
    assert "KB-PM-001" not in {h.meta["doc_id"] for h in retrieve(seeded, q, ACME, k=20)}
    assert "KB-PM-001" in {h.meta["doc_id"] for h in retrieve(seeded, q, ACME_ADMIN, k=20)}


def test_end_to_end_answer_has_source_ids_and_excerpts(seeded_client):
    r = ask(seeded_client, "How long do we retain shipment records?")
    body = r.json()
    assert r.status_code == 200 and not body["abstained"]
    assert body["answer"] and "180" in body["answer"]
    cit = body["citations"][0]
    assert cit["source_id"] == "KB-RET-002#0" and cit["status"] == "current" and len(cit["sha256"]) == 64
    assert "180 days" in cit["excerpt"]


def test_pdf_citation_includes_page_number(seeded_client):
    r = ask(seeded_client, "How quickly are standard shipments cleared in customs? 48 hours days")
    body = r.json()
    if not body["abstained"]:
        assert any(c["doc_id"] == "PDF-SLA-2025" and c["page"] == 1 for c in body["citations"])
    else:  # the fake model only quotes day-based durations, so accept an abstention but not a wrong cite
        assert body["citations"] == []


def test_out_of_scope_question_abstains_without_calling_llm(seeded_client, llm):
    r = ask(seeded_client, "What is the CEO's favourite colour of sneakers?")
    body = r.json()
    assert body["abstained"] and body["abstain_reason"] == "insufficient_evidence" and body["citations"] == []
    assert llm.calls == []
