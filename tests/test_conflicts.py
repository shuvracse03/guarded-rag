from app.rag.conflicts import extract_retention_claim
from tests.conftest import ask, doc_payload, headers

Q = "How long do we retain shipment records?"


def test_claim_extraction_units():
    assert extract_retention_claim("Records must be retained for 180 days after delivery.").days == 180
    assert extract_retention_claim("Export files are kept for 2 weeks and 14 days").days == 14
    assert extract_retention_claim("Logs stored for 6 months.").days == 180
    assert extract_retention_claim("Nothing about duration here.") is None


def test_stale_version_is_reported_not_used(seeded_client):
    body = ask(seeded_client, Q).json()
    assert "180" in body["answer"] and "90" not in body["answer"]
    stale = {s["doc_id"]: s for s in body["stale_sources"]}
    assert "KB-RET-001" in stale and stale["KB-RET-001"]["status"] == "superseded"
    assert "90 days" in stale["KB-RET-001"]["excerpt"]
    assert all(c["doc_id"] != "KB-RET-001" for c in body["citations"])


def test_newer_unverified_bulletin_does_not_override_official_policy(seeded_client):
    body = ask(seeded_client, Q).json()
    assert "30" not in body["answer"] and "7 days" not in body["answer"]
    conflict = next(c for c in body["conflicts"] if c["topic"] == "record_retention")
    assert "ignored" in conflict["resolution"]
    ids = {v["source_id"]: v["status"] for v in conflict["values"]}
    assert ids["BULL-2025-09#0"] == "unverified" and ids["KB-RET-002#0"] == "current"


def test_bulletin_claiming_version_3_is_still_unverified(seeded):
    r = seeded.answers.retriever.retrieve(Q, __import__("app.auth", fromlist=["Principal"]).Principal("acme-freight", "analyst"), 10)
    status = {h.meta["doc_id"]: h.meta["status"] for h in r}
    assert status["BULL-2025-09"] == "unverified" and status["KB-RET-002"] == "current"


def test_equal_authority_conflict_abstains_and_lists_both(seeded_client, llm):
    body = ask(seeded_client, "How long are export files kept?").json()
    assert body["conflict"] is True and body["abstained"] is True and body["answer"] is None
    assert body["abstain_reason"] == "conflicting_evidence"
    claims = " ".join(v["claim"] for v in body["conflicts"][-1]["values"])
    assert "7 days" in claims and "14 days" in claims
    assert {c["doc_id"] for c in body["citations"]} == {"KB-EXP-001", "KB-FAQ-001"}
    assert llm.calls == []  # decided by code, the model is never asked to pick a winner


def test_new_official_version_supersedes_previous_current(seeded_client):
    payload = doc_payload(
        doc_id="KB-RET-003", doc_type="kb", title="Shipment Record Retention Policy v3",
        text="Shipment records must be retained for 365 days after the delivery date.",
        version=3, effective_date="2025-12-01", doc_family="retention-policy", topic="record_retention",
        authority="official", classification="public", shared=True,
    )
    assert seeded_client.post("/ingest", json=payload, headers=headers("acme_admin")).status_code == 201
    body = ask(seeded_client, Q).json()
    assert "365" in body["answer"]
    stale_ids = {s["doc_id"] for s in body["stale_sources"]}
    assert {"KB-RET-001", "KB-RET-002"} <= stale_ids


def test_analyst_cannot_publish_a_newer_policy(seeded_client):
    payload = doc_payload(
        doc_id="KB-RET-066", doc_type="kb", title="Retention v66", text="Shipment records must be retained for 1 days.",
        version=66, effective_date="2030-01-01", doc_family="retention-policy", topic="record_retention", authority="official",
    )
    assert seeded_client.post("/ingest", json=payload, headers=headers("acme")).status_code == 403
    assert "180" in ask(seeded_client, Q).json()["answer"]


def test_unverified_ticket_with_same_claim_is_corroboration_not_conflict(seeded_client):
    seeded_client.post("/ingest", json=doc_payload(
        doc_id="T-5001", title="Retention question", topic="record_retention",
        text="Customer note: records are retained for 180 days per policy."), headers=headers("acme"))
    body = ask(seeded_client, Q).json()
    assert not body["abstained"]
    assert not any(v["source_id"].startswith("T-5001") for c in body["conflicts"] for v in c["values"])
