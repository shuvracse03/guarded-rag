import json

from app.rag.citations import find_quote, parse_model_json, verify_citations
from tests.conftest import FakeLLM, ask

EVIDENCE = {"KB-RET-002#0": "Shipment records must be retained for 180 days after the delivery date."}


def test_valid_citation_kept_and_excerpt_is_real_slice():
    out = verify_citations([{"source_id": "KB-RET-002#0", "quote": "retained for 180 days"}], EVIDENCE)
    assert len(out) == 1 and out[0].excerpt in EVIDENCE["KB-RET-002#0"]


def test_quote_matching_tolerates_case_and_whitespace_but_returns_original_text():
    out = verify_citations([{"source_id": "KB-RET-002#0", "quote": "RETAINED  for\n180 days"}], EVIDENCE)
    assert out and out[0].excerpt == "retained for 180 days"


def test_fabricated_source_id_is_dropped():
    assert verify_citations([{"source_id": "T-2001#0", "quote": "retained for 180 days"}], EVIDENCE) == []


def test_altered_quote_is_dropped():
    assert verify_citations([{"source_id": "KB-RET-002#0", "quote": "retained for 7 days after delivery"}], EVIDENCE) == []


def test_too_short_quote_is_dropped():
    assert find_quote("180", EVIDENCE["KB-RET-002#0"]) is None


def test_garbage_citation_shapes_are_ignored():
    assert verify_citations("not a list", EVIDENCE) == []
    assert verify_citations([None, 5, {"source_id": 3, "quote": []}, {}], EVIDENCE) == []


def test_duplicate_citations_deduplicated():
    c = {"source_id": "KB-RET-002#0", "quote": "retained for 180 days"}
    assert len(verify_citations([c, c], EVIDENCE)) == 1


def test_parse_model_json_variants():
    assert parse_model_json('{"answer": "x", "citations": []}') == {"answer": "x", "citations": []}
    assert parse_model_json('Sure! {"answer": "x"} hope this helps') == {"answer": "x"}
    assert parse_model_json("no json here") is None
    assert parse_model_json("[1,2]") is None
    assert parse_model_json(None) is None


def test_uncited_answer_is_replaced_by_abstention(app, seeded_client):
    app.state.container.llm.handler = lambda s, u: json.dumps({"answer": "Records are kept 180 days.", "citations": []})
    body = ask(seeded_client, "How long do we retain shipment records?").json()
    assert body["abstained"] and body["abstain_reason"] == "no_verifiable_citation" and body["answer"] is None


def test_hallucinated_quote_is_not_returned(app, seeded_client):
    bogus = {"answer": "Kept for 365 days.", "citations": [{"source_id": "KB-RET-002#0", "quote": "retained for 365 days after delivery"}]}
    app.state.container.llm.handler = lambda s, u: json.dumps(bogus)
    body = ask(seeded_client, "How long do we retain shipment records?").json()
    assert body["abstained"] and body["citations"] == []


def test_partially_valid_citations_keep_only_valid(app, seeded_client):
    resp = {"answer": "180 days.", "citations": [
        {"source_id": "KB-RET-002#0", "quote": "retained for 180 days after the delivery date"},
        {"source_id": "MADE-UP#9", "quote": "totally invented evidence"},
    ]}
    app.state.container.llm.handler = lambda s, u: json.dumps(resp)
    body = ask(seeded_client, "How long do we retain shipment records?").json()
    assert not body["abstained"] and [c["source_id"] for c in body["citations"]] == ["KB-RET-002#0"]
