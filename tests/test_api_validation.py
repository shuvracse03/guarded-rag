import pytest

from tests.conftest import PDF_DIR, ask, doc_payload, headers


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["ollama_reachable"] is True
    assert body["documents"] == 0


def test_health_degraded_when_ollama_down(client, llm):
    llm.ping = lambda: False
    body = client.get("/health").json()
    assert body["status"] == "degraded" and body["ollama_reachable"] is False


@pytest.mark.parametrize("path,payload", [("/ingest", doc_payload()), ("/ask", {"question": "hello there"})])
def test_missing_api_key_is_401(client, path, payload):
    r = client.post(path, json=payload)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "missing_api_key"


def test_invalid_api_key_is_401(client):
    r = client.post("/ask", json={"question": "hello there"}, headers={"X-API-Key": "nope"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"


def test_ingest_ok_and_error_shape(client):
    r = client.post("/ingest", json=doc_payload(), headers=headers())
    assert r.status_code == 201
    body = r.json()
    assert body["tenant"] == "acme-freight" and body["chunks"] == 1 and len(body["sha256"]) == 64
    assert body["authority"] == "unverified"


@pytest.mark.parametrize(
    "override,field",
    [
        ({"doc_id": "x"}, "doc_id"),
        ({"doc_id": "bad id with spaces"}, "doc_id"),
        ({"doc_type": "spreadsheet"}, "doc_type"),
        ({"title": ""}, "title"),
        ({"text": "short"}, "text"),
        ({"text": " " * 50}, "text"),
        ({"text": "x" * 100_001}, "text"),
        ({"version": 0}, "version"),
        ({"effective_date": "not-a-date"}, "effective_date"),
        ({"doc_family": "Bad Family!"}, "doc_family"),
        ({"classification": "top-secret"}, "classification"),
    ],
)
def test_ingest_validation_errors(client, override, field):
    r = client.post("/ingest", json=doc_payload(**override), headers=headers())
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "validation_error"
    assert any(field in d["field"] for d in err["details"])


def test_ingest_rejects_tenant_spoofing_field(client):
    r = client.post("/ingest", json=doc_payload(tenant="bluepeak"), headers=headers())
    assert r.status_code == 422
    assert any("tenant" in d["field"] for d in r.json()["error"]["details"])


def test_malformed_json_is_422(client):
    r = client.post("/ingest", content=b"{not json", headers={**headers(), "Content-Type": "application/json"})
    assert r.status_code == 422


def test_tickets_endpoint_only_accepts_tickets(client):
    ok = client.post("/tickets", json=doc_payload(doc_id="T-9002"), headers=headers())
    assert ok.status_code == 201
    bad = client.post("/tickets", json=doc_payload(doc_id="KB-9002", doc_type="kb"), headers=headers())
    assert bad.status_code == 422


def test_non_admin_cannot_register_official_or_shared(client):
    r = client.post("/ingest", json=doc_payload(authority="official"), headers=headers())
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"
    r = client.post("/ingest", json=doc_payload(shared=True), headers=headers())
    assert r.status_code == 403


def test_analyst_cannot_register_confidential(client):
    r = client.post("/ingest", json=doc_payload(classification="confidential"), headers=headers())
    assert r.status_code == 403
    r = client.post("/ingest", json=doc_payload(classification="confidential"), headers=headers("acme_admin"))
    assert r.status_code == 201


def test_doc_id_cannot_overwrite_other_tenant(client):
    assert client.post("/ingest", json=doc_payload(), headers=headers("bluepeak")).status_code == 201
    r = client.post("/ingest", json=doc_payload(text="Overwrite attempt from another tenant."), headers=headers("acme"))
    assert r.status_code == 409 and r.json()["error"]["code"] == "doc_id_conflict"


def test_reingest_same_tenant_replaces(client):
    client.post("/ingest", json=doc_payload(), headers=headers())
    r = client.post("/ingest", json=doc_payload(text="Updated text for the very same ticket."), headers=headers())
    assert r.status_code == 201 and r.json()["replaced_existing"] is True


@pytest.mark.parametrize(
    "body",
    [{}, {"question": ""}, {"question": "  "}, {"question": "hi"}, {"question": "x" * 1001}, {"question": "valid question?", "top_k": 0},
     {"question": "valid question?", "top_k": 21}, {"question": "valid question?", "tenant": "bluepeak"}],
)
def test_ask_validation_errors(client, body):
    r = client.post("/ask", json=body, headers=headers())
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def pdf_form(**over):
    form = {"doc_id": "PDF-9001", "title": "Uploaded PDF"}
    form.update(over)
    return form


def test_pdf_upload_ok(client):
    data = (PDF_DIR / "SLA-2025.pdf").read_bytes()
    r = client.post("/ingest/pdf", data=pdf_form(), files={"file": ("a.pdf", data, "application/pdf")}, headers=headers())
    assert r.status_code == 201 and r.json()["parser"] == "pypdf" and r.json()["chunks"] >= 1


def test_pdf_upload_rejects_non_pdf(client):
    r = client.post("/ingest/pdf", data=pdf_form(), files={"file": ("a.pdf", b"hello, I am text", "application/pdf")}, headers=headers())
    assert r.status_code == 415 and r.json()["error"]["code"] == "not_a_pdf"


def test_pdf_upload_empty_and_oversize(client, settings):
    r = client.post("/ingest/pdf", data=pdf_form(), files={"file": ("a.pdf", b"", "application/pdf")}, headers=headers())
    assert r.status_code == 422 and r.json()["error"]["code"] == "empty_file"
    big = b"%PDF-1.4\n" + b"0" * (settings.max_upload_bytes + 1)
    r = client.post("/ingest/pdf", data=pdf_form(), files={"file": ("a.pdf", big, "application/pdf")}, headers=headers())
    assert r.status_code == 413


def test_pdf_upload_bad_metadata_is_422(client):
    data = (PDF_DIR / "SLA-2025.pdf").read_bytes()
    r = client.post("/ingest/pdf", data=pdf_form(doc_id="!"), files={"file": ("a.pdf", data, "application/pdf")}, headers=headers())
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_pdf_truncated_garbage_is_clear_error(client):
    r = client.post("/ingest/pdf", data=pdf_form(), files={"file": ("a.pdf", b"%PDF-1.4\n garbage only", "application/pdf")}, headers=headers())
    assert r.status_code == 422 and r.json()["error"]["code"] in {"pdf_unreadable", "pdf_empty"}


def test_unknown_route_uses_error_schema(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "http_error"


def test_ask_on_empty_store_abstains(client):
    r = ask(client, "How long do we retain shipment records?")
    assert r.status_code == 200 and r.json()["abstained"] is True
