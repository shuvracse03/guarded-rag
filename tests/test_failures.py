import httpx
import pytest

from app.errors import ApiError
from app.rag.llm import OllamaLLM
from tests.conftest import FakeLLM, RaisingLLM, ask

Q = "How long do we retain shipment records?"


def test_llm_unavailable_returns_503_error_schema(app, seeded_client):
    app.state.container.answers.llm = RaisingLLM(ApiError(503, "llm_unavailable", "Cannot reach the local Ollama server"))
    r = ask(seeded_client, Q)
    assert r.status_code == 503 and r.json()["error"]["code"] == "llm_unavailable"


def test_llm_timeout_returns_504(app, seeded_client):
    app.state.container.answers.llm = RaisingLLM(ApiError(504, "llm_timeout", "The language model timed out"))
    assert ask(seeded_client, Q).status_code == 504


def test_malformed_model_output_retries_once_then_abstains(app, seeded_client):
    bad = FakeLLM(lambda s, u: "I am not JSON at all")
    app.state.container.answers.llm = bad
    body = ask(seeded_client, Q).json()
    assert body["abstained"] and body["abstain_reason"] == "invalid_model_output"
    assert len(bad.calls) == 2


def test_malformed_then_valid_output_recovers(app, seeded_client):
    from tests.conftest import honest_handler

    state = {"n": 0}

    def flaky(s, u):
        state["n"] += 1
        return "oops" if state["n"] == 1 else honest_handler(s, u)

    app.state.container.answers.llm = FakeLLM(flaky)
    body = ask(seeded_client, Q).json()
    assert not body["abstained"] and "180" in body["answer"]


def test_model_answering_null_is_an_abstention(app, seeded_client):
    app.state.container.answers.llm = FakeLLM(lambda s, u: '{"answer": null, "citations": []}')
    body = ask(seeded_client, Q).json()
    assert body["abstained"] and body["abstain_reason"] == "insufficient_evidence"


def test_ollama_connection_refused_maps_to_503():
    llm = OllamaLLM("http://127.0.0.1:9", "llama3.2", timeout=1)  # port 9 (discard): nothing listens
    with pytest.raises(ApiError) as e:
        llm.generate("s", "u")
    assert e.value.status == 503 and e.value.code == "llm_unavailable"
    assert llm.ping() is False


def test_ollama_http_error_and_bad_body_mapping(monkeypatch):
    llm = OllamaLLM("http://x", "m")

    class R:
        def __init__(self, code, js):
            self.status_code, self._js = code, js

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("e", request=httpx.Request("POST", "http://x"), response=self)

        def json(self):
            return self._js

    monkeypatch.setattr(httpx, "post", lambda *a, **k: R(500, {}))
    with pytest.raises(ApiError) as e:
        llm.generate("s", "u")
    assert e.value.status == 502 and e.value.code == "llm_error"
    monkeypatch.setattr(httpx, "post", lambda *a, **k: R(200, {"unexpected": True}))
    with pytest.raises(ApiError) as e:
        llm.generate("s", "u")
    assert e.value.code == "llm_bad_response"

    def timeout(*a, **k):
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(httpx, "post", timeout)
    with pytest.raises(ApiError) as e:
        llm.generate("s", "u")
    assert e.value.status == 504


def test_embedding_service_down_returns_503(settings, llm):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.retrieval.embeddings import OllamaEmbedder

    app = create_app(settings, llm=llm, embedder=OllamaEmbedder("http://127.0.0.1:9", "nomic-embed-text", timeout=1))
    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/ingest", json={"doc_id": "T-8001", "doc_type": "ticket", "title": "Sample", "text": "Some ticket text here."},
               headers={"X-API-Key": "key-acme-analyst"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "embedding_unavailable"


def test_unexpected_exception_returns_generic_500(app, seeded_client):
    def boom(s, u):
        raise RuntimeError("secret internal detail")

    app.state.container.answers.llm = FakeLLM(boom)
    r = ask(seeded_client, Q)
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal_error"
    assert "secret internal detail" not in r.text
