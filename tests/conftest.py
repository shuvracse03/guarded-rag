import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.errors import ApiError
from app.main import create_app
from app.seed import seed

KEYS = {
    "acme": "key-acme-analyst",
    "acme_admin": "key-acme-admin",
    "bluepeak": "key-bluepeak-analyst",
    "bluepeak_admin": "key-bluepeak-admin",
}


def headers(who: str = "acme") -> dict:
    return {"X-API-Key": KEYS[who]}


class FakeLLM:
    """Scriptable stand-in for Ollama. Records every prompt it receives."""

    model = "fake-llm"

    def __init__(self, handler=None):
        self.handler = handler or honest_handler
        self.calls: list[tuple[str, str]] = []

    def ping(self) -> bool:
        return True

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.handler(system, user)


def parse_sources(user_prompt: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in re.finditer(r'<source id="([^"]+)"[^>]*>\n(.*?)\n</source>', user_prompt, re.S)}


def honest_handler(system: str, user: str) -> str:
    """Behaves like a well-behaved model: quotes the source sentence that best overlaps the question."""
    from app.retrieval.embeddings import tokenize

    question = user.rsplit("Question:", 1)[-1].split("JSON answer:")[0]
    q = set(tokenize(question))
    best, best_score = None, 1
    for sid, text in parse_sources(user).items():
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            score = len(q & set(tokenize(sentence))) + (0.5 if re.search(r"\d", sentence) else 0)
            if score > best_score:
                best, best_score = (sid, sentence.strip()), score
    if best is None:
        return json.dumps({"answer": None, "citations": []})
    return json.dumps({"answer": best[1], "citations": [{"source_id": best[0], "quote": best[1]}]})


class RaisingLLM(FakeLLM):
    def __init__(self, error: ApiError):
        super().__init__()
        self.error = error

    def generate(self, system, user):
        self.calls.append((system, user))
        raise self.error


@pytest.fixture
def settings() -> Settings:
    return Settings(embedding_backend="hash", use_memory_store=True, pdf_parser="pypdf", min_relevance=0.30)


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def app(settings, llm):
    application = create_app(settings, llm=llm)
    return application


@pytest.fixture
def container(app):
    return app.state.container


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def seeded(container, settings):
    seed(container.ingest, settings)
    return container


@pytest.fixture
def seeded_client(client, seeded) -> TestClient:
    return client


def ask(client: TestClient, question: str, who: str = "acme", **extra):
    return client.post("/ask", json={"question": question, **extra}, headers=headers(who))


def doc_payload(**over) -> dict:
    base = {"doc_id": "T-9001", "doc_type": "ticket", "title": "Sample ticket", "text": "The printer on dock 4 is jammed again."}
    base.update(over)
    return base


PDF_DIR = Path(__file__).resolve().parents[1] / "data" / "pdfs"
