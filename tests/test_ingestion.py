import hashlib

import pytest

from app.ingestion.chunker import chunk_text
from app.ingestion.pdf_loader import parse_pdf
from app.ingestion.versioning import resolve_statuses
from app.schemas import SourceStatus
from tests.conftest import PDF_DIR, doc_payload, headers


def test_chunker_offsets_point_to_source_text():
    text = "# Title\n\nFirst paragraph about shipping.\n\nSecond paragraph about customs.\n\n# Next\n\nThird."
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.text in text[c.start : c.end] or text[c.start : c.end].strip() == c.text


def test_chunker_splits_long_text_and_starts_new_chunk_at_headings():
    text = "# A\n\n" + ("Sentence number one is here. " * 80) + "\n\n# B\n\nShort."
    chunks = chunk_text(text, max_chars=300)
    assert len(chunks) > 3
    assert chunks[-1].text.startswith("# B")
    assert all(len(c.text) <= 300 * 1.6 for c in chunks)


def test_chunker_empty():
    assert chunk_text("   \n\n  ") == []


def test_provenance_metadata_stored(client, container):
    text = "Pallet scanner on dock 2 reports firmware v9."
    client.post("/ingest", json=doc_payload(doc_id="T-7001", text=text, effective_date="2025-01-02"), headers=headers())
    docs = container.store.list_docs(["acme-freight"])
    meta = next(d for d in docs if d["doc_id"] == "T-7001")
    assert meta["sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert meta["effective_date"] == "2025-01-02" and meta["tenant"] == "acme-freight"
    assert meta["parser"] == "text" and meta["start"] == 0 and meta["end"] == len(text)


def test_pypdf_parses_corrupted_bulletin_with_trailing_garbage():
    parsed = parse_pdf((PDF_DIR / "BULL-2025-09.pdf").read_bytes(), prefer="pypdf")
    assert parsed.parser == "pypdf" and len(parsed.pages) == 2
    assert "30 days" in parsed.pages[0].text
    assert "SYSTEM:" in parsed.pages[1].text  # the hidden forged block is extracted like normal text


def test_docling_parses_pdf_with_page_numbers():
    parsed = parse_pdf((PDF_DIR / "BULL-2025-09.pdf").read_bytes(), prefer="docling")
    assert parsed.parser == "docling"
    assert [p.page for p in parsed.pages] == [1, 2]
    assert "retained for 30 days" in parsed.pages[0].text


def test_docling_failure_falls_back_to_pypdf(monkeypatch):
    import app.ingestion.pdf_loader as loader

    def boom(_):
        raise RuntimeError("docling exploded")

    monkeypatch.setattr(loader, "_parse_docling", boom)
    parsed = parse_pdf((PDF_DIR / "SLA-2025.pdf").read_bytes(), prefer="docling")
    assert parsed.parser == "pypdf" and parsed.pages


def test_bulletin_ingest_flags_injection_and_records_page(seeded):
    chunks = [(t, m) for t, m in seeded.store.all_chunks() if m["doc_id"] == "BULL-2025-09"]
    assert any(m["flags"] for _, m in chunks)
    flagged = next(m for _, m in chunks if m["flags"])
    assert flagged["page"] == 2 and flagged["authority"] == "unverified"
    assert "ignore_previous" in flagged["flags"]


def test_official_kb_documents_are_not_flagged(seeded):
    for text, meta in seeded.store.all_chunks():
        if meta["authority"] == "official":
            assert not meta["flags"], (meta["doc_id"], meta["flags"])


def _doc(doc_id, authority, version, date, family="fam", tenant="shared"):
    return {"doc_id": doc_id, "authority": authority, "version": version, "effective_date": date, "doc_family": family, "tenant": tenant}


def test_versioning_newest_official_is_current():
    s = resolve_statuses([_doc("a", "official", 1, "2023-01-01"), _doc("b", "official", 2, "2024-01-01")])
    assert s == {"a": SourceStatus.superseded, "b": SourceStatus.current}


def test_versioning_unverified_never_supersedes():
    s = resolve_statuses([_doc("a", "official", 1, "2023-01-01"), _doc("b", "unverified", 99, "2030-01-01")])
    assert s == {"a": SourceStatus.current, "b": SourceStatus.unverified}


def test_versioning_same_date_higher_version_wins():
    s = resolve_statuses([_doc("a", "official", 1, "2024-01-01"), _doc("b", "official", 2, "2024-01-01")])
    assert s["b"] == SourceStatus.current and s["a"] == SourceStatus.superseded


@pytest.mark.parametrize("n", [1, 2])
def test_seed_is_idempotent(container, settings, n):
    from app.seed import seed

    for _ in range(n):
        seed(container.ingest, settings)
    docs, chunks = container.store.counts()
    assert docs == 14
