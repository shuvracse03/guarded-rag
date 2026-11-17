import hashlib
from dataclasses import dataclass

from app.auth import Principal
from app.errors import ApiError
from app.ingestion.chunker import chunk_text
from app.ingestion.pdf_loader import parse_pdf
from app.retrieval.embeddings import Embedder
from app.retrieval.store import VectorStore
from app.schemas import Authority, DocType, DocumentIn, DocumentMeta, IngestResult
from app.security.injection_detector import detect
from app.security.policy import SHARED_TENANT


@dataclass
class IngestService:
    store: VectorStore
    embedder: Embedder
    pdf_parser: str = "docling"

    def ingest_document(self, doc: DocumentIn, principal: Principal) -> IngestResult:
        return self._index(doc, doc.doc_type, principal, chunk_text(doc.text), "text", doc.text.encode())

    def ingest_pdf(self, meta: DocumentMeta, data: bytes, principal: Principal) -> IngestResult:
        parsed = parse_pdf(data, prefer=self.pdf_parser)
        chunks = [c for p in parsed.pages for c in chunk_text(p.text, page=p.page)]
        return self._index(meta, DocType.pdf, principal, chunks, parsed.parser, data)

    def _index(self, meta: DocumentMeta, doc_type: DocType, principal: Principal, chunks, parser: str, source: bytes):
        # Privileged attributes can only be set by an admin; content can never grant them.
        if (meta.authority == Authority.official or meta.shared) and principal.role != "admin":
            raise ApiError(403, "forbidden", "Only admins can register official or shared documents")
        if meta.classification.value not in principal.clearance:
            raise ApiError(403, "forbidden", "Cannot register a document above your clearance")
        if not chunks:
            raise ApiError(422, "empty_document", "Document produced no indexable text")

        tenant = SHARED_TENANT if meta.shared else principal.tenant
        owner = self.store.doc_tenant(meta.doc_id)
        if owner is not None and owner != tenant:
            raise ApiError(409, "doc_id_conflict", "doc_id is already in use")
        replaced = owner is not None

        sha = hashlib.sha256(source).hexdigest()
        ids, texts, metas, flagged = [], [], [], 0
        for i, c in enumerate(chunks):
            page = c.page
            det = detect(c.text)
            flagged += det.flagged
            ids.append(f"{meta.doc_id}#{i}")
            texts.append(c.text)
            metas.append(
                {
                    "tenant": tenant,
                    "doc_id": meta.doc_id,
                    "title": meta.title,
                    "doc_type": doc_type.value,
                    "version": meta.version,
                    "effective_date": meta.effective_date.isoformat() if meta.effective_date else "",
                    "doc_family": meta.doc_family or "",
                    "topic": meta.topic or "",
                    "authority": meta.authority.value,
                    "classification": meta.classification.value,
                    "page": page or -1,
                    "start": c.start,
                    "end": c.end,
                    "sha256": sha,
                    "parser": parser,
                    "flags": ",".join(det.flags),
                    "chunk_index": i,
                }
            )
        embeddings = self.embedder.embed(texts)  # embed first so a failure never deletes the old version
        if replaced:
            self.store.delete_doc(meta.doc_id)
        self.store.add_chunks(ids, texts, embeddings, metas)
        return IngestResult(
            doc_id=meta.doc_id,
            tenant=tenant,
            chunks=len(ids),
            parser=parser,
            sha256=sha,
            authority=meta.authority,
            injection_flagged_chunks=flagged,
            replaced_existing=replaced,
        )
