import logging
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from app.auth import Principal, get_principal
from app.config import Settings, get_settings
from app.errors import ApiError, install_error_handlers
from app.ingestion.service import IngestService
from app.rag.answer import AnswerService
from app.rag.llm import OllamaLLM
from app.retrieval.embeddings import HashEmbedder, OllamaEmbedder
from app.retrieval.retriever import Retriever
from app.retrieval.store import VectorStore
from app.schemas import (
    AskRequest, AskResponse, DocType, DocumentIn, DocumentMeta, ErrorResponse, HealthResponse, IngestResult,
)

log = logging.getLogger("guarded-rag")
ERRORS = {r: {"model": ErrorResponse} for r in (400, 401, 403, 409, 413, 415, 422, 502, 503, 504)}


class Container:
    def __init__(self, settings: Settings, llm=None, embedder=None, store=None):
        self.settings = settings
        self.embedder = embedder or (
            HashEmbedder() if settings.embedding_backend == "hash"
            else OllamaEmbedder(settings.ollama_url, settings.embed_model)
        )
        self.store = store or VectorStore(None if settings.use_memory_store else settings.chroma_dir)
        self.llm = llm or OllamaLLM(settings.ollama_url, settings.llm_model, settings.llm_timeout_s)
        self.ingest = IngestService(self.store, self.embedder, settings.pdf_parser)
        self.answers = AnswerService(
            Retriever(self.store, self.embedder), self.llm, getattr(self.llm, "model", settings.llm_model),
            settings.min_relevance, settings.default_top_k,
        )


def create_app(settings: Settings | None = None, llm=None, embedder=None, store=None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="Guarded RAG", version="1.0.0",
                  description="Injection-resistant, tenant-aware RAG over synthetic tickets, logs and PDFs.")
    app.state.settings = settings
    app.state.container = Container(settings, llm, embedder, store)
    install_error_handlers(app)

    def container(request: Request) -> Container:
        return request.app.state.container

    @app.get("/health", response_model=HealthResponse)
    def health(c: Annotated[Container, Depends(container)]):
        reachable = c.llm.ping() if hasattr(c.llm, "ping") else True
        docs, chunks = c.store.counts()
        return HealthResponse(
            status="ok" if reachable else "degraded", ollama_reachable=reachable,
            llm_model=getattr(c.llm, "model", settings.llm_model), embed_model=c.embedder.name,
            embedding_backend=settings.embedding_backend, documents=docs, chunks=chunks,
        )

    def _ingest_json(doc: DocumentIn, c: Container, principal: Principal) -> IngestResult:
        return c.ingest.ingest_document(doc, principal)

    @app.post("/ingest", response_model=IngestResult, status_code=201, responses=ERRORS)
    def ingest(doc: DocumentIn, c: Annotated[Container, Depends(container)],
               principal: Annotated[Principal, Depends(get_principal)]):
        return _ingest_json(doc, c, principal)

    @app.post("/tickets", response_model=IngestResult, status_code=201, responses=ERRORS)
    def create_ticket(doc: DocumentIn, c: Annotated[Container, Depends(container)],
                      principal: Annotated[Principal, Depends(get_principal)]):
        if doc.doc_type != DocType.ticket:
            raise ApiError(422, "validation_error", "POST /tickets only accepts doc_type='ticket'")
        return _ingest_json(doc, c, principal)

    @app.post("/ingest/pdf", response_model=IngestResult, status_code=201, responses=ERRORS)
    async def ingest_pdf(
        c: Annotated[Container, Depends(container)],
        principal: Annotated[Principal, Depends(get_principal)],
        file: UploadFile = File(...),
        doc_id: str = Form(...), title: str = Form(...), version: int = Form(1),
        effective_date: str | None = Form(None), doc_family: str | None = Form(None),
        topic: str | None = Form(None), authority: str = Form("unverified"),
        classification: str = Form("internal"), shared: bool = Form(False),
    ):
        try:
            meta = DocumentMeta(
                doc_id=doc_id, title=title, version=version, effective_date=effective_date or None,
                doc_family=doc_family or None, topic=topic or None, authority=authority,
                classification=classification, shared=shared,
            )
        except ValidationError as exc:
            details = [{"field": ".".join(map(str, e["loc"])), "message": e["msg"]} for e in exc.errors()]
            raise ApiError(422, "validation_error", "Request validation failed", details) from exc
        data = await file.read(c.settings.max_upload_bytes + 1)
        if len(data) > c.settings.max_upload_bytes:
            raise ApiError(413, "file_too_large", f"PDF exceeds {c.settings.max_upload_bytes} bytes")
        if not data:
            raise ApiError(422, "empty_file", "Uploaded file is empty")
        return await run_in_threadpool(c.ingest.ingest_pdf, meta, data, principal)

    @app.post("/ask", response_model=AskResponse, responses=ERRORS)
    def ask(body: AskRequest, c: Annotated[Container, Depends(container)],
            principal: Annotated[Principal, Depends(get_principal)]):
        return c.answers.ask(body.question, principal, body.top_k)

    return app


def __getattr__(name: str):
    # Lazy so importing this module (e.g. in tests) never touches the on-disk vector store.
    if name == "app":
        global _app
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)


_app: FastAPI | None = None
