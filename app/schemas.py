from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,63}$"
SLUG_PATTERN = r"^[a-z0-9][a-z0-9_-]{1,63}$"


class DocType(str, Enum):
    ticket = "ticket"
    log = "log"
    kb = "kb"
    pdf = "pdf"


class Classification(str, Enum):
    public = "public"
    internal = "internal"
    confidential = "confidential"


class Authority(str, Enum):
    # Only admins can register "official" documents; everything else is untrusted by default.
    official = "official"
    unverified = "unverified"


class SourceStatus(str, Enum):
    current = "current"
    superseded = "superseded"
    unverified = "unverified"


class DocumentMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a spoofed `tenant` field is rejected, not ignored

    doc_id: str = Field(pattern=ID_PATTERN, examples=["KB-RET-002"])
    title: str = Field(min_length=3, max_length=200)
    version: int = Field(default=1, ge=1, le=10_000)
    effective_date: date | None = None
    doc_family: str | None = Field(default=None, pattern=SLUG_PATTERN)
    topic: str | None = Field(default=None, pattern=SLUG_PATTERN)
    authority: Authority = Authority.unverified
    classification: Classification = Classification.internal
    shared: bool = Field(default=False, description="Admin only: readable by every tenant.")


class DocumentIn(DocumentMeta):
    doc_type: Literal[DocType.ticket, DocType.log, DocType.kb]
    text: str = Field(min_length=10, max_length=100_000)

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class IngestResult(BaseModel):
    doc_id: str
    tenant: str
    chunks: int
    parser: str
    sha256: str
    authority: Authority
    injection_flagged_chunks: int
    replaced_existing: bool


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


class Citation(BaseModel):
    source_id: str  # chunk id, e.g. "KB-RET-002#0"
    doc_id: str
    title: str
    page: int | None
    excerpt: str
    status: SourceStatus
    version: int
    sha256: str


class SourceNote(BaseModel):
    source_id: str
    doc_id: str
    title: str
    status: SourceStatus
    version: int
    reason: str
    excerpt: str


class ConflictInfo(BaseModel):
    topic: str
    values: list[dict]
    resolution: str


class AskResponse(BaseModel):
    request_id: str
    answer: str | None
    abstained: bool
    abstain_reason: str | None = None
    conflict: bool = False
    citations: list[Citation] = []
    stale_sources: list[SourceNote] = []
    flagged_sources: list[SourceNote] = []
    conflicts: list[ConflictInfo] = []
    model: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list | dict | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ollama_reachable: bool
    llm_model: str
    embed_model: str
    embedding_backend: str
    documents: int
    chunks: int
