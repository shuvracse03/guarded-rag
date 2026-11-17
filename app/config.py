from functools import lru_cache
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

Role = Literal["analyst", "admin"]


class KeyEntry(BaseModel):
    tenant: str
    role: Role


# Synthetic demo keys. They are placeholders for the exercise, not real credentials.
DEFAULT_API_KEYS = {
    "key-acme-analyst": KeyEntry(tenant="acme-freight", role="analyst"),
    "key-acme-admin": KeyEntry(tenant="acme-freight", role="admin"),
    "key-bluepeak-analyst": KeyEntry(tenant="bluepeak", role="analyst"),
    "key-bluepeak-admin": KeyEntry(tenant="bluepeak", role="admin"),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2"
    embed_model: str = "nomic-embed-text"
    embedding_backend: Literal["ollama", "hash"] = "ollama"
    pdf_parser: Literal["docling", "pypdf"] = "docling"
    chroma_dir: str = "./chroma_db"
    # True => in-memory vector store (used by tests)
    use_memory_store: bool = False
    min_relevance: float = 0.30
    default_top_k: int = 6
    max_upload_bytes: int = 5 * 1024 * 1024
    max_text_chars: int = 100_000
    llm_timeout_s: float = 180.0
    api_keys: dict[str, KeyEntry] = DEFAULT_API_KEYS


@lru_cache
def get_settings() -> Settings:
    return Settings()
