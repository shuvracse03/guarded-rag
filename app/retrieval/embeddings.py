import hashlib
import math
import re
from typing import Protocol

import httpx

from app.errors import ApiError

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it long many much of on or our the "
    "their this to we what when where which who why with you your should would could there that "
    "about any all has have had was were will".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def _stem(tok: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic feature-hashing embedder. Offline, dependency-free; used in tests and as a fallback."""

    name = "hash-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            toks = tokenize(text)
            grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for g in grams:
                h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big")
                vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            r = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()["embeddings"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise ApiError(
                503, "embedding_unavailable", f"Embedding service (Ollama model '{self.model}') is unavailable"
            ) from exc
