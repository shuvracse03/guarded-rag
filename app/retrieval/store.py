import uuid
from dataclasses import dataclass

import chromadb


@dataclass
class Hit:
    chunk_id: str
    text: str
    meta: dict
    vector_score: float  # cosine similarity in [-1, 1]
    relevance: float = 0.0


class VectorStore:
    """Thin wrapper over ChromaDB. Tenant filtering is applied inside the query itself."""

    def __init__(self, persist_dir: str | None = None):
        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
            name = "documents"
        else:
            self._client = chromadb.EphemeralClient()
            name = f"documents-{uuid.uuid4().hex[:12]}"  # isolate in-memory instances from each other
        self._col = self._client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    def add_chunks(self, ids, texts, embeddings, metas) -> None:
        self._col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)

    def delete_doc(self, doc_id: str) -> int:
        existing = self._col.get(where={"doc_id": doc_id})
        if existing["ids"]:
            self._col.delete(ids=existing["ids"])
        return len(existing["ids"])

    def doc_tenant(self, doc_id: str) -> str | None:
        res = self._col.get(where={"doc_id": doc_id}, limit=1, include=["metadatas"])
        return res["metadatas"][0]["tenant"] if res["ids"] else None

    def query(self, embedding: list[float], tenants: list[str], n: int) -> list[Hit]:
        total = self._col.count()
        if total == 0:
            return []
        res = self._col.query(
            query_embeddings=[embedding],
            n_results=min(n, total),
            where={"tenant": {"$in": tenants}},
            include=["documents", "metadatas", "distances"],
        )
        return [
            Hit(chunk_id=i, text=d, meta=m, vector_score=1.0 - dist)
            for i, d, m, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def list_docs(self, tenants: list[str]) -> list[dict]:
        res = self._col.get(where={"tenant": {"$in": tenants}}, include=["metadatas"])
        docs: dict[str, dict] = {}
        for m in res["metadatas"]:
            docs.setdefault(m["doc_id"], m)
        return list(docs.values())

    def all_chunks(self) -> list[tuple[str, dict]]:
        res = self._col.get(include=["documents", "metadatas"])
        return list(zip(res["documents"], res["metadatas"]))

    def counts(self) -> tuple[int, int]:
        res = self._col.get(include=["metadatas"])
        return len({m["doc_id"] for m in res["metadatas"]}), len(res["ids"])
