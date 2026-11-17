from app.auth import Principal
from app.ingestion.versioning import resolve_statuses
from app.retrieval.embeddings import Embedder, tokenize
from app.retrieval.store import Hit, VectorStore
from app.security import policy


def keyword_coverage(question: str, text: str, title: str = "") -> float:
    q = set(tokenize(question))
    if not q:
        return 0.0
    doc = set(tokenize(text)) | set(tokenize(title))
    return len(q & doc) / len(q)


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, question: str, principal: Principal, top_k: int) -> list[Hit]:
        tenants = policy.allowed_tenants(principal)  # tenant filter is part of the store query itself
        hits = self.store.query(self.embedder.embed([question])[0], tenants, n=top_k * 3)
        hits = policy.filter_readable(principal, hits)  # independent second check
        readable_docs = [m for m in self.store.list_docs(tenants) if policy.can_read(principal, m)]
        statuses = resolve_statuses(readable_docs)
        for h in hits:
            h.meta = {**h.meta, "status": statuses[h.meta["doc_id"]].value}
            cov = keyword_coverage(question, h.text, h.meta.get("title", ""))
            h.relevance = round(0.6 * cov + 0.4 * max(h.vector_score, 0.0), 4)
        hits.sort(key=lambda h: h.relevance, reverse=True)
        return hits[:top_k]
