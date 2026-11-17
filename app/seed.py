"""Load the synthetic corpus: `python -m app.seed [--reset]`."""
import argparse
import json
from pathlib import Path

from app.auth import Principal
from app.config import get_settings
from app.ingestion.service import IngestService
from app.main import Container
from app.schemas import DocumentIn, DocumentMeta

DATA = Path(__file__).resolve().parents[1] / "data"
META_FIELDS = set(DocumentMeta.model_fields)


def seed(service: IngestService, settings, data_dir: Path = DATA) -> list:
    manifest = json.loads((data_dir / "synthetic" / "manifest.json").read_text())
    results = []
    for entry in manifest:
        key = settings.api_keys[entry["principal"]]
        principal = Principal(tenant=key.tenant, role=key.role)
        meta = {k: v for k, v in entry.items() if k in META_FIELDS}
        path = (data_dir / "pdfs" / Path(entry["file"]).name) if entry["doc_type"] == "pdf" else data_dir / "synthetic" / entry["file"]
        if entry["doc_type"] == "pdf":
            results.append(service.ingest_pdf(DocumentMeta(**meta), path.read_bytes(), principal))
        else:
            results.append(service.ingest_document(
                DocumentIn(**meta, doc_type=entry["doc_type"], text=path.read_text()), principal))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="delete the existing documents collection first")
    args = parser.parse_args()
    settings = get_settings()
    c = Container(settings)
    if args.reset:
        c.store.reset()  # not rmtree: deleting the folder would break a server that is already running
    for r in seed(c.ingest, settings):
        print(f"{r.doc_id:14} tenant={r.tenant:13} chunks={r.chunks} parser={r.parser:7} flagged={r.injection_flagged_chunks}")


if __name__ == "__main__":
    main()
