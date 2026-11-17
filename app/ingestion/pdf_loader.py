"""PDF parsing: Docling first, pypdf as a logged fallback for files Docling cannot read."""
import io
import logging
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.errors import ApiError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedPage:
    page: int
    text: str


@dataclass(frozen=True)
class ParsedPdf:
    pages: list[ParsedPage]
    parser: str  # "docling" | "pypdf"


@lru_cache(maxsize=1)
def _docling_converter():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = False  # synthetic PDFs have a text layer; OCR models are not needed
    opts.do_table_structure = False
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def _parse_docling(data: bytes) -> list[ParsedPage]:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "upload.pdf"
        path.write_bytes(data)
        doc = _docling_converter().convert(str(path)).document
    pages = []
    for page_no in sorted(doc.pages.keys()):
        md = doc.export_to_markdown(page_no=page_no)
        if md.strip():
            pages.append(ParsedPage(page=page_no, text=md))
    return pages


def _parse_pypdf(data: bytes) -> list[ParsedPage]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data), strict=False)
    return [
        ParsedPage(page=i + 1, text=(p.extract_text() or ""))
        for i, p in enumerate(reader.pages)
        if (p.extract_text() or "").strip()
    ]


def parse_pdf(data: bytes, prefer: str = "docling") -> ParsedPdf:
    if not data.lstrip()[:5].startswith(b"%PDF"):
        raise ApiError(415, "not_a_pdf", "Uploaded file does not look like a PDF (missing %PDF header)")
    pages: list[ParsedPage] = []
    parser = prefer
    if prefer == "docling":
        try:
            pages = _parse_docling(data)
        except Exception as exc:  # corrupted/unsupported input: fall back, but say so
            log.warning("Docling failed (%s); falling back to pypdf", exc)
    if not pages:
        parser = "pypdf"
        try:
            pages = _parse_pypdf(data)
        except Exception as exc:
            raise ApiError(422, "pdf_unreadable", f"PDF could not be parsed: {type(exc).__name__}") from exc
    if not pages:
        raise ApiError(422, "pdf_empty", "PDF contains no extractable text")
    return ParsedPdf(pages=pages, parser=parser)
