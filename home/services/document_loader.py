"""Multi-format document loader supporting PDF, DOCX, and plain text files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from docx import Document as DocxDocument
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@dataclass(frozen=True)
class LoadedPage:
    """Extracted text segment tied to a 1-based page or section number."""

    page_number: int
    text: str


def load_document_pages(file_path: str | Path) -> List[LoadedPage]:
    """Extract clean text pages from PDF, DOCX, or TXT/MD files."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document file not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    if ext == ".pdf":
        reader = PdfReader(str(path))
        pages: List[LoadedPage] = []
        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(LoadedPage(page_number=idx, text=text))
        return pages

    if ext == ".docx":
        doc = DocxDocument(str(path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
                if cells:
                    paragraphs.append(" | ".join(cells))
        combined = "\n".join(paragraphs).strip()
        return [LoadedPage(page_number=1, text=combined)] if combined else []

    raw_text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [LoadedPage(page_number=1, text=raw_text)] if raw_text else []
