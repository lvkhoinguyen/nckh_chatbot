"""Sentence-aware Vietnamese text chunker with guaranteed monotonic forward progress."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from home.services.document_loader import LoadedPage


@dataclass(frozen=True)
class TextChunk:
    """Chunk with contextual metadata for grounded citations."""

    chunk_index: int
    page_number: int
    text: str


def split_text_safely(text: str, chunk_size: int = 750, overlap: int = 120) -> List[str]:
    """Split text into overlapping chunks without infinite loops on edge punctuation."""
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    safe_overlap = max(0, min(overlap, chunk_size - 1))
    chunks: List[str] = []
    start = 0
    total_len = len(cleaned)

    while start < total_len:
        end = min(start + chunk_size, total_len)
        if end < total_len:
            # Prefer snapping backward to sentence or whitespace boundary inside upper half of window
            min_snap = start + max(1, chunk_size // 2)
            snapped = -1
            for sep in (". ", "! ", "? ", "; ", "\n", " "):
                pos = cleaned.rfind(sep, min_snap, end)
                if pos != -1:
                    snapped = pos + (0 if sep == " " else 1)
                    break
            if snapped > start:
                end = snapped

        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= total_len:
            break

        # Guarantee strict forward progress so start never stalls or goes backward
        next_start = end - safe_overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def build_document_chunks(
    pages: Iterable[LoadedPage],
    file_label: str,
    chunk_size: int = 750,
    overlap: int = 120,
) -> List[TextChunk]:
    """Chunk loaded document pages and prepend lightweight file/page context headers."""
    result: List[TextChunk] = []
    idx = 0
    for page in pages:
        for piece in split_text_safely(page.text, chunk_size=chunk_size, overlap=overlap):
            contextualized = f"[{file_label} | Trang {page.page_number}]\n{piece}"
            result.append(TextChunk(chunk_index=idx, page_number=page.page_number, text=contextualized))
            idx += 1
    return result
