"""High-level orchestration connecting Django Document/Chunk models with RAG services."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Sequence, Tuple

from django.db import transaction

from home.models import Document, DocumentChunk, ProcessedDocument
from home.services.chunker import build_document_chunks
from home.services.document_loader import load_document_pages
from home.services.llm_client import generate_grounded_answer
from home.services.retriever import (
    ChunkCandidate,
    EmbedFn,
    RetrievedChunk,
    decode_vector_bytes,
    encode_vector_bytes,
    get_default_embedder,
    hybrid_retrieve,
)


def ingest_document(doc: Document, embed_fn: EmbedFn | None = None) -> int:
    """Extract, chunk, embed, and persist normalized DocumentChunk rows for a Document."""
    file_path = Path(doc.document.path)
    file_label = Path(doc.document.name).name
    pages = load_document_pages(file_path)
    if not pages:
        raise ValueError(f"Không trích xuất được nội dung văn bản từ tệp '{file_label}'.")

    chunks = build_document_chunks(pages, file_label=file_label)
    if not chunks:
        raise ValueError(f"Tệp '{file_label}' không có đoạn văn bản hợp lệ.")

    embedder = embed_fn or get_default_embedder()
    vectors = embedder([c.text for c in chunks])
    full_text = "\n\n".join(p.text for p in pages)

    with transaction.atomic():
        DocumentChunk.objects.filter(document=doc).delete()
        chunk_rows = [
            DocumentChunk(
                document=doc,
                chunk_index=c.chunk_index,
                page_number=c.page_number,
                text=c.text,
                embedding=encode_vector_bytes(vectors[idx]),
            )
            for idx, c in enumerate(chunks)
        ]
        DocumentChunk.objects.bulk_create(chunk_rows)

        ProcessedDocument.objects.filter(document=doc).delete()
        ProcessedDocument.objects.create(
            file_name=file_label,
            document=doc,
            text_content=full_text,
            embeddings=None,
        )

        doc.is_processed = True
        doc.save(update_fields=["is_processed"])

    return len(chunk_rows)


def process_pending_documents(embed_fn: EmbedFn | None = None) -> int:
    """Process all uploaded documents where is_processed is False."""
    processed_count = 0
    for doc in Document.objects.filter(is_processed=False):
        ingest_document(doc, embed_fn=embed_fn)
        processed_count += 1
    return processed_count


def retrieve_relevant_chunks(
    question: str,
    selected_doc_ids: Sequence[int] | None = None,
    top_k: int = 5,
    embed_fn: EmbedFn | None = None,
) -> List[RetrievedChunk]:
    """Load candidate chunks (optionally filtered by selected document IDs) and run Hybrid RRF."""
    qs = DocumentChunk.objects.select_related("document").all()
    if selected_doc_ids:
        qs = qs.filter(document_id__in=list(selected_doc_ids))

    candidates = [
        ChunkCandidate(
            chunk_id=row.id,
            document_id=row.document_id,
            file_name=Path(row.document.document.name).name,
            page_number=row.page_number,
            text=row.text,
            embedding=decode_vector_bytes(bytes(row.embedding)),
        )
        for row in qs
    ]
    embedder = embed_fn or get_default_embedder()
    return hybrid_retrieve(question, candidates, top_k=top_k, embed_fn=embedder)


def answer_question(
    question: str,
    history: Sequence[Tuple[str, str]] | None = None,
    selected_doc_ids: Sequence[int] | None = None,
    use_web_search: bool = False,
    top_k: int = 5,
    embed_fn: EmbedFn | None = None,
    generator_fn: Callable[[str], str] | None = None,
) -> dict:
    """Run end-to-end retrieval + grounded generation and return answer with citations."""
    chunks = retrieve_relevant_chunks(
        question=question,
        selected_doc_ids=selected_doc_ids,
        top_k=top_k,
        embed_fn=embed_fn,
    )
    if not chunks and not use_web_search:
        return {
            "found": False,
            "answer": "Xin lỗi, hiện chưa có tài liệu phù hợp trong kho lưu trữ hoặc danh sách tài liệu đã chọn. Vui lòng tải lên hoặc chọn lại tài liệu.",
            "citations": [],
            "provider": "none",
        }

    gen = generate_grounded_answer(
        question=question,
        chunks=chunks,
        history=history,
        use_web_search=use_web_search,
        generator_fn=generator_fn,
    )
    citations = [
        {
            "document_id": c.document_id,
            "file_name": c.file_name,
            "page_number": c.page_number,
            "score": c.score,
            "excerpt": c.text[:260] + ("..." if len(c.text) > 260 else ""),
        }
        for c in chunks
    ]
    return {
        "found": True,
        "answer": gen["answer"],
        "citations": citations,
        "provider": gen["provider"],
        "web_used": gen["web_used"],
    }
