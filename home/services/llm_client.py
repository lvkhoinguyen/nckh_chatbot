"""Grounded answer generation using modern google-genai (Gemini) or Anthropic Claude."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Callable, List, Sequence, Tuple

from home.services.retriever import RetrievedChunk


def search_google_cse(query: str, num_results: int = 3) -> List[str]:
    """Optional Google Custom Search lookup when explicitly enabled and configured."""
    api_key = os.getenv("GOOGLE_CSE_API_KEY", "").strip()
    cse_id = os.getenv("GOOGLE_CSE_ID", "").strip()
    if not api_key or not cse_id or not query.strip():
        return []

    params = urllib.parse.urlencode({"key": api_key, "cx": cse_id, "q": query, "num": num_results})
    url = f"https://www.googleapis.com/customsearch/v1?{params}"
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = data.get("items", [])
        return [
            f"- {item.get('title', '')}: {item.get('snippet', '')} ({item.get('link', '')})"
            for item in items
        ]
    except Exception:
        return []


def build_rag_prompt(
    question: str,
    chunks: Sequence[RetrievedChunk],
    history: Sequence[Tuple[str, str]] | None = None,
    web_snippets: Sequence[str] | None = None,
) -> str:
    """Assemble grounded Vietnamese RAG prompt with conversation window and citations."""
    context_blocks = [
        f"[Nguồn {i + 1}: {c.file_name} | Trang {c.page_number}]\n{c.text}"
        for i, c in enumerate(chunks)
    ]
    context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "Không có đoạn trích nào từ tài liệu."

    history_lines = []
    for prev_q, prev_a in (history or [])[-4:]:
        history_lines.append(f"Người dùng: {prev_q}\nTrợ lý: {prev_a}")
    history_text = "\n\n".join(history_lines) if history_lines else "Chưa có hội thoại trước đó."

    web_block = ""
    if web_snippets:
        web_block = "\n\nThông tin tham khảo bổ sung từ Web:\n" + "\n".join(web_snippets)

    return (
        "Bạn là trợ lý AI hỏi đáp tài liệu tiếng Việt chính xác và trung thực.\n"
        "Quy tắc bắt buộc:\n"
        "1. Chỉ trả lời dựa trên NGỮ CẢNH TÀI LIỆU được cung cấp dưới đây (và phần Web nếu có).\n"
        "2. Trích dẫn nguồn rõ ràng theo định dạng `[Tên file - Trang X]` sau các ý chính.\n"
        "3. Nếu tài liệu không chứa thông tin để trả lời, hãy nói rõ 'Tài liệu hiện tại không đề cập thông tin này' và KHÔNG tự suy diễn.\n\n"
        f"### Lịch sử hội thoại gần đây:\n{history_text}\n\n"
        f"### Ngữ cảnh tài liệu:\n{context_text}"
        f"{web_block}\n\n"
        f"### Câu hỏi hiện tại:\n{question}\n\n"
        "### Trả lời (bằng tiếng Việt, định dạng Markdown rõ ràng):"
    )


def _call_gemini(prompt: str, api_key: str) -> str:
    from google import genai

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=prompt)
    return (response.text or "").strip()


def _call_claude(prompt: str, api_key: str) -> str:
    model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    endpoint = os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages")
    payload = json.dumps(
        {
            "model": model_name,
            "max_tokens": 1200,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=payload,
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    parts = [block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"]
    return "\n".join(parts).strip()


def generate_grounded_answer(
    question: str,
    chunks: Sequence[RetrievedChunk],
    history: Sequence[Tuple[str, str]] | None = None,
    use_web_search: bool = False,
    generator_fn: Callable[[str], str] | None = None,
) -> dict:
    """Generate a grounded answer plus metadata and explicit fallback notice if no LLM key is set."""
    web_snippets = search_google_cse(question) if use_web_search else []
    prompt = build_rag_prompt(question, chunks, history=history, web_snippets=web_snippets)

    if generator_fn is not None:
        return {"answer": generator_fn(prompt), "provider": "custom", "web_used": bool(web_snippets)}

    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    claude_key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    try:
        if provider == "claude" and claude_key:
            return {"answer": _call_claude(prompt, claude_key), "provider": "claude", "web_used": bool(web_snippets)}
        if gemini_key:
            return {"answer": _call_gemini(prompt, gemini_key), "provider": "gemini", "web_used": bool(web_snippets)}
        if claude_key:
            return {"answer": _call_claude(prompt, claude_key), "provider": "claude", "web_used": bool(web_snippets)}
    except Exception as exc:
        top_excerpts = "\n".join(
            f"- **[{c.file_name} - Trang {c.page_number}]**: {c.text[:260]}..."
            for c in chunks[:3]
        )
        return {
            "answer": (
                f"⚠️ **Lỗi kết nối LLM ({exc})** — Dưới đây là các đoạn trích phù hợp nhất tìm thấy từ tài liệu:\n\n"
                f"{top_excerpts}"
            ),
            "provider": "retrieval-fallback",
            "web_used": bool(web_snippets),
        }

    # Honest offline extractive summary when no API key is configured
    top_excerpts = "\n".join(
        f"- **[{c.file_name} - Trang {c.page_number}]**: {c.text[:320]}..."
        for c in chunks[:3]
    )
    return {
        "answer": (
            "ℹ️ *Chưa cấu hình `GEMINI_API_KEY` hoặc `ANTHROPIC_API_KEY` trong `.env`. "
            "Hệ thống hiển thị trực tiếp các đoạn trích phù hợp nhất từ Hybrid Search (BM25 + Vector):*\n\n"
            f"{top_excerpts}"
        ),
        "provider": "extractive-offline",
        "web_used": bool(web_snippets),
    }
