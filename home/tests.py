"""Automated Django tests for document ingestion, hybrid RAG retrieval, chat API, and admin permissions."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from docx import Document as DocxDocument

from home.forms import MAX_UPLOAD_SIZE_BYTES, DocumentForm
from home.models import Answer, Document, DocumentChunk
from home.services.chunker import build_document_chunks, split_text_safely
from home.services.document_loader import LoadedPage, load_document_pages
from home.services.llm_client import build_rag_prompt
from home.services.rag_pipeline import answer_question, ingest_document
from home.services.retriever import ChunkCandidate, hybrid_retrieve


def _deterministic_embed(texts):
    """Lightweight deterministic bag-of-words projection for fast offline tests."""
    vocab = ["học", "bổng", "vnpay", "thanh", "toán", "rag", "python", "django", "tuyển", "sinh", "khác"]
    vecs = []
    for text in texts:
        lower = text.lower()
        vec = np.array([float(lower.count(tok)) for tok in vocab], dtype=np.float32)
        norm = np.linalg.norm(vec)
        vecs.append(vec / norm if norm > 0 else vec)
    return np.asarray(vecs, dtype=np.float32)


class ChunkerTests(TestCase):
    def test_split_text_never_infinite_loops_with_punctuation_inside_overlap(self):
        # Punctuation positioned early so end - overlap <= start would stall naive chunkers
        text = "A. " + ("từ khóa kiểm thử tiếng Việt " * 40)
        chunks = split_text_safely(text, chunk_size=80, overlap=75)
        self.assertGreater(len(chunks), 3)
        self.assertLess(len(chunks), 200)

    def test_build_document_chunks_attaches_page_header(self):
        pages = [LoadedPage(page_number=2, text="Quy định xét học bổng khuyến khích học tập học kỳ 1.")]
        chunks = build_document_chunks(pages, file_label="quy_dinh.pdf", chunk_size=200, overlap=30)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_number, 2)
        self.assertIn("[quy_dinh.pdf | Trang 2]", chunks[0].text)


class DocumentLoaderTests(TestCase):
    def test_load_docx_extracts_paragraphs_and_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            docx_path = Path(tmp) / "sample.docx"
            doc = DocxDocument()
            doc.add_paragraph("Thông báo tuyển sinh đại học chính quy năm 2026.")
            table = doc.add_table(rows=1, cols=2)
            table.cell(0, 0).text = "Ngành"
            table.cell(0, 1).text = "Khoa học máy tính"
            doc.save(docx_path)

            pages = load_document_pages(docx_path)
            self.assertEqual(len(pages), 1)
            self.assertIn("tuyển sinh", pages[0].text)
            self.assertIn("Khoa học máy tính", pages[0].text)

    def test_unsupported_extension_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "binary.exe"
            bad.write_bytes(b"MZ")
            with self.assertRaises(ValueError):
                load_document_pages(bad)


class HybridRetrieverTests(TestCase):
    def test_hybrid_retrieve_handles_fewer_chunks_than_top_k_and_ranks_correctly(self):
        texts = [
            "Hướng dẫn tích hợp cổng thanh toán VNPay Sandbox cho website bán hàng.",
            "Chính sách cấp học bổng khuyến khích học tập dành cho sinh viên.",
        ]
        vecs = _deterministic_embed(texts)
        candidates = [
            ChunkCandidate(chunk_id=1, document_id=10, file_name="vnpay.pdf", page_number=1, text=texts[0], embedding=vecs[0]),
            ChunkCandidate(chunk_id=2, document_id=11, file_name="hocbong.pdf", page_number=3, text=texts[1], embedding=vecs[1]),
        ]
        results = hybrid_retrieve("điều kiện học bổng sinh viên", candidates, top_k=5, embed_fn=_deterministic_embed)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].file_name, "hocbong.pdf")
        self.assertEqual(results[0].page_number, 3)

    def test_build_rag_prompt_includes_context_and_history(self):
        texts = ["Nội dung về Django RAG."]
        vecs = _deterministic_embed(texts)
        candidates = [ChunkCandidate(1, 1, "doc.pdf", 1, texts[0], vecs[0])]
        results = hybrid_retrieve("Django RAG", candidates, top_k=1, embed_fn=_deterministic_embed)
        prompt = build_rag_prompt("Django RAG là gì?", results, history=[("Chào", "Xin chào!")])
        self.assertIn("Nội dung về Django RAG.", prompt)
        self.assertIn("Người dùng: Chào", prompt)


class RagViewsAndApiTests(TestCase):
    def setUp(self):
        self.tmp_media = tempfile.TemporaryDirectory()
        self.override = override_settings(MEDIA_ROOT=self.tmp_media.name)
        self.override.enable()

        self.user = User.objects.create_user(username="student1", password="StrongPassword123!")
        self.staff = User.objects.create_superuser(username="admin1", password="StrongPassword123!", email="a@b.com")

        self.doc1 = Document.objects.create(
            description="Tài liệu học bổng",
            document=SimpleUploadedFile("hoc_bong.txt", "Học bổng loại xuất sắc yêu cầu GPA >= 3.6.".encode("utf-8")),
            uploaded_by=self.staff,
        )
        self.doc2 = Document.objects.create(
            description="Tài liệu thanh toán",
            document=SimpleUploadedFile("vnpay.txt", "Thanh toán VNPay sử dụng chữ ký HMAC-SHA512.".encode("utf-8")),
            uploaded_by=self.staff,
        )
        ingest_document(self.doc1, embed_fn=_deterministic_embed)
        ingest_document(self.doc2, embed_fn=_deterministic_embed)

    def tearDown(self):
        self.override.disable()
        self.tmp_media.cleanup()

    @patch("home.services.retriever.get_default_embedder", return_value=_deterministic_embed)
    @patch("home.services.rag_pipeline.get_default_embedder", return_value=_deterministic_embed)
    def test_api_chat_session_history_citations_and_clear(self, _mock_rp, _mock_ret):
        client = Client()
        client.force_login(self.user)

        bad = client.post(reverse("api_chat"), data=json.dumps({"question": "   "}), content_type="application/json")
        self.assertEqual(bad.status_code, 400)

        res = client.post(
            reverse("api_chat"),
            data=json.dumps({"question": "Điều kiện học bổng là gì?"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body["ok"])
        self.assertTrue(body["found"])
        self.assertEqual(body["history_count"], 1)
        self.assertGreaterEqual(len(body["citations"]), 1)
        self.assertEqual(body["citations"][0]["file_name"], "hoc_bong.txt")
        self.assertEqual(Answer.objects.filter(uploaded_by=self.user).count(), 1)

        clear_res = client.post(reverse("api_chat_clear"))
        self.assertEqual(clear_res.status_code, 200)
        self.assertNotIn("chat_history", client.session)

    @patch("home.services.retriever.get_default_embedder", return_value=_deterministic_embed)
    @patch("home.services.rag_pipeline.get_default_embedder", return_value=_deterministic_embed)
    def test_select_files_filters_rag_scope(self, _mock_rp, _mock_ret):
        client = Client()
        client.post(reverse("select_file"), {"selected_files": [str(self.doc2.id)]})
        self.assertEqual(client.session.get("selected_doc_ids"), [self.doc2.id])

        res = client.post(
            reverse("api_chat"),
            data=json.dumps({"question": "học bổng"}),
            content_type="application/json",
        )
        body = res.json()
        self.assertTrue(body["ok"])
        # Because scope is restricted to doc2, citations only come from doc2
        self.assertTrue(all(c["document_id"] == self.doc2.id for c in body["citations"]))

    def test_non_staff_redirected_from_upload_and_account(self):
        client = Client()
        client.force_login(self.user)
        self.assertEqual(client.get(reverse("upload")).status_code, 302)
        self.assertEqual(client.get(reverse("account")).status_code, 302)

    def test_answer_question_custom_generator_and_empty_db(self):
        DocumentChunk.objects.all().delete()
        empty_res = answer_question("test", embed_fn=_deterministic_embed)
        self.assertFalse(empty_res["found"])

    def test_message_error_tag_maps_to_bootstrap_danger(self):
        self.assertEqual(settings.MESSAGE_TAGS.get(messages.ERROR), "danger")

    def test_document_form_validates_extension_and_size(self):
        bad_ext = DocumentForm(
            data={"description": "Bad"},
            files={"document": SimpleUploadedFile("malware.exe", b"MZ")},
        )
        self.assertFalse(bad_ext.is_valid())
        self.assertIn("document", bad_ext.errors)

        oversized_file = SimpleUploadedFile("big.txt", b"a")
        oversized_file.size = MAX_UPLOAD_SIZE_BYTES + 1
        too_big = DocumentForm(
            data={"description": "Too big"},
            files={"document": oversized_file},
        )
        self.assertFalse(too_big.is_valid())
        self.assertIn("document", too_big.errors)

        valid_txt = DocumentForm(
            data={"description": "Valid"},
            files={"document": SimpleUploadedFile("notes.txt", "Nội dung hợp lệ.".encode("utf-8"))},
        )
        self.assertTrue(valid_txt.is_valid())

    @patch("home.services.rag_pipeline.get_default_embedder", return_value=_deterministic_embed)
    def test_ingest_pdfs_command_ingests_and_skips_duplicates(self, _mock_embedder):
        with tempfile.TemporaryDirectory() as folder:
            sample = Path(folder) / "new_policy.txt"
            sample.write_text("Quy định mới về học bổng sinh viên khoa học máy tính.", encoding="utf-8")
            ignored = Path(folder) / "ignore.exe"
            ignored.write_bytes(b"binary")

            out1 = io.StringIO()
            call_command("ingest_pdfs", folder, stdout=out1)
            self.assertIn("Ingested: 1, Skipped: 0", out1.getvalue())
            self.assertTrue(Document.objects.filter(document__icontains="new_policy").exists())

            out2 = io.StringIO()
            call_command("ingest_pdfs", folder, stdout=out2)
            self.assertIn("Ingested: 0, Skipped: 1", out2.getvalue())

    def test_api_chat_history_hydrates_from_answer_and_respects_clear(self):
        Answer.objects.create(
            ask_content="Câu hỏi đã lưu",
            answer_content="Câu trả lời đã lưu",
            uploaded_by=self.user,
        )
        client = Client()
        client.force_login(self.user)

        hist_res = client.get(reverse("api_chat_history"))
        self.assertEqual(hist_res.status_code, 200)
        msgs = hist_res.json()["messages"]
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["sender"], "user")
        self.assertEqual(msgs[0]["message"], "Câu hỏi đã lưu")
        self.assertEqual(msgs[1]["sender"], "bot")
        self.assertEqual(msgs[1]["message"], "Câu trả lời đã lưu")

        client.post(reverse("api_chat_clear"))
        after_clear = client.get(reverse("api_chat_history"))
        self.assertEqual(after_clear.json()["messages"], [])

    def test_api_select_documents_persists_and_validates(self):
        client = Client()
        bad = client.post(
            reverse("api_select_documents"),
            data=json.dumps({"document_ids": "not-a-list"}),
            content_type="application/json",
        )
        self.assertEqual(bad.status_code, 400)

        res = client.post(
            reverse("api_select_documents"),
            data=json.dumps({"document_ids": [self.doc1.id, self.doc1.id]}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["selected_doc_ids"], [self.doc1.id])
        self.assertEqual(client.session.get("selected_doc_ids"), [self.doc1.id])

        cleared = client.post(
            reverse("api_select_documents"),
            data=json.dumps({"document_ids": []}),
            content_type="application/json",
        )
        self.assertEqual(cleared.status_code, 200)
        self.assertEqual(client.session.get("selected_doc_ids"), [])
