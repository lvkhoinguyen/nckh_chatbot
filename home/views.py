import json
import logging
import os
from pathlib import Path
from typing import List

from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_POST

from home.forms import DocumentForm
from home.models import Answer, Document, DocumentChunk
from home.services.rag_pipeline import answer_question, ingest_document

logger = logging.getLogger("django")
MAX_CHAT_HISTORY = 10


def admin_check(user) -> bool:
    return bool(user.is_authenticated and user.is_staff)


def _get_session_history(request) -> List[tuple[str, str]]:
    raw = request.session.get("chat_history", [])
    return [(str(item[0]), str(item[1])) for item in raw if isinstance(item, (list, tuple)) and len(item) == 2]


def _get_selected_doc_ids(request) -> List[int]:
    raw = request.session.get("selected_doc_ids", [])
    return [int(x) for x in raw if str(x).isdigit()]


def chatGoD(request):
    """Render the chat interface and handle legacy form POST/clear_history gracefully."""
    if request.method == "POST":
        if "clear_history" in request.POST:
            request.session.pop("chat_history", None)
            messages.success(request, "Đã xóa lịch sử hội thoại.")
            return redirect("home")

        question = request.POST.get("question", "").strip()
        if not question:
            messages.warning(request, "Vui lòng nhập câu hỏi.")
            return redirect("home")

        use_web = request.POST.get("use_web_search") == "on"
        history = _get_session_history(request)
        selected_ids = _get_selected_doc_ids(request)
        result = answer_question(
            question=question,
            history=history,
            selected_doc_ids=selected_ids or None,
            use_web_search=use_web,
        )

        if result["found"]:
            history.append((question, result["answer"]))
            request.session["chat_history"] = history[-MAX_CHAT_HISTORY:]
            if request.user.is_authenticated:
                Answer.objects.create(
                    ask_content=question,
                    answer_content=result["answer"],
                    uploaded_by=request.user,
                )
        else:
            messages.warning(request, result["answer"])

        return render(
            request,
            "home/chatGoD.html",
            {
                "answer": {"ask_content": question, "answer_content": result["answer"]},
                "citations": result["citations"],
                "documents": Document.objects.filter(is_processed=True).prefetch_related("chunks").order_by("-uploaded_at"),
                "selected_ids": set(selected_ids),
                "selected_count": len(selected_ids),
                "total_chunks": DocumentChunk.objects.count(),
            },
        )

    last_answer = None
    if request.user.is_authenticated:
        last_answer = Answer.objects.filter(uploaded_by=request.user).last()
    selected_ids = _get_selected_doc_ids(request)
    return render(
        request,
        "home/chatGoD.html",
        {
            "answer": last_answer,
            "citations": [],
            "documents": Document.objects.filter(is_processed=True).prefetch_related("chunks").order_by("-uploaded_at"),
            "selected_ids": set(selected_ids),
            "selected_count": len(selected_ids),
            "total_chunks": DocumentChunk.objects.count(),
        },
    )


@require_POST
def api_chat(request):
    """JSON chat endpoint returning answer, citations, provider metadata, and updated history."""
    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "error": "Payload JSON không hợp lệ."}, status=400)

    question = str(payload.get("question", "")).strip()
    if not question:
        return JsonResponse({"ok": False, "error": "Vui lòng nhập nội dung câu hỏi."}, status=400)

    use_web_search = bool(payload.get("use_web_search", False))
    raw_doc_ids = payload.get("document_ids")
    if isinstance(raw_doc_ids, list):
        selected_ids = [int(x) for x in raw_doc_ids if str(x).isdigit()]
        request.session["selected_doc_ids"] = selected_ids
    else:
        selected_ids = _get_selected_doc_ids(request)

    history = _get_session_history(request)
    result = answer_question(
        question=question,
        history=history,
        selected_doc_ids=selected_ids or None,
        use_web_search=use_web_search,
    )

    if result["found"]:
        request.session.pop("chat_cleared", None)
        history.append((question, result["answer"]))
        request.session["chat_history"] = history[-MAX_CHAT_HISTORY:]
        if request.user.is_authenticated:
            Answer.objects.create(
                ask_content=question,
                answer_content=result["answer"],
                uploaded_by=request.user,
            )

    return JsonResponse(
        {
            "ok": True,
            "found": result["found"],
            "answer": result["answer"],
            "citations": result["citations"],
            "provider": result.get("provider", "unknown"),
            "web_used": result.get("web_used", False),
            "document_ids": selected_ids,
            "history_count": len(request.session.get("chat_history", [])),
        }
    )


@require_GET
def api_chat_history(request):
    """Return chat history from session, hydrating from recent Answer records for authenticated users."""
    history = _get_session_history(request)
    if not history and not request.session.get("chat_cleared") and request.user.is_authenticated:
        recent_rows = list(
            Answer.objects.filter(uploaded_by=request.user).order_by("-ask_at", "-id")[:MAX_CHAT_HISTORY]
        )
        history = [(row.ask_content, row.answer_content) for row in reversed(recent_rows)]

    messages_out = []
    for ask_text, answer_text in history:
        messages_out.append({"sender": "user", "message": ask_text, "citations": []})
        messages_out.append({"sender": "bot", "message": answer_text, "citations": []})

    return JsonResponse({"ok": True, "messages": messages_out})


@require_POST
def api_chat_clear(request):
    """Clear conversation history stored in Django session."""
    request.session.pop("chat_history", None)
    request.session["chat_cleared"] = True
    return JsonResponse({"ok": True, "message": "Đã xóa toàn bộ lịch sử hội thoại."})


@require_POST
def api_select_documents(request):
    """Persist selected document IDs for scoped RAG retrieval via JSON."""
    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "error": "Payload JSON không hợp lệ."}, status=400)

    raw_ids = payload.get("document_ids")
    if not isinstance(raw_ids, list):
        return JsonResponse({"ok": False, "error": "Trường 'document_ids' phải là danh sách."}, status=400)

    selected_ids = []
    seen = set()
    for item in raw_ids:
        if isinstance(item, bool) or not str(item).isdigit():
            return JsonResponse({"ok": False, "error": f"ID tài liệu không hợp lệ: {item}"}, status=400)
        doc_id = int(item)
        if doc_id not in seen:
            seen.add(doc_id)
            selected_ids.append(doc_id)

    request.session["selected_doc_ids"] = selected_ids
    return JsonResponse({"ok": True, "selected_doc_ids": selected_ids})


@user_passes_test(admin_check, login_url="home")
def upload(request):
    """Manage uploaded PDF/DOCX/TXT documents and trigger RAG chunking/embedding ingestion."""
    if request.method == "POST":
        if "delete_document" in request.POST:
            try:
                document_id = request.POST.get("id")
                document = get_object_or_404(Document, id=document_id)
                file_path = document.document.path if document.document else None
                document.delete()
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                messages.success(request, "Tài liệu đã được xóa thành công!")
            except Exception as exc:
                logger.exception("Error deleting document: %s", exc)
                messages.error(request, "Có lỗi khi xóa tài liệu.")
            return redirect("upload")

        if "update_note" in request.POST:
            try:
                document_id = request.POST.get("id")
                document = get_object_or_404(Document, id=document_id)
                document.description = request.POST.get("input-req", "").strip()
                document.save(update_fields=["description"])
                messages.success(request, "Cập nhật mô tả thành công!")
            except Exception as exc:
                logger.exception("Error updating note: %s", exc)
                messages.error(request, "Có lỗi khi cập nhật mô tả.")
            return redirect("upload")

        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            document = form.save(commit=False)
            document.uploaded_by = request.user
            document.save()
            try:
                chunk_count = ingest_document(document)
                messages.success(request, f"Tải lên và lập chỉ mục thành công ({chunk_count} đoạn văn bản)!")
            except Exception as exc:
                logger.exception("Error ingesting document: %s", exc)
                file_path = document.document.path if document.document else None
                document.delete()
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                messages.error(request, f"Không thể xử lý tài liệu: {exc}")
            return redirect("upload")

        error_list = [err for field_errs in form.errors.values() for err in field_errs]
        messages.error(
            request,
            " ".join(error_list) if error_list else "Biểu mẫu tải lên không hợp lệ. Vui lòng chọn tệp hợp lệ.",
        )
        return redirect("upload")

    documents = Document.objects.prefetch_related("chunks").order_by("-uploaded_at")
    return render(
        request,
        "admin/uploadManage.html",
        {
            "documents": documents,
            "total_chunks": DocumentChunk.objects.count(),
        },
    )


def select_files(request):
    """Allow users to choose which processed documents are scoped during RAG retrieval."""
    if request.method == "POST":
        if "clear_selection" in request.POST:
            request.session.pop("selected_doc_ids", None)
            messages.success(request, "Đã bỏ lọc: Chatbot sẽ tìm trên toàn bộ tài liệu.")
            return redirect("select_file")

        raw_ids = request.POST.getlist("selected_files")
        selected_ids = [int(x) for x in raw_ids if str(x).isdigit()]
        request.session["selected_doc_ids"] = selected_ids
        messages.success(request, f"Đã lưu phạm vi tìm kiếm ({len(selected_ids)} tài liệu).")
        return redirect("select_file")

    files = Document.objects.filter(is_processed=True).order_by("-uploaded_at")
    selected_ids = set(_get_selected_doc_ids(request))
    return render(
        request,
        "home/select_files.html",
        {
            "files": files,
            "selected_ids": selected_ids,
        },
    )


def logout_view(request):
    logout(request)
    messages.success(request, "Đăng xuất thành công.")
    return redirect("login")


@login_required(login_url="login")
@user_passes_test(admin_check, login_url="home")
def account(request):
    users = User.objects.all().order_by("id")
    if request.method == "POST":
        if "delete_account" in request.POST:
            account_id = request.POST.get("id")
            acc = get_object_or_404(User, id=account_id)
            if acc == request.user:
                messages.error(request, "Không thể tự xóa tài khoản đang đăng nhập.")
                return redirect("account")
            acc.delete()
            messages.success(request, "Tài khoản đã được xóa thành công!")
            return redirect("account")

        if "update_auth" in request.POST:
            try:
                account_id = request.POST.get("id")
                acc = get_object_or_404(User, id=account_id)
                role = request.POST.get("newauth")
                if role == "Superadmin":
                    acc.is_superuser = True
                    acc.is_staff = True
                elif role == "Staff":
                    acc.is_superuser = False
                    acc.is_staff = True
                else:
                    acc.is_superuser = False
                    acc.is_staff = False
                acc.save(update_fields=["is_superuser", "is_staff"])
                messages.success(request, "Cập nhật phân quyền thành công!")
            except Exception as exc:
                logger.exception("Error updating authority: %s", exc)
                messages.error(request, "Có lỗi xảy ra khi cập nhật phân quyền.")
            return redirect("account")

    return render(request, "admin/account.html", {"users": users})


def register_view(request):
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        confirm_password = request.POST.get("confirm_password", "")

        if not username:
            messages.error(request, "Vui lòng nhập tên tài khoản.")
            return render(request, "home/register.html")

        if password != confirm_password:
            messages.error(request, "Mật khẩu không khớp!")
            return render(request, "home/register.html")

        if User.objects.filter(username=username).exists():
            messages.error(request, "Tên tài khoản đã tồn tại!")
            return render(request, "home/register.html")

        try:
            validate_password(password)
        except ValidationError as exc:
            messages.error(request, " ".join(exc.messages))
            return render(request, "home/register.html")

        user = User.objects.create_user(username=username, password=password)
        login(request, user)
        messages.success(request, "Đăng ký thành công!")
        return redirect("home")

    return render(request, "home/register.html")


def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("home")
        messages.error(request, "Tên tài khoản hoặc mật khẩu không đúng.")

    return render(request, "home/login.html")
