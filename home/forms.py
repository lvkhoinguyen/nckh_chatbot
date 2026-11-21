from pathlib import Path

from django import forms

from home.models import Answer, Document
from home.services.document_loader import SUPPORTED_EXTENSIONS

MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB


class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ("description", "document")

    def clean_document(self):
        uploaded = self.cleaned_data.get("document")
        if not uploaded:
            return uploaded

        ext = Path(uploaded.name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise forms.ValidationError(f"Định dạng '{ext or 'không xác định'}' không được hỗ trợ. Chỉ chấp nhận: {allowed}.")

        if uploaded.size > MAX_UPLOAD_SIZE_BYTES:
            max_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
            raise forms.ValidationError(f"Kích thước tệp vượt quá giới hạn {max_mb} MB.")

        return uploaded


class AnswerForm(forms.ModelForm):
    class Meta:
        model = Answer
        fields = ("ask_content", "answer_content")
