from django.contrib import admin
from home.models import Answer, Document, DocumentChunk, ProcessedDocument


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("id", "document", "description", "is_processed", "uploaded_by", "uploaded_at")
    list_filter = ("is_processed",)


@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ("id", "document", "page_number", "chunk_index", "created_at")


@admin.register(Answer)
class AnswerAdmin(admin.ModelAdmin):
    list_display = ("id", "ask_content", "uploaded_by", "ask_at")


admin.site.register(ProcessedDocument)
