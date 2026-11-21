from django.contrib.auth.models import User
from django.db import models


class Document(models.Model):
    description = models.CharField(max_length=255, blank=True)
    document = models.FileField(upload_to="pdfs/")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    is_processed = models.BooleanField(default=False)

    def __str__(self) -> str:
        return self.document.name or f"Document #{self.pk}"


class Answer(models.Model):
    answer_at = models.DateTimeField(auto_now_add=True)
    answer_content = models.TextField()
    ask_at = models.DateTimeField(auto_now_add=True)
    ask_content = models.TextField()
    uploaded_file = models.FileField(upload_to="", blank=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self) -> str:
        return f"Q: {self.ask_content[:50]}"


class ProcessedDocument(models.Model):
    file_name = models.CharField(max_length=255)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, null=True, blank=True)
    text_content = models.TextField()
    embeddings = models.BinaryField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"{self.file_name} - {self.uploaded_at}"


class DocumentChunk(models.Model):
    """Normalized document chunk with page citation and float32 dense embedding."""

    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    chunk_index = models.PositiveIntegerField(default=0)
    page_number = models.PositiveIntegerField(default=1)
    text = models.TextField()
    embedding = models.BinaryField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["document_id", "chunk_index"]

    def __str__(self) -> str:
        return f"{self.document.document.name} (p.{self.page_number}, #{self.chunk_index})"
