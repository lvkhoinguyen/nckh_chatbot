"""Django management command to ingest documents from a local directory into the RAG store."""

from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.files import File
from django.core.management.base import BaseCommand, CommandError

from home.models import Document
from home.services.document_loader import SUPPORTED_EXTENSIONS
from home.services.rag_pipeline import ingest_document


class Command(BaseCommand):
    help = "Ingest supported documents (.pdf, .docx, .txt, .md) from a folder into the RAG index."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "folder",
            nargs="?",
            default="pdfs",
            help="Path to the folder containing documents (default: pdfs).",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Re-ingest files even if a processed document with the same basename already exists.",
        )

    def handle(self, *args, **options) -> None:
        folder_arg = Path(options["folder"])
        target_dir = folder_arg if folder_arg.is_absolute() else Path(settings.BASE_DIR) / folder_arg
        if not target_dir.is_dir():
            raise CommandError(f"Directory not found: {target_dir}")

        force = bool(options["force"])
        existing_basenames = {
            Path(name).name
            for name in Document.objects.filter(is_processed=True).values_list("document", flat=True)
        }

        files = sorted(
            p for p in target_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        ingested = 0
        skipped = 0
        for file_path in files:
            if not force and file_path.name in existing_basenames:
                skipped += 1
                self.stdout.write(f"Skipped (already indexed): {file_path.name}")
                continue

            if force:
                for old in Document.objects.all():
                    if Path(old.document.name).name == file_path.name:
                        old.delete()

            with file_path.open("rb") as fh:
                doc = Document(description=f"Imported from {target_dir.name}")
                doc.document.save(file_path.name, File(fh), save=True)

            try:
                chunk_count = ingest_document(doc)
                ingested += 1
                self.stdout.write(self.style.SUCCESS(f"Ingested {file_path.name} ({chunk_count} chunks)"))
            except Exception as exc:
                saved_path = doc.document.path if doc.document else None
                doc.delete()
                if saved_path and Path(saved_path).exists():
                    Path(saved_path).unlink()
                self.stderr.write(self.style.ERROR(f"Failed {file_path.name}: {exc}"))

        self.stdout.write(self.style.SUCCESS(f"Done. Ingested: {ingested}, Skipped: {skipped}"))
