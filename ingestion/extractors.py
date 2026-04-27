from __future__ import annotations
import structlog
from pathlib import Path
from models import DocumentMetadata
from .core.interfaces import TextExtractor

log = structlog.get_logger()


class PlainTextExtractor(TextExtractor):

    SUPPORTED = {".txt", ".md", ".rst"}

    def supports(self, suffix: str) -> bool:
        return suffix in self.SUPPORTED

    def extract(self, path: str) -> list[tuple[str, dict]]:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        meta = {"source": path, "title": Path(path).stem}
        return [(text, meta)]

class PDFExtractor(TextExtractor):

    def supports(self, suffix: str) -> bool:
        return suffix == ".pdf"
    
    def extract(self, path: str) -> list[tuple[str, dict]]:
        from pypdf import PdfReader
        reader = PdfReader(path)
        results = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                meta = {"source": path, "title": Path(path).stem, "page": i + 1}
                results.append((text, meta))
        return results

class DocxExtractor(TextExtractor):

    def supports(self, suffix: str) -> bool:
        return suffix == ".docx"

    def extract(self, path: str) -> list[tuple[str, dict]]:
        from docx import Document

        doc = Document(path)
        results = []
        current_section = ""
        buffer: list[str] = []

        def flush(section: str) -> None:
            if buffer:
                text = "\n".join(buffer)
                meta = {"source": path, "title": Path(path).stem, "section": section}
                results.append((text, meta))
                buffer.clear()

        for para in doc.paragraphs:
            if para.style.name.startswith("Heading"):
                flush(current_section)
                current_section = para.text
            else:
                if para.text.strip():
                    buffer.append(para.text)
        flush(current_section)
        return results

class ExtractorRegistry:

    def __init__(self) -> None:
        self.extractors: list[TextExtractor] = []

    def register(self, extractor: TextExtractor) -> None:
        self.extractors.append(extractor)

    def extract(self, path:str) -> list[tuple[str, dict]]:
        suffix = Path(path).suffix.lower()
        for extractor in self.extractors:
            if extractor.supports(suffix):
                return extractor.extract(path)
        log.warning("No extractor found for file type", suffix=suffix, path=path)
        return []

def build_default_registry() -> ExtractorRegistry:
    registry = ExtractorRegistry()
    registry.register(PlainTextExtractor())
    registry.register(PDFExtractor())
    registry.register(DocxExtractor())
    return registry 
