"""Extract and split text from the source PDF."""

import logging
import re
from pathlib import Path

import PyPDF2

from src.config import PDF_PATH


class PdfProcessor:
    """Extract text from the configured PDF and split it into chunks."""

    def __init__(self, pdf_path: Path = PDF_PATH) -> None:
        self.pdf_path = pdf_path

    def extract_text(self) -> str:
        """Extract text from the PDF and fail explicitly if none is available."""
        logging.info("Extracting text from PDF: %s", self.pdf_path)
        with self.pdf_path.open("rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        if not text.strip():
            raise ValueError(f"No extractable text found in PDF: {self.pdf_path}")
        return text

    @staticmethod
    def split_into_chunks(text: str, max_chunk_size: int = 500) -> list[str]:
        """Split text into bounded chunks, preferring sentence boundaries."""
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be greater than zero")

        logging.info("Splitting text into chunks.")
        sentences = re.split(r"(?<=[.?!])\s+", text.strip())
        chunks: list[str] = []
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                chunks.extend(
                    sentence[start : start + max_chunk_size].strip()
                    for start in range(0, len(sentence), max_chunk_size)
                )
            elif not current_chunk:
                current_chunk = sentence
            elif len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk)
        logging.info("Created %d text chunks.", len(chunks))
        return chunks
