"""Coordinate PDF processing, embeddings, and the local vector store."""

import logging

import faiss
import numpy as np

from src.document_processor import PdfProcessor
from src.embedding_service import EmbeddingService
from src.vector_store import FaissVectorStore


class ManualIndex:
    """Load or build the manual index and retrieve relevant chunks."""

    def __init__(
        self,
        document_processor: PdfProcessor | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: FaissVectorStore | None = None,
    ) -> None:
        self.document_processor = document_processor or PdfProcessor()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or FaissVectorStore()
        self._embeddings: list[list[float]] | None = None
        self._chunks: list[str] | None = None
        self._index: faiss.IndexFlatIP | None = None

    @property
    def chunks(self) -> list[str]:
        """Return the indexed source chunks, loading or building the index first."""
        self.ensure_loaded()
        if self._chunks is None:
            raise RuntimeError("The manual index has no chunks.")
        return self._chunks

    def ensure_loaded(self) -> None:
        """Load the local cache or build it from the configured PDF."""
        if self._chunks is not None and self._index is not None:
            return

        embeddings, chunks, index = self.vector_store.load()
        if chunks is not None and index is not None:
            self._embeddings = embeddings
            self._chunks = chunks
            self._index = index
            logging.info("Local embeddings and FAISS index loaded.")
            return

        logging.info("Local files not found. Processing the PDF...")
        text = self.document_processor.extract_text()
        chunks = self.document_processor.split_into_chunks(text)
        embeddings = self.embedding_service.embed_texts(chunks)
        index = self.vector_store.create_index(embeddings)
        self.vector_store.save(embeddings, chunks, index)
        self._embeddings = embeddings
        self._chunks = chunks
        self._index = index
        logging.info("Local embeddings and FAISS index saved.")

    def search(self, query: str, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """Search the manual index using an embedded query."""
        self.ensure_loaded()
        if self._index is None:
            raise RuntimeError("The manual index is not available.")
        query_embedding = self.embedding_service.embed_text(query)
        return self.vector_store.search(self._index, query_embedding, k)
