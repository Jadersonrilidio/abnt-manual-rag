"""Generate normalized local text embeddings."""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME


class EmbeddingService:
    """Load and use the sentence-transformer embedding model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_text(self, text: str) -> list[float]:
        """Generate a normalized embedding for a single text."""
        embedding = self.model.encode(
            text.replace("\n", " "),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embedding, dtype=np.float32).tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized local embeddings for a batch of texts."""
        if not texts:
            raise ValueError("Cannot create embeddings from an empty text list")
        logging.info("Generating local embeddings for %d chunks.", len(texts))
        embeddings = self.model.encode(
            [text.replace("\n", " ") for text in texts],
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32).tolist()
