"""Create, persist, load, and search the local FAISS vector index."""

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np

from src.config import CHUNKS_FILE, EMBEDDINGS_FILE, INDEX_FILE


class FaissVectorStore:
    """Manage a cosine-similarity FAISS index and its local cache."""

    def __init__(
        self,
        embeddings_file: Path = EMBEDDINGS_FILE,
        chunks_file: Path = CHUNKS_FILE,
        index_file: Path = INDEX_FILE,
    ) -> None:
        self.embeddings_file = embeddings_file
        self.chunks_file = chunks_file
        self.index_file = index_file

    def load(
        self,
    ) -> tuple[
        list[list[float]] | None,
        list[str] | None,
        faiss.IndexFlatIP | None,
    ]:
        """Load the local cache, or return empty values when it does not exist."""
        if (
            self.embeddings_file.exists()
            and self.chunks_file.exists()
            and self.index_file.exists()
        ):
            logging.info("Loading local embeddings and FAISS index.")
            with self.embeddings_file.open("rb") as file:
                embeddings = pickle.load(file)
            with self.chunks_file.open("rb") as file:
                chunks = pickle.load(file)
            index = faiss.read_index(str(self.index_file))
            return embeddings, chunks, index
        logging.info("Local embedding cache not found.")
        return None, None, None

    def save(
        self,
        embeddings: list[list[float]],
        chunks: list[str],
        index: faiss.IndexFlatIP,
    ) -> None:
        """Persist local embeddings, their source chunks, and the FAISS index."""
        logging.info("Saving local embeddings and FAISS index.")
        with self.embeddings_file.open("wb") as file:
            pickle.dump(embeddings, file)
        with self.chunks_file.open("wb") as file:
            pickle.dump(chunks, file)
        faiss.write_index(index, str(self.index_file))

    @staticmethod
    def create_index(embeddings: list[list[float]]) -> faiss.IndexFlatIP:
        """Create a cosine-similarity index from normalized embeddings."""
        if not embeddings:
            raise ValueError("Cannot create a FAISS index without embeddings")
        logging.info("Creating FAISS cosine-similarity index.")
        vectors = np.asarray(embeddings, dtype=np.float32)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    @staticmethod
    def search(
        index: faiss.IndexFlatIP, query_embedding: list[float], k: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the nearest chunk indices and cosine-similarity scores."""
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if index.ntotal == 0:
            raise ValueError("Cannot search an empty FAISS index")
        logging.info("Searching FAISS index.")
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = index.search(query, min(k, index.ntotal))
        return indices[0], distances[0]
