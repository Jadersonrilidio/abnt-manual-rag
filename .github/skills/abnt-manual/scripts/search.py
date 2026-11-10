"""Retrieve relevant excerpts from the workspace's local ABNT vector index."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from main import (  # noqa: E402
    EMBEDDING_MODEL_NAME,
    PDF_PATH,
    create_faiss_index,
    extract_text_from_pdf,
    get_embedding,
    get_embeddings,
    load_embeddings,
    save_embeddings,
    search_index,
    split_text_into_chunks,
)
from sentence_transformers import SentenceTransformer  # noqa: E402


def positive_integer(value: str) -> int:
    """Parse a positive integer argument."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def load_or_build_index(
    embedding_model: SentenceTransformer,
) -> Tuple[List[str], faiss.IndexFlatIP]:
    """Load cached vectors or build them from the configured PDF."""
    embeddings, chunks, index = load_embeddings()
    if embeddings is not None and chunks is not None and index is not None:
        return chunks, index

    logging.info("Building the local ABNT vector index from %s", PDF_PATH)
    text = extract_text_from_pdf(str(PDF_PATH))
    chunks = split_text_into_chunks(text)
    embeddings = get_embeddings(chunks, embedding_model)
    index = create_faiss_index(embeddings)
    save_embeddings(embeddings, chunks, index)
    return chunks, index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the local ABNT manual vector index."
    )
    parser.add_argument("--query", required=True, help="Question to search for.")
    parser.add_argument(
        "--top-k", type=positive_integer, default=5, help="Number of excerpts."
    )
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("--query must not be empty")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    chunks, index = load_or_build_index(embedding_model)
    indices, scores = search_index(
        index, get_embedding(args.query, embedding_model), args.top_k
    )

    for rank, (chunk_index, score) in enumerate(zip(indices, scores), start=1):
        if 0 <= chunk_index < len(chunks):
            print(f"[Trecho {rank} | similaridade {score:.3f}]")
            print(chunks[int(chunk_index)])
            print()


if __name__ == "__main__":
    main()
