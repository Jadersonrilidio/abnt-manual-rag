"""Retrieve relevant excerpts from the workspace's local ABNT vector index."""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.manual_index import ManualIndex  # noqa: E402


def positive_integer(value: str) -> int:
    """Parse a positive integer argument."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


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
    manual_index = ManualIndex()
    indices, scores = manual_index.search(args.query, args.top_k)
    chunks = manual_index.chunks

    for rank, (chunk_index, score) in enumerate(zip(indices, scores), start=1):
        if 0 <= chunk_index < len(chunks):
            print(f"[Trecho {rank} | similaridade {score:.3f}]")
            print(chunks[int(chunk_index)])
            print()


if __name__ == "__main__":
    main()
