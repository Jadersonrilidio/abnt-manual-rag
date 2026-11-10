"""Application paths and model settings."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_DIR / "pdfs" / "manual_de_normalizacao_abnt.pdf"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDINGS_FILE = PROJECT_DIR / "embeddings_local.pkl"
CHUNKS_FILE = PROJECT_DIR / "chunks_local.pkl"
INDEX_FILE = PROJECT_DIR / "faiss_local.index"
