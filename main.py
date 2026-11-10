"""Build a local FAISS index from the ABNT PDF and query it with GitHub Copilot."""

import asyncio
import logging
import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
import PyPDF2
from copilot import CopilotClient
from copilot.session import CopilotSession
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent
PDF_PATH = PROJECT_DIR / "pdfs" / "manual_de_normalizacao_abnt.pdf"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDINGS_FILE = PROJECT_DIR / "embeddings_local.pkl"
CHUNKS_FILE = PROJECT_DIR / "chunks_local.pkl"
INDEX_FILE = PROJECT_DIR / "faiss_local.index"


async def main() -> None:
    """Load or build the local vector index and answer questions with Copilot."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    embeddings, chunks, index = load_embeddings()
    if embeddings is None or chunks is None or index is None:
        logging.info("Local embeddings not found. Processing the PDF...")
        text = extract_text_from_pdf(str(PDF_PATH))
        chunks = split_text_into_chunks(text)
        embeddings = get_embeddings(chunks, embedding_model)
        index = create_faiss_index(embeddings)
        save_embeddings(embeddings, chunks, index)
        logging.info("Local embeddings and FAISS index saved.")
    else:
        logging.info("Local embeddings and FAISS index loaded.")

    client = CopilotClient()
    await client.start()
    try:
        auth_status = await client.get_auth_status()
        if not auth_status.isAuthenticated:
            raise RuntimeError(
                "GitHub Copilot is not authenticated for the SDK runtime. "
                "Install the GitHub Copilot CLI and run `copilot auth login`, "
                "then restart this app."
            )
        session = await client.create_session(model="auto", available_tools=[])
        print("Ask about the ABNT manual (or type 'sair' to quit):")
        while True:
            query = (await asyncio.to_thread(input, ">> ")).strip()
            if query.lower() == "sair":
                break
            if not query:
                continue
            answer = await answer_query(query, index, chunks, embedding_model, session)
            print("\nAnswer:\n", answer)
    finally:
        await client.stop()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF and fail explicitly if no text is available."""
    logging.info("Extracting text from PDF: %s", pdf_path)
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
    if not text.strip():
        raise ValueError(f"No extractable text found in PDF: {pdf_path}")
    return text


def split_text_into_chunks(text: str, max_chunk_size: int = 500) -> List[str]:
    """Split text into bounded chunks, preferring sentence boundaries."""
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than zero")

    logging.info("Splitting text into chunks.")
    sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    chunks: List[str] = []
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


def get_embedding(text: str, model: SentenceTransformer) -> List[float]:
    """Generate a normalized embedding locally for a single text."""
    embedding = model.encode(
        text.replace("\n", " "),
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(embedding, dtype=np.float32).tolist()


def get_embeddings(
    texts: List[str], model: SentenceTransformer
) -> List[List[float]]:
    """Generate normalized local embeddings for a batch of texts."""
    if not texts:
        raise ValueError("Cannot create embeddings from an empty text list")
    logging.info("Generating local embeddings for %d chunks.", len(texts))
    embeddings = model.encode(
        [text.replace("\n", " ") for text in texts],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32).tolist()


def create_faiss_index(embeddings: List[List[float]]) -> faiss.IndexFlatIP:
    """Create a cosine-similarity FAISS index from normalized embeddings."""
    if not embeddings:
        raise ValueError("Cannot create a FAISS index without embeddings")
    logging.info("Creating FAISS cosine-similarity index.")
    vectors = np.asarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_embeddings(
    embeddings: List[List[float]],
    chunks: List[str],
    index: faiss.IndexFlatIP,
    embeddings_file: Path = EMBEDDINGS_FILE,
    chunks_file: Path = CHUNKS_FILE,
    index_file: Path = INDEX_FILE,
) -> None:
    """Persist local embeddings, their source chunks, and the FAISS index."""
    logging.info("Saving local embeddings and FAISS index.")
    with open(embeddings_file, "wb") as file:
        pickle.dump(embeddings, file)
    with open(chunks_file, "wb") as file:
        pickle.dump(chunks, file)
    faiss.write_index(index, str(index_file))


def load_embeddings(
    embeddings_file: Path = EMBEDDINGS_FILE,
    chunks_file: Path = CHUNKS_FILE,
    index_file: Path = INDEX_FILE,
) -> Tuple[
    Optional[List[List[float]]],
    Optional[List[str]],
    Optional[faiss.IndexFlatIP],
]:
    """Load the local cache, or return empty values when it does not exist."""
    if embeddings_file.exists() and chunks_file.exists() and index_file.exists():
        logging.info("Loading local embeddings and FAISS index.")
        with open(embeddings_file, "rb") as file:
            embeddings = pickle.load(file)
        with open(chunks_file, "rb") as file:
            chunks = pickle.load(file)
        index = faiss.read_index(str(index_file))
        return embeddings, chunks, index
    logging.info("Local embedding cache not found.")
    return None, None, None


def search_index(
    index: faiss.IndexFlatIP, query_embedding: List[float], k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the nearest chunk indices and cosine-similarity scores."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if index.ntotal == 0:
        raise ValueError("Cannot search an empty FAISS index")
    logging.info("Searching FAISS index.")
    query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(query, min(k, index.ntotal))
    return indices[0], distances[0]


async def answer_query(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: List[str],
    embedding_model: SentenceTransformer,
    session: CopilotSession,
    k: int = 5,
) -> str:
    """Retrieve relevant PDF chunks locally and ask Copilot to answer from them."""
    logging.info("Retrieving context for user question.")
    query_embedding = get_embedding(query, embedding_model)
    indices, _ = search_index(index, query_embedding, k)
    context = "\n\n".join(
        f"[Trecho {rank}]\n{chunks[int(chunk_index)]}"
        for rank, chunk_index in enumerate(indices, start=1)
        if 0 <= chunk_index < len(chunks)
    )
    prompt = (
        "Responda em português usando somente o contexto do manual abaixo. "
        "Se o contexto não trouxer a resposta, diga que não encontrou essa "
        "informação no manual. Cite os trechos relevantes pelo número entre "
        "colchetes.\n\n"
        f"Contexto:\n{context}\n\nPergunta: {query}"
    )
    response = await session.send_and_wait(prompt)
    if response is None or response.data.content is None:
        raise RuntimeError("Copilot did not return an answer.")
    return response.data.content


if __name__ == "__main__":
    asyncio.run(main())
