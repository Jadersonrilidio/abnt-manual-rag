"""Answer questions using retrieved manual excerpts and GitHub Copilot."""

import logging

from copilot.session import CopilotSession

from src.manual_index import ManualIndex


class CopilotAssistant:
    """Retrieve supporting excerpts and ask Copilot to answer from them."""

    def __init__(self, manual_index: ManualIndex) -> None:
        self.manual_index = manual_index

    async def answer(
        self, query: str, session: CopilotSession, k: int = 5
    ) -> str:
        """Answer a question using only the relevant excerpts from the manual."""
        logging.info("Retrieving context for user question.")
        indices, _ = self.manual_index.search(query, k)
        chunks = self.manual_index.chunks
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
