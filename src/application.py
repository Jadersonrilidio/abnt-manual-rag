"""Interactive console application for the ABNT manual assistant."""

import asyncio
import logging

from copilot import CopilotClient

from src.copilot_assistant import CopilotAssistant
from src.manual_index import ManualIndex


class RagApplication:
    """Run the interactive question-and-answer loop."""

    async def run(self) -> None:
        """Load the index, authenticate Copilot, and answer user questions."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        manual_index = ManualIndex()
        manual_index.ensure_loaded()

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
            assistant = CopilotAssistant(manual_index)
            print("Ask about the ABNT manual (or type 'sair' to quit):")
            while True:
                query = (await asyncio.to_thread(input, ">> ")).strip()
                if query.lower() == "sair":
                    break
                if not query:
                    continue
                answer = await assistant.answer(query, session)
                print("\nAnswer:\n", answer)
        finally:
            await client.stop()
