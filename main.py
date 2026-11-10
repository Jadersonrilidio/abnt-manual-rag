"""Run the interactive ABNT manual assistant."""

import asyncio

from src.application import RagApplication


async def main() -> None:
    """Start the interactive assistant."""
    await RagApplication().run()


if __name__ == "__main__":
    asyncio.run(main())
