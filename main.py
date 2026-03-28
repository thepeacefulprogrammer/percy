import asyncio
from enum import Enum

import typer

import text_main
import voice_main


class PercyMode(str, Enum):
    text = "text"
    voice = "voice"


app = typer.Typer(add_completion=True, invoke_without_command=True)


@app.callback()
def run(
    mode: PercyMode = typer.Option(
        PercyMode.voice,
        "--mode",
        "-m",
        help="Select Percy mode: text or voice.",
        case_sensitive=False,
    ),
):
    """Run Percy in text or voice mode."""
    if mode == PercyMode.text:
        asyncio.run(text_main.main())
    else:
        asyncio.run(voice_main.main())


if __name__ == "__main__":
    app()
