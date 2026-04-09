from rich.console import Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.text import Text

from percy.theme import CATPPUCCIN_MOCHA


def build_stream_renderable(reasoning_text: str, response_text: str):
    renderables = []

    if reasoning_text.strip():
        renderables.append(Markdown(reasoning_text, style=CATPPUCCIN_MOCHA["sky"]))

    if response_text.strip():
        if renderables:
            renderables.append(Text())
        renderables.append(Markdown(response_text, style=CATPPUCCIN_MOCHA["text"]))

    if not renderables:
        renderables.append(Text("Thinking...", style=CATPPUCCIN_MOCHA["overlay0"]))

    return Group(*renderables)


def build_user_renderable(prompt: str):
    return Padding(Markdown(prompt, style=CATPPUCCIN_MOCHA["text"]), (1, 1))
