import asyncio
import sys
import tomllib
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from agent_framework import AgentSession
from agent_framework.openai import OpenAIChatClient
from azure.identity import AzureCliCredential
from rich.console import Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.widgets import Input, Static
from ventures_agent_framework import config, tools

from memory import load_persistant_memory, persist_session_memory

CATPPUCCIN_MOCHA = {
    "rosewater": "#f5e0dc",
    "flamingo": "#f2cdcd",
    "pink": "#f5c2e7",
    "mauve": "#cba6f7",
    "red": "#f38ba8",
    "maroon": "#eba0ac",
    "peach": "#fab387",
    "yellow": "#f9e2af",
    "green": "#a6e3a1",
    "teal": "#94e2d5",
    "sky": "#89dceb",
    "sapphire": "#74c7ec",
    "blue": "#89b4fa",
    "lavender": "#b4befe",
    "text": "#cdd6f4",
    "subtext1": "#bac2de",
    "subtext0": "#a6adc8",
    "overlay2": "#9399b2",
    "overlay1": "#7f849c",
    "overlay0": "#6c7086",
    "surface2": "#585b70",
    "surface1": "#45475a",
    "surface0": "#313244",
    "base": "#1e1e2e",
    "mantle": "#181825",
    "crust": "#11111b",
}


def _get_incremental_text(seen: dict[str, str], key: str, incoming: str | None) -> str:
    if not incoming:
        return ""

    previous = seen.get(key, "")
    if not previous:
        seen[key] = incoming
        return incoming

    if incoming.startswith(previous):
        seen[key] = incoming
        return incoming[len(previous) :]

    max_overlap = min(len(previous), len(incoming))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if previous.endswith(incoming[:size]):
            overlap = size
            break

    new_text = incoming[overlap:]
    seen[key] = previous + new_text
    return new_text


def _join_stream_text(parts: dict[str, str], order: list[str]) -> str:
    return "\n\n".join(
        parts[key].strip() for key in order if parts.get(key, "").strip()
    )


def _content_event_type(content) -> str | None:
    raw = getattr(content, "raw_representation", None)
    return getattr(raw, "type", None)


@lru_cache(maxsize=1)
def _load_local_config() -> dict:
    config_path = Path(__file__).with_name("config.toml")
    if not config_path.exists():
        return {}

    with config_path.open("rb") as f:
        return tomllib.load(f)


def _usage_int(usage: dict | None, key: str) -> int:
    if not usage:
        return 0
    value = usage.get(key)
    return int(value or 0)


def _format_cost(value: float | Decimal | None) -> str:
    if value is None:
        return "—"
    return f"${float(value):,.2f}"


def _estimate_cost(usage: dict | None) -> float | None:
    if not usage:
        return None

    pricing = _load_local_config().get("pricing", {})
    if not pricing:
        return None

    input_rate = pricing.get("input_per_million")
    output_rate = pricing.get("output_per_million")
    cached_input_rate = pricing.get("cached_input_per_million", input_rate)
    reasoning_output_rate = pricing.get("reasoning_output_per_million", output_rate)

    if input_rate is None or output_rate is None:
        return None

    input_tokens = _usage_int(usage, "input_token_count")
    output_tokens = _usage_int(usage, "output_token_count")
    cached_input_tokens = _usage_int(usage, "openai.cached_input_tokens")
    reasoning_tokens = _usage_int(usage, "openai.reasoning_tokens")

    non_cached_input_tokens = max(input_tokens - cached_input_tokens, 0)
    non_reasoning_output_tokens = max(output_tokens - reasoning_tokens, 0)

    cost = 0.0
    cost += (non_cached_input_tokens / 1_000_000) * float(input_rate)
    cost += (cached_input_tokens / 1_000_000) * float(cached_input_rate)
    cost += (non_reasoning_output_tokens / 1_000_000) * float(output_rate)
    cost += (reasoning_tokens / 1_000_000) * float(reasoning_output_rate)
    return cost


def _update_session_usage_tally(
    session: AgentSession, usage: dict | None, turn_cost: float | None
) -> dict:
    tally = dict(session.state.get("_usage_tally", {}))

    for key in (
        "input_token_count",
        "output_token_count",
        "total_token_count",
        "openai.cached_input_tokens",
        "openai.reasoning_tokens",
    ):
        tally[key] = int(tally.get(key, 0)) + _usage_int(usage, key)

    tally["turn_count"] = int(tally.get("turn_count", 0)) + 1
    tally["cost_usd"] = float(tally.get("cost_usd", 0.0)) + float(turn_cost or 0.0)
    session.state["_usage_tally"] = tally
    return tally


def _build_stream_renderable(reasoning_text: str, response_text: str):
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


def _build_user_renderable(prompt: str):
    return Padding(Markdown(prompt, style=CATPPUCCIN_MOCHA["text"]), (1, 1))



def _build_usage_renderable(usage: dict | None, session_tally: dict | None = None):
    if not usage:
        return Panel(
            Text("context remaining —", style=CATPPUCCIN_MOCHA["subtext1"]),
            border_style=CATPPUCCIN_MOCHA["surface2"],
            padding=(0, 1),
            expand=True,
        )

    token_limits = _load_local_config().get("token_limits", {})
    context_window = int(token_limits.get("context_window", 0) or 0)
    total_tokens = _usage_int(usage, "total_token_count")
    turn_cost = _estimate_cost(usage)
    used_pct = (
        ((total_tokens / context_window) * 100)
        if context_window and total_tokens
        else 0
    )
    remaining_pct = max(0.0, 100.0 - used_pct) if context_window else None
    session_cost = (
        float(session_tally.get("cost_usd", 0.0)) if session_tally is not None else None
    )

    left_text = Text(style=CATPPUCCIN_MOCHA["subtext1"])
    if remaining_pct is not None:
        left_text.append(f"context remaining {remaining_pct:.1f}%")
    else:
        left_text.append("context remaining —")

    right_text = Text(style=CATPPUCCIN_MOCHA["subtext1"])
    right_text.append(f"turn {_format_cost(turn_cost)}")
    if session_cost is not None:
        right_text.append(" · ")
        right_text.append(f"session {_format_cost(session_cost)}")

    usage_row = Table.grid(expand=True)
    usage_row.add_column(ratio=1, justify="left")
    usage_row.add_column(ratio=1, justify="right")
    usage_row.add_row(left_text, right_text)

    return Panel(
        usage_row,
        border_style=CATPPUCCIN_MOCHA["surface2"],
        padding=(0, 1),
        expand=True,
    )


class PercyApp(App[None]):
    CSS = f"""
    Screen {{
        background: {CATPPUCCIN_MOCHA['base']};
        color: {CATPPUCCIN_MOCHA['text']};
    }}

    #app-shell {{
        layout: vertical;
        height: 100%;
        width: 100%;
    }}

    #conversation {{
        height: 1fr;
        overflow-y: auto;
        padding: 0 1;
        scrollbar-size-vertical: 1;
        scrollbar-color: {CATPPUCCIN_MOCHA['surface2']} {CATPPUCCIN_MOCHA['mantle']};
        scrollbar-color-hover: {CATPPUCCIN_MOCHA['overlay0']} {CATPPUCCIN_MOCHA['mantle']};
        scrollbar-color-active: {CATPPUCCIN_MOCHA['blue']} {CATPPUCCIN_MOCHA['mantle']};
    }}

    #conversation:focus {{
        border: tall {CATPPUCCIN_MOCHA['surface0']};
    }}

    #new-message-indicator {{
        height: 1;
        margin: 0 1;
        color: {CATPPUCCIN_MOCHA['blue']};
        content-align: center middle;
    }}

    #new-message-indicator.hidden {{
        display: none;
    }}

    #usage {{
        height: 3;
        margin: 0 1;
    }}

    #prompt {{
        margin: 0 1 1 1;
    }}

    Input {{
        background: {CATPPUCCIN_MOCHA['mantle']};
        color: {CATPPUCCIN_MOCHA['text']};
        border: tall {CATPPUCCIN_MOCHA['surface2']};
    }}

    Input:focus {{
        border: tall {CATPPUCCIN_MOCHA['mauve']};
    }}

    .message {{
        width: 1fr;
    }}

    .user-message {{
        background: {CATPPUCCIN_MOCHA['surface0']};
        color: {CATPPUCCIN_MOCHA['text']};
        margin: 1 0 1 0;
    }}

    .assistant-message {{
        margin: 0 0 1 0;
    }}

    .status-message {{
        color: {CATPPUCCIN_MOCHA['overlay0']};
        margin: 1 1;
    }}
    """

    BINDINGS = [
        Binding("pageup", "conversation_page_up", "Page up", show=False),
        Binding("pagedown", "conversation_page_down", "Page down", show=False),
        Binding("end", "conversation_end", "Jump to latest", show=False),
    ]

    def __init__(self, agent, session: AgentSession, memory_file: Path) -> None:
        super().__init__()
        self.agent = agent
        self.session = session
        self.memory_file = memory_file
        self.busy = False
        self.pending_new_messages = False

    def compose(self) -> ComposeResult:
        with Container(id="app-shell"):
            yield VerticalScroll(
                Static("No conversation yet.", classes="status-message", id="empty-state"),
                id="conversation",
            )
            yield Static(
                "new messages below · press End",
                id="new-message-indicator",
                classes="hidden",
            )
            yield Static(_build_usage_renderable(None), id="usage")
            yield Input(placeholder="Send a message…", id="prompt")

    def on_mount(self) -> None:
        self.query_one(Input).focus()
        self.set_interval(0.2, self._sync_scroll_state)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if self.busy:
            return

        prompt = event.value.rstrip()
        if not prompt.strip():
            event.input.value = ""
            return

        event.input.value = ""

        if prompt in ["/quit", "/exit", "exit", "quit", "q"]:
            self.exit()
            return

        await self._process_prompt(prompt)

    async def _remove_empty_state(self) -> None:
        try:
            empty_state = self.query_one("#empty-state", Static)
        except Exception:
            return
        await empty_state.remove()

    def _conversation(self) -> VerticalScroll:
        return self.query_one("#conversation", VerticalScroll)

    def _usage_widget(self) -> Static:
        return self.query_one("#usage", Static)

    def _input_widget(self) -> Input:
        return self.query_one("#prompt", Input)

    def _indicator_widget(self) -> Static:
        return self.query_one("#new-message-indicator", Static)

    def _should_autoscroll(self) -> bool:
        conversation = self._conversation()
        return conversation.scroll_y >= conversation.max_scroll_y - 2

    def _set_new_message_indicator(self, visible: bool) -> None:
        indicator = self._indicator_widget()
        if visible:
            indicator.remove_class("hidden")
        else:
            indicator.add_class("hidden")

    def _sync_scroll_state(self) -> None:
        if self._should_autoscroll():
            self.pending_new_messages = False
            self._set_new_message_indicator(False)
        else:
            self._set_new_message_indicator(self.pending_new_messages)

    def _maybe_follow_or_notify(self, was_at_bottom: bool) -> None:
        if was_at_bottom:
            self.pending_new_messages = False
            self._set_new_message_indicator(False)
            self._conversation().scroll_end(animate=False, immediate=True, x_axis=False)
        else:
            self.pending_new_messages = True
            self._set_new_message_indicator(True)

    async def _append_widget(self, widget: Static, *, autoscroll: bool = True) -> None:
        await self._remove_empty_state()
        conversation = self._conversation()
        await conversation.mount(widget)
        if autoscroll:
            self._maybe_follow_or_notify(True)

    async def _process_prompt(self, prompt: str) -> None:
        self.busy = True
        input_widget = self._input_widget()
        input_widget.disabled = True
        was_at_bottom = self._should_autoscroll()

        user_widget = Static(_build_user_renderable(prompt), classes="message user-message")
        assistant_widget = Static(
            _build_stream_renderable("", ""), classes="message assistant-message"
        )

        await self._append_widget(user_widget, autoscroll=False)
        await self._append_widget(assistant_widget, autoscroll=False)
        self._maybe_follow_or_notify(was_at_bottom)

        usage_content = None
        reasoning_seen: dict[str, str] = {}
        response_seen: dict[str, str] = {}
        reasoning_order: list[str] = []
        response_order: list[str] = []
        reasoning_mode = "full"

        try:
            stream = self.agent.run(prompt, stream=True, session=self.session)
            async for chunk in stream:
                if not chunk.contents:
                    continue

                updated = False
                for index, content in enumerate(chunk.contents):
                    if content.type == "text_reasoning":
                        event_type = _content_event_type(content)
                        if event_type and event_type.startswith(
                            "response.reasoning_summary_text"
                        ):
                            if reasoning_mode != "summary":
                                reasoning_mode = "summary"
                                reasoning_seen.clear()
                                reasoning_order.clear()
                            key = f"reasoning_summary:{content.id or index}"
                        else:
                            if reasoning_mode == "summary":
                                continue
                            key = f"reasoning:{content.id or index}"

                        if key not in reasoning_seen:
                            reasoning_order.append(key)
                        incremental_text = _get_incremental_text(
                            reasoning_seen, key, content.text
                        )
                        if incremental_text:
                            updated = True
                    elif content.type == "text":
                        event_type = _content_event_type(content)
                        if event_type == "response.content_part.added":
                            continue

                        key = f"text:{content.id or index}"
                        if key not in response_seen:
                            response_order.append(key)
                        incremental_text = _get_incremental_text(
                            response_seen, key, content.text
                        )
                        if incremental_text:
                            updated = True
                    elif content.type == "usage":
                        usage_content = content

                if updated:
                    was_at_bottom = self._should_autoscroll()
                    assistant_widget.update(
                        _build_stream_renderable(
                            _join_stream_text(reasoning_seen, reasoning_order),
                            _join_stream_text(response_seen, response_order),
                        )
                    )
                    self._maybe_follow_or_notify(was_at_bottom)

            response = await stream.get_final_response()
            reasoning_text = _join_stream_text(reasoning_seen, reasoning_order)
            response_text = (
                _join_stream_text(response_seen, response_order) or response.text or ""
            )
            was_at_bottom = self._should_autoscroll()
            assistant_widget.update(_build_stream_renderable(reasoning_text, response_text))

            usage_details = response.usage_details or getattr(
                usage_content, "usage_details", None
            )
            turn_cost = _estimate_cost(usage_details)
            session_tally = _update_session_usage_tally(
                self.session, usage_details, turn_cost
            )
            self._usage_widget().update(
                _build_usage_renderable(usage_details, session_tally=session_tally)
            )
            self._maybe_follow_or_notify(was_at_bottom)
        except Exception as exc:
            error_text = Text(f"Error: {exc}", style=f"bold {CATPPUCCIN_MOCHA['red']}")
            was_at_bottom = self._should_autoscroll()
            assistant_widget.update(Padding(error_text, (0, 1)))
            self._maybe_follow_or_notify(was_at_bottom)
        finally:
            input_widget.disabled = False
            input_widget.focus()
            self.busy = False

    def action_conversation_page_up(self) -> None:
        self._conversation().scroll_page_up(animate=False)
        self._sync_scroll_state()

    def action_conversation_page_down(self) -> None:
        self._conversation().scroll_page_down(animate=False)
        self._sync_scroll_state()

    def action_conversation_end(self) -> None:
        self._conversation().scroll_end(animate=False, immediate=True, x_axis=False)
        self.pending_new_messages = False
        self._set_new_message_indicator(False)


def build_agent():
    agent_tools = [tools.run_shell_command, tools.apply_patch, tools.web_search]
    return OpenAIChatClient(
        model=config.azure.deployment,
        azure_endpoint=config.azure.endpoint,
        credential=AzureCliCredential(),
    ).as_agent(
        instructions="You are Percy, an autonomous AI assistant that can run tools. Reason over how best to approach the query and share your thought process.",
        tools=agent_tools,
        default_options={
            "reasoning": {"effort": "medium", "summary": "detailed"},
        },
    )


def main() -> None:
    agent = build_agent()
    memory_file = Path(config.sections["memory"]["session_file"]).expanduser().absolute()
    session = load_persistant_memory(memory_file)
    if session is None:
        session = agent.create_session()

    app = PercyApp(agent, session, memory_file)
    try:
        app.run()
    finally:
        persist_session_memory(session, memory_file)

    sys.exit()


if __name__ == "__main__":
    main()
