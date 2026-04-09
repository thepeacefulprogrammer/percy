from pathlib import Path

from agent_framework import AgentSession
from rich.padding import Padding
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual import work
from textual.widgets import Input, Static

from percy.streaming import content_event_type, get_incremental_text, join_stream_text
from percy.theme import CATPPUCCIN_MOCHA
from percy.ui.renderers import build_stream_renderable, build_user_renderable
from percy.usage import build_usage_renderable, estimate_cost, update_session_usage_tally


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
        border: none;
        scrollbar-size-vertical: 0;
        scrollbar-color: {CATPPUCCIN_MOCHA['surface2']} {CATPPUCCIN_MOCHA['mantle']};
        scrollbar-color-hover: {CATPPUCCIN_MOCHA['overlay0']} {CATPPUCCIN_MOCHA['mantle']};
        scrollbar-color-active: {CATPPUCCIN_MOCHA['blue']} {CATPPUCCIN_MOCHA['mantle']};
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

    #prompt {{
        margin: 0 1;
    }}

    #usage {{
        height: 1;
        margin: 0 1 1 1;
        padding: 0 1;
        background: {CATPPUCCIN_MOCHA['mantle']};
        color: {CATPPUCCIN_MOCHA['overlay1']};
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
                Static(
                    "Welcome to Percy. Ask for code changes, debugging help, refactors, or research.",
                    classes="status-message",
                    id="empty-state",
                ),
                id="conversation",
            )
            yield Static(
                "new messages below · press End",
                id="new-message-indicator",
                classes="hidden",
            )
            yield Input(placeholder="Send a message…", id="prompt")
            yield Static(build_usage_renderable(None), id="usage")

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

        self.busy = True
        self._process_prompt(prompt)

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

    async def _append_widget(self, widget: Static) -> None:
        await self._remove_empty_state()
        await self._conversation().mount(widget)

    @work(exclusive=True)
    async def _process_prompt(self, prompt: str) -> None:
        input_widget = self._input_widget()
        input_widget.disabled = True
        was_at_bottom = self._should_autoscroll()

        user_widget = Static(build_user_renderable(prompt), classes="message user-message")
        assistant_widget = Static(
            build_stream_renderable("", ""), classes="message assistant-message"
        )

        await self._append_widget(user_widget)
        await self._append_widget(assistant_widget)
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
                        event_type = content_event_type(content)
                        if event_type and event_type.startswith("response.reasoning_summary_text"):
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
                        if get_incremental_text(reasoning_seen, key, content.text):
                            updated = True
                    elif content.type == "text":
                        event_type = content_event_type(content)
                        if event_type == "response.content_part.added":
                            continue

                        key = f"text:{content.id or index}"
                        if key not in response_seen:
                            response_order.append(key)
                        if get_incremental_text(response_seen, key, content.text):
                            updated = True
                    elif content.type == "usage":
                        usage_content = content

                if updated:
                    was_at_bottom = self._should_autoscroll()
                    assistant_widget.update(
                        build_stream_renderable(
                            join_stream_text(reasoning_seen, reasoning_order),
                            join_stream_text(response_seen, response_order),
                        )
                    )
                    self._maybe_follow_or_notify(was_at_bottom)

            response = await stream.get_final_response()
            reasoning_text = join_stream_text(reasoning_seen, reasoning_order)
            response_text = join_stream_text(response_seen, response_order) or response.text or ""
            was_at_bottom = self._should_autoscroll()
            assistant_widget.update(build_stream_renderable(reasoning_text, response_text))

            usage_details = response.usage_details or getattr(usage_content, "usage_details", None)
            turn_cost = estimate_cost(usage_details)
            session_tally = update_session_usage_tally(self.session, usage_details, turn_cost)
            self._usage_widget().update(
                build_usage_renderable(usage_details, session_tally=session_tally)
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
