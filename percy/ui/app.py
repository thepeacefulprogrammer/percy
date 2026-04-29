from pathlib import Path

from agent_framework import AgentSession
from rich.padding import Padding
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.suggester import SuggestFromList
from textual import work
from textual.widgets import Input, OptionList, Static

from percy.streaming import content_event_type, get_incremental_text, join_stream_text
from percy.theme import CATPPUCCIN_MOCHA
from percy.ui.path_input import PathCompletionInput, PathCompletionMenu
from percy.ui.renderers import build_stream_renderable, build_user_renderable
from percy.usage import (
    build_usage_renderable,
    estimate_cost,
    update_session_usage_tally,
)

SLASH_COMMAND_SUGGESTIONS = [
    "/caveman",
    "/caveman lite",
    "/caveman full",
    "/caveman ultra",
    "/caveman off",
    "/normal",
    "/normal mode",
    "/stop caveman",
    "/quit",
]


class PercyApp(App[None]):
    CSS = f"""
    Screen {{
        background: {CATPPUCCIN_MOCHA["base"]};
        color: {CATPPUCCIN_MOCHA["text"]};
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
        scrollbar-color: {CATPPUCCIN_MOCHA["surface2"]} {CATPPUCCIN_MOCHA["mantle"]};
        scrollbar-color-hover: {CATPPUCCIN_MOCHA["overlay0"]} {CATPPUCCIN_MOCHA["mantle"]};
        scrollbar-color-active: {CATPPUCCIN_MOCHA["blue"]} {CATPPUCCIN_MOCHA["mantle"]};
    }}

    #new-message-indicator {{
        height: 1;
        margin: 0 1;
        color: {CATPPUCCIN_MOCHA["blue"]};
        content-align: center middle;
    }}

    #new-message-indicator.hidden {{
        display: none;
    }}

    #composer {{
        layout: vertical;
        height: auto;
    }}

    #prompt {{
        margin: 0 1;
    }}

    #path-completion-menu {{
        margin: 0 1;
        background: {CATPPUCCIN_MOCHA["mantle"]};
        border: tall {CATPPUCCIN_MOCHA["surface2"]};
        color: {CATPPUCCIN_MOCHA["text"]};
    }}

    #path-completion-menu:focus {{
        border: tall {CATPPUCCIN_MOCHA["mauve"]};
    }}

    #usage {{
        height: 1;
        margin: 0 1 1 1;
        padding: 0 1;
        background: {CATPPUCCIN_MOCHA["mantle"]};
        color: {CATPPUCCIN_MOCHA["overlay1"]};
    }}

    Input {{
        background: {CATPPUCCIN_MOCHA["mantle"]};
        color: {CATPPUCCIN_MOCHA["text"]};
        border: tall {CATPPUCCIN_MOCHA["surface2"]};
    }}

    Input:focus {{
        border: tall {CATPPUCCIN_MOCHA["mauve"]};
    }}

    .message {{
        width: 1fr;
    }}

    .user-message {{
        background: {CATPPUCCIN_MOCHA["surface0"]};
        color: {CATPPUCCIN_MOCHA["text"]};
        margin: 1 0 1 0;
    }}

    .assistant-message {{
        margin: 0 0 1 0;
    }}

    .status-message {{
        color: {CATPPUCCIN_MOCHA["overlay0"]};
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
        self.active_skill_name: str | None = None
        self.active_skill_level: str | None = None

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
            with Container(id="composer"):
                yield PathCompletionInput(
                    placeholder="Send a message…",
                    id="prompt",
                    suggester=SuggestFromList(
                        SLASH_COMMAND_SUGGESTIONS, case_sensitive=False
                    ),
                )
                yield PathCompletionMenu(id="path-completion-menu")
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

        command_handled = await self._handle_local_command(prompt)
        if command_handled:
            return

        self.busy = True
        self._process_prompt(prompt)

    async def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        if event.option_list.id != "path-completion-menu":
            return

        input_widget = self._input_widget()
        if isinstance(input_widget, PathCompletionInput):
            await input_widget.action_submit()
            input_widget.focus()
        event.stop()

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

    def _completion_menu_widget(self) -> PathCompletionMenu:
        return self.query_one("#path-completion-menu", PathCompletionMenu)

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

    async def _append_status_message(self, message: str) -> None:
        was_at_bottom = self._should_autoscroll()
        await self._append_widget(Static(message, classes="status-message"))
        self._maybe_follow_or_notify(was_at_bottom)

    def _set_prompt_placeholder(self) -> None:
        input_widget = self._input_widget()
        if self.active_skill_name == "caveman":
            level = self.active_skill_level or "full"
            input_widget.placeholder = f"Send a message… [caveman:{level}]"
        else:
            input_widget.placeholder = "Send a message…"

    async def _handle_local_command(self, prompt: str) -> bool:
        stripped = prompt.strip()
        lower = stripped.lower()

        if lower in {"/normal", "/normal mode", "/stop caveman", "/caveman off"}:
            self.active_skill_name = None
            self.active_skill_level = None
            self._set_prompt_placeholder()
            await self._append_status_message("Caveman mode off.")
            return True

        if lower.startswith("/caveman"):
            parts = stripped.split(maxsplit=1)
            level = "full"
            if len(parts) > 1 and parts[1].strip():
                level = parts[1].strip()

            self.active_skill_name = "caveman"
            self.active_skill_level = level
            self._set_prompt_placeholder()
            await self._append_status_message(f"Caveman mode on ({level}).")
            return True

        return False

    def _build_effective_prompt(self, prompt: str) -> str:
        if self.active_skill_name != "caveman":
            return prompt

        level = self.active_skill_level or "full"
        return (
            "Activate and use the skill 'caveman' for this response. "
            "Use load_skill('caveman') if needed. "
            f"Use intensity level '{level}'. "
            "This mode is active until the user says to stop.\n\n"
            f"User request:\n{prompt}"
        )

    @work(exclusive=True)
    async def _process_prompt(self, prompt: str) -> None:
        input_widget = self._input_widget()
        input_widget.disabled = True
        was_at_bottom = self._should_autoscroll()
        effective_prompt = self._build_effective_prompt(prompt)

        user_widget = Static(
            build_user_renderable(prompt), classes="message user-message"
        )
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
            if self.session.service_session_id:
                self.session.service_session_id = None
            stream = self.agent.run(
                effective_prompt,
                stream=True,
                session=self.session,
                options={"store": False},
            )
            async for chunk in stream:
                if not chunk.contents:
                    continue

                updated = False
                for index, content in enumerate(chunk.contents):
                    if content.type == "text_reasoning":
                        event_type = content_event_type(content)
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
            response_text = (
                join_stream_text(response_seen, response_order) or response.text or ""
            )
            was_at_bottom = self._should_autoscroll()
            assistant_widget.update(
                build_stream_renderable(reasoning_text, response_text)
            )

            usage_details = response.usage_details or getattr(
                usage_content, "usage_details", None
            )
            turn_cost = estimate_cost(usage_details)
            session_tally = update_session_usage_tally(
                self.session, usage_details, turn_cost
            )
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
