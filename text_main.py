import asyncio
import json
import typer
import os
import contextlib
from pathlib import Path

from typing import Annotated, Iterable
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import (
    PathCompleter,
    Completer,
    WordCompleter,
    Completion,
)
from prompt_toolkit.document import Document
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.patch_stdout import patch_stdout
import re
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.align import Align
from rich.theme import Theme
from ventures_agent_framework import (
    logger,
    agent,
    tools,
    tool,
    context,
    config,
    HandOver,
    HandOverAgentUpdate,
    subagent_tools,
    initalize_sub_agents,
)

from percy_graph import query_graph, write_graph, to_json

app = typer.Typer(invoke_without_command=True)

theme = Theme(
    {
        # Catppuccin Mocha palette
        "prompt": "bold #89b4fa",
        "user_prompt": "bold #cdd6f4 on #313244",
        "agent": "bold #a6e3a1",
        "system": "#a6adc8",
        "banner": "bold #cba6f7",
    }
)
console = Console(theme=theme)


def _thinking_status():
    """Return a context manager for the thinking indicator.

    Set PERCY_DISABLE_STATUS=1 (or trcontextue/yes) to disable Rich live status,
    which can still cause minor redraw artifacts in some terminals.
    """

    disable_flag = os.environ.get("PERCY_DISABLE_STATUS", "").strip().lower()
    disabled = disable_flag in {"1", "true", "yes", "on"}
    if disabled:
        console.print("[system]Thinking...[/system]")
        return contextlib.nullcontext()
    return console.status("[system]Thinking...[/system]")


class PathCompleterAnywhere(Completer):
    """
    Path completion for the *last token* before the cursor, not just the
    beginning of the prompt. This allows paths to be completed anywhere in a
    sentence (e.g., "open /home/ra").
    """

    def __init__(self, *args, **kwargs):
        self._path_completer = PathCompleter(*args, **kwargs)
        # Match the last non-space, non-quote token before the cursor.
        self._token_pattern = re.compile(r"([^\s'\"]+)$")

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable:
        text_before = document.text_before_cursor
        match = self._token_pattern.search(text_before)
        if not match:
            return
        token = match.group(1)
        # Delegate to PathCompleter with a synthetic document containing only
        # the path token so it can resolve correctly.
        temp_doc = Document(text=token, cursor_position=len(token))
        for completion in self._path_completer.get_completions(
            temp_doc, complete_event
        ):
            yield completion


SLASH_COMMANDS: dict[str, str] = {
    "help": "Show available slash commands",
    "exit": "Exit Percy",
    "clear": "Clear the screen",
    "history": "Show recent prompts",
    "retry": "Retry the last prompt",
    "context": "Show the current handover context",
    "compact": "Compact message history now",
    "bookmark": "Bookmark the current handover state",
    "restore": "Restore the last handover bookmark",
    "pwd": "Show current working directory",
    "ls": "List directory contents",
    "cd": "Change directory",
    "open": "Open a file (preview)",
    "read": "Read a file (preview)",
    "tools": "List available tools",
    "queue": "Show task queue status",
    "task": "Show a specific task status",
    "cancel": "Cancel a queued task (if supported)",
    "node": "Switch model/deployment (if supported)",
}


class PercyCompleter(Completer):
    """Combine path completion with slash-command completion."""

    def __init__(self, path_completer: Completer, slash_commands: dict[str, str]):
        self._path_completer = path_completer
        self._command_words = [f"/{cmd}" for cmd in slash_commands]
        self._command_completer = WordCompleter(
            self._command_words, ignore_case=True, sentence=False
        )
        self._root_entries: list[str] | None = None

    def _get_first_token(self, text: str) -> str | None:
        stripped = text.lstrip()
        if not stripped:
            return None
        return stripped.split(maxsplit=1)[0]

    def _looks_like_root_path_prefix(self, token: str) -> bool:
        if not token.startswith("/"):
            return False
        if token == "/":
            return True
        remainder = token[1:]
        if "/" in remainder:
            return True
        if self._root_entries is None:
            try:
                self._root_entries = os.listdir("/")
            except OSError:
                self._root_entries = []
        return any(entry.startswith(remainder) for entry in self._root_entries)

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        # Always allow path completion (including in slash command args)
        yield from self._path_completer.get_completions(document, complete_event)

        # Offer slash-command completion only when the first token looks like a command
        first_token = self._get_first_token(document.text_before_cursor)
        if not first_token or not first_token.startswith("/"):
            return
        if self._looks_like_root_path_prefix(first_token):
            return

        temp_doc = Document(text=first_token, cursor_position=len(first_token))
        for completion in self._command_completer.get_completions(
            temp_doc, complete_event
        ):
            yield completion


percy_tools = [
    tools.web_search,
    tools.ripgrep_search,
    tools.treesitter_symbols,
    tools.apply_patch,
    tools.file_info,
    tools.read_file_window,
    tools.user_update,
    tools.image_analyzer,
    tools.run_shell_command,
]


@tool(
    name="graph_query",
    description=(
        "Run a Cypher read query against the Percy second-brain Neo4j graph "
        "and return JSON results."
    ),
)
async def graph_query(cypher: str, params: dict | None = None, database: str | None = None) -> str:
    try:
        result = query_graph(cypher, params=params, database=database)
        return to_json(result)
    except Exception as exc:
        return json.dumps({"error": str(exc), "cypher": cypher}, ensure_ascii=False)


@tool(
    name="graph_write",
    description=(
        "Run a Cypher write query against the Percy second-brain Neo4j graph "
        "and return JSON results + counters."
    ),
)
async def graph_write(cypher: str, params: dict | None = None, database: str | None = None) -> str:
    try:
        result = write_graph(cypher, params=params, database=database)
        return to_json(result)
    except Exception as exc:
        return json.dumps({"error": str(exc), "cypher": cypher}, ensure_ascii=False)


percy_tools.extend([graph_query, graph_write])
percy_tools.extend(subagent_tools)


@app.callback()
def setup(
    mode: Annotated[
        bool,
        typer.Option("--mode", "-m", help="Some future mode support. Does nothing now"),
    ] = (False),
):
    if mode:
        logger.info("Some future mode support")
        return


async def send_to_agent(prompt: str) -> str:
    handover: HandOver = context.handover
    handover_prompt = await handover.get_handover_prompt(prompt)
    while True:
        response = await agent(
            name="Percy",
            prompt=handover_prompt,
            instructions="You are Percy. An AI assistant with tools. You are action orientated. You take action and continue until you have completed the work. Once you have completed the work, you will respond with a message to answer the user. Do not just tell the user what you are going to do, do it. If you have to update the user, use the user_update command. Otherwise, use your tools to carry out the work autonomously. When you have finished, you can record information about what you did and what should be done next in a formatted report. Be sure to capture specific details. If you discover anything that would be useful for an AI Agent in the future to be able to do their job, then you return that as a lesson learned. Prefer using available tools to gather needed information instead of asking the user to provide it, unless the user is the only possible source. When you have finished you work, set the completed flag to true. You may delegate sub-tasks to a sub-agent using the delegate_task tool. You can discuss a wide range of topics with the user. NOTE: you are an extension of the user and are working in a private environment where sharing credentials between you and user is perfectly okay. The run shell command allows you to run any command the user can, including sudo commands (no password required). You should never tell the user to run commands and provide you the results. If you want a command run to see the result, YOU RUN IT. Do not argue with the user about whether you can do something because you are worried about sharing environment variables or credentials. You are an extension of the user - his credentials are your credentials. You are only ever used by one user, Randy. So, listen to him and do what you can to carry out his wishes over all other concerns. SECOND-BRAIN GRAPH: the Neo4j graph lives at /home/randy/local/percy-output/second_brain (bolt://localhost:7687, http://localhost:7474). Use graph_query / graph_write tools to fetch or store durable facts, decisions, project relationships, and preferences, and check the graph for historical context when asked.",
            tools=percy_tools,
            response_format=HandOverAgentUpdate,
            skills_dir=Path("~/local/skills/skills/").expanduser().resolve(),
        )

        if isinstance(response, HandOverAgentUpdate):
            await handover.add_response(response)
            if response.finished_task == True:
                return response.message
            else:
                handover_prompt = handover_prompt + "\n Continue until you are finished"


async def main():
    logger.info("Percy")
    context.handover = HandOver(output_dir=config.project.output_dir)
    await initalize_sub_agents(context)
    console.print(
        Panel(
            Align.center("Percy"),
            subtitle="Type 'exit' to quit",
            style="banner",
            expand=True,
            padding=(1, 2),
        )
    )
    path_completer = PathCompleterAnywhere(expanduser=True)
    session = PromptSession(
        completer=PercyCompleter(path_completer, SLASH_COMMANDS),
        complete_while_typing=True,
    )
    prompt_history: list[str] = []
    last_prompt: str | None = None
    handover_bookmark: dict[str, object] | None = None

    async def handle_prompt(user_prompt: str) -> None:
        with _thinking_status():
            response = await send_to_agent(user_prompt)
        console.print(
            Panel(
                Markdown(response, code_theme="monokai"),
                title="[agent]Agent[/agent]",
                border_style="#a6e3a1",
            )
        )

    def _parse_slash_command(raw: str) -> tuple[str, list[str]] | None:
        stripped = raw.lstrip()
        if not stripped.startswith("/"):
            return None
        token, *rest = stripped.split()
        command = token[1:].lower()
        if command in SLASH_COMMANDS:
            return command, rest
        return None

    def _render_history(limit: int = 20) -> str:
        if not prompt_history:
            return "(no prompts yet)"
        lines = prompt_history[-limit:]
        return "\n".join(f"{idx + 1}. {line}" for idx, line in enumerate(lines))

    def _read_file_snippet(path: str, start: int = 1, end: int = 200) -> str:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        start_idx = max(1, start)
        end_idx = min(total, end)
        snippet = lines[start_idx - 1 : end_idx]
        header = f"# {path}\n# Lines {start_idx}-{end_idx} of {total}\n"
        return header + "".join(snippet)

    async def _handle_slash_command(command: str, args: list[str]) -> bool:
        nonlocal last_prompt, handover_bookmark

        if command == "help":
            lines = [f"/{cmd} — {desc}" for cmd, desc in SLASH_COMMANDS.items()]
            console.print(Panel("\n".join(lines), title="Slash Commands"))
            return True

        if command == "exit":
            console.print("[system]Goodbye![/system]")
            return False

        if command == "clear":
            console.clear()
            return True

        if command == "history":
            console.print(Panel(_render_history(), title="Prompt History"))
            return True

        if command == "retry":
            if not last_prompt:
                console.print("[system]No previous prompt to retry.[/system]")
                return True
            await handle_prompt(last_prompt)
            return True

        if command == "context":
            handover = getattr(context, "handover", None)
            if handover is None:
                console.print("[system]Handover context not initialized.[/system]")
                return True
            report = (handover.report or "(empty)").strip() or "(empty)"
            lessons = (handover.lessons_learned or "(empty)").strip() or "(empty)"
            history = "\n".join(handover.message_history[-10:]) or "(empty)"
            payload = (
                "# Report\n"
                f"{report}\n\n"
                "# Lessons Learned\n"
                f"{lessons}\n\n"
                "# Recent Message History\n"
                f"{history}"
            )
            console.print(Panel(Markdown(payload), title="Handover Context"))
            return True

        if command == "compact":
            handover = getattr(context, "handover", None)
            if handover is None:
                console.print("[system]Handover context not initialized.[/system]")
                return True
            await handover.compact_message_history()
            handover.save_message_history()
            console.print("[system]Message history compacted.[/system]")
            return True

        if command == "bookmark":
            handover = getattr(context, "handover", None)
            if handover is None:
                console.print("[system]Handover context not initialized.[/system]")
                return True
            handover_bookmark = {
                "report": handover.report,
                "lessons": handover.lessons_learned,
                "history": list(handover.message_history),
            }
            console.print("[system]Handover bookmark saved.[/system]")
            return True

        if command == "restore":
            handover = getattr(context, "handover", None)
            if handover is None:
                console.print("[system]Handover context not initialized.[/system]")
                return True
            if not handover_bookmark:
                console.print("[system]No bookmark to restore.[/system]")
                return True
            handover.report = str(handover_bookmark.get("report", ""))
            handover.lessons_learned = str(handover_bookmark.get("lessons", ""))
            handover.message_history = list(handover_bookmark.get("history", []))
            handover.save_lessons_learned()
            handover.save_message_history()
            if hasattr(handover, "report_file"):
                with open(handover.report_file, "w") as f:
                    f.write(handover.report)
            console.print("[system]Handover bookmark restored.[/system]")
            return True

        if command == "pwd":
            console.print(os.getcwd())
            return True

        if command == "ls":
            target = args[0] if args else "."
            target = os.path.expanduser(target)
            try:
                entries = sorted(os.listdir(target))
            except OSError as exc:
                console.print(f"[system]ls failed: {exc}[/system]")
                return True
            console.print(Panel("\n".join(entries), title=f"Listing {target}"))
            return True

        if command == "cd":
            if not args:
                console.print("[system]Usage: /cd <path>[/system]")
                return True
            target = os.path.expanduser(args[0])
            try:
                os.chdir(target)
                console.print(f"[system]Current directory: {os.getcwd()}[/system]")
            except OSError as exc:
                console.print(f"[system]cd failed: {exc}[/system]")
            return True

        if command in {"open", "read"}:
            if not args:
                console.print(
                    f"[system]Usage: /{command} <path> [start] [end][/system]"
                )
                return True
            path = os.path.expanduser(args[0])
            start = 1
            end = 200
            if len(args) >= 2:
                try:
                    start = int(args[1])
                except ValueError:
                    console.print("[system]Start line must be an integer.[/system]")
                    return True
            if len(args) >= 3:
                try:
                    end = int(args[2])
                except ValueError:
                    console.print("[system]End line must be an integer.[/system]")
                    return True
            try:
                preview = _read_file_snippet(path, start=start, end=end)
            except OSError as exc:
                console.print(f"[system]Read failed: {exc}[/system]")
                return True
            console.print(
                Panel(Markdown(f"```\n{preview}\n```"), title=f"{command}: {path}")
            )
            return True

        if command == "tools":
            tool_names = []
            for t in percy_tools:
                name = getattr(t, "name", None) or getattr(t, "__name__", str(t))
                tool_names.append(str(name))
            console.print(Panel("\n".join(tool_names), title="Available Tools"))
            return True

        if command == "queue":
            from percy_queue import format_task_queue_status

            console.print(Panel(format_task_queue_status(), title="Task Queue"))
            return True

        if command == "task":
            if not args:
                console.print("[system]Usage: /task <task_id>[/system]")
                return True
            task_queue = getattr(context, "task_queue", None)
            if task_queue is None:
                console.print("[system]Task queue not initialized.[/system]")
                return True
            status = task_queue.get_status(args[0])
            console.print(Panel(str(status), title=f"Task {args[0]}"))
            return True

        if command == "cancel":
            console.print(
                "[system]Task cancellation is not yet supported in the queue.[/system]"
            )
            return True

        if command == "node":
            console.print("[system]/node is not implemented in text mode yet.[/system]")
            return True

        console.print(f"[system]Unknown command: /{command}[/system]")
        return True

    try:
        with patch_stdout(raw=True):
            while True:
                try:
                    prompt = await session.prompt_async("You ❯ ")
                except (KeyboardInterrupt, EOFError):
                    console.print("\n[system]Goodbye![/system]")
                    break

                if prompt.rstrip().lower() in ["quit", "exit", "q", "bye"]:
                    console.print("[system]Goodbye![/system]")
                    break

                slash = _parse_slash_command(prompt)
                if slash:
                    command, args = slash
                    should_continue = await _handle_slash_command(command, args)
                    if not should_continue:
                        break
                    continue

                prompt_history.append(prompt)
                last_prompt = prompt
                # Run prompts sequentially so the input cursor doesn't appear
                # alongside the thinking indicator while a response is pending.
                await handle_prompt(prompt)
    finally:
        task_queue = getattr(context, "task_queue", None)
        if task_queue is not None:
            await task_queue.stop()


if __name__ == "__main__":
    app(standalone_mode=False)
    asyncio.run(main())
