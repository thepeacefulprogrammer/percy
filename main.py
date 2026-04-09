import sys
import asyncio
import tomllib
from functools import lru_cache
from pathlib import Path
from decimal import Decimal
from agent_framework.openai import OpenAIChatClient
from agent_framework import Agent, AgentSession
from azure.identity import AzureCliCredential
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from ventures_agent_framework import config, tools
from memory import load_persistant_memory, persist_session_memory

console = Console()


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


def _build_stream_renderable(reasoning_text: str, response_text: str):
    renderables = []

    if reasoning_text.strip():
        renderables.append(Text("Reasoning", style="bold blue"))
        renderables.append(Markdown(reasoning_text))

    if response_text.strip():
        if renderables:
            renderables.append(Text())
        renderables.append(Text("Response", style="bold green"))
        renderables.append(Markdown(response_text))

    if not renderables:
        renderables.append(Text("Thinking...", style="dim"))

    return Group(*renderables)


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


def _format_int(value: int) -> str:
    return f"{value:,}"


def _format_cost(value: float | Decimal | None) -> str:
    if value is None:
        return "—"

    amount = float(value)
    return f"${amount:,.2f}"


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


def _build_usage_renderable(usage: dict | None, session_tally: dict | None = None):
    if not usage:
        return None

    token_limits = _load_local_config().get("token_limits", {})
    context_window = int(token_limits.get("context_window", 0) or 0)

    input_tokens = _usage_int(usage, "input_token_count")
    output_tokens = _usage_int(usage, "output_token_count")
    total_tokens = _usage_int(usage, "total_token_count")
    cached_input_tokens = _usage_int(usage, "openai.cached_input_tokens")
    reasoning_tokens = _usage_int(usage, "openai.reasoning_tokens")

    turn_cost = _estimate_cost(usage)
    remaining_context = (
        max(context_window - total_tokens, 0) if context_window else None
    )
    used_pct = (
        ((total_tokens / context_window) * 100)
        if context_window and total_tokens
        else 0
    )

    grid = Table.grid(expand=True)
    grid.add_column(style="bold")
    grid.add_column(justify="right")
    grid.add_row("📥 Input", _format_int(input_tokens))
    grid.add_row("📤 Output", _format_int(output_tokens))
    if reasoning_tokens:
        grid.add_row("🧠 Reasoning", _format_int(reasoning_tokens))
    if cached_input_tokens:
        grid.add_row("♻️ Cached input", _format_int(cached_input_tokens))
    grid.add_row("🧮 Total", _format_int(total_tokens))

    if context_window:
        grid.add_row(
            "🪟 Context",
            f"{_format_int(total_tokens)} / {_format_int(context_window)} ({used_pct:.1f}%)",
        )
        grid.add_row("💾 Remaining", _format_int(remaining_context or 0))

    grid.add_row("🪙 Turn cost", _format_cost(turn_cost))

    if session_tally:
        grid.add_row("", "")
        grid.add_row(
            "🔢 Session turns", _format_int(int(session_tally.get("turn_count", 0)))
        )
        grid.add_row(
            "∑ Session tokens",
            _format_int(int(session_tally.get("total_token_count", 0))),
        )
        grid.add_row(
            "💰 Session cost",
            _format_cost(float(session_tally.get("cost_usd", 0.0))),
        )

    return Panel(grid, title="Usage", border_style="dim")


async def handle_prompt(prompt: str, agent: Agent, session: AgentSession):
    console.print()

    usage_content = None
    reasoning_seen: dict[str, str] = {}
    response_seen: dict[str, str] = {}
    reasoning_order: list[str] = []
    response_order: list[str] = []
    reasoning_mode = "full"
    final_renderable = _build_stream_renderable("", "")

    stream = agent.run(prompt, stream=True, session=session)
    with Live(
        final_renderable,
        console=console,
        refresh_per_second=8,
        transient=True,
    ) as live:
        async for chunk in stream:
            if chunk.contents:
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
                    final_renderable = _build_stream_renderable(
                        _join_stream_text(reasoning_seen, reasoning_order),
                        _join_stream_text(response_seen, response_order),
                    )
                    live.update(final_renderable)

    response = await stream.get_final_response()

    reasoning_text = _join_stream_text(reasoning_seen, reasoning_order)
    response_text = (
        _join_stream_text(response_seen, response_order) or response.text or ""
    )
    usage_details = response.usage_details or getattr(
        usage_content, "usage_details", None
    )
    turn_cost = _estimate_cost(usage_details)
    session_tally = _update_session_usage_tally(session, usage_details, turn_cost)

    if reasoning_text or response_text:
        final_renderable = _build_stream_renderable(reasoning_text, response_text)
        console.print(final_renderable)

    usage_panel = _build_usage_renderable(usage_details, session_tally=session_tally)
    if usage_panel:
        console.print(usage_panel)

    console.print()


async def main():

    agent_tools = [tools.run_shell_command, tools.apply_patch, tools.web_search]

    agent = OpenAIChatClient(
        model=config.azure.deployment,
        azure_endpoint=config.azure.endpoint,
        credential=AzureCliCredential(),
    ).as_agent(
        instructions="You are Percy, an autonomous AI assistant that can run tools. Reason over how best to approach the query and share your thought process.",
        tools=agent_tools,
        default_options={"reasoning": {"effort": "medium", "summary": "detailed"}},
    )

    # Persistant Memory
    memory_file = (
        Path(config.sections["memory"]["session_file"]).expanduser().absolute()
    )
    session = load_persistant_memory(memory_file)
    if session is None:
        session = agent.create_session()

    while True:
        prompt = input("> ")
        if prompt in ["/quit", "/exit", "exit", "quit", "q"]:
            print("Goodbye")
            persist_session_memory(session, memory_file)
            sys.exit()
        await handle_prompt(prompt, agent, session)


if __name__ == "__main__":
    asyncio.run(main())
